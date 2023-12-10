"""
GPS L1 signal acquisition.


References:
- https://gnss-sdr.org/docs/sp-blocks/acquisition/
- https://gssc.esa.int/navipedia/index.php/Baseband_Processing
"""

import numpy as np
import matplotlib.pyplot as plt

from .gps_ca_code import GpsCaCode
from .sdr_wave import SdrWave


class Acquisition:
    def __init__(self, prn_id: int, sampling_dt) -> None:
        self.ca_code = GpsCaCode(prn_id, sampling_dt)      

    def search(self, baseband: SdrWave,
               delta_freq_step=500, delta_freq_range=(-5000,5000)):
        """
        Searches for SV with PRN id in baseband signal and reports optimal demodulation parameters.
        Optimizes correlation of baseband signal with given SV PRN id L1 C/A code 
        by optimizing C/A code delay $tau$ and Doppler frequency. This search is performed 
        individually for every code period (1ms).

        Arguments:
            - delta_freq_* [Hz]: range and step size used during scanning for best signal
              in case of inaccurate clock, adjust range accordingly, up to 100kHz

        Returns: 
            An ndarray with optimal correlation parameters at every code period (1ms)
              - time [s]: code period timestamp since beginning of recording,
                  in local receiver time
              - correlator: complex-valued peak correlation, normalized to baseband RMS
              - delay [s]: code time delay for peak correlation magnitude
              - delta_freq [Hz]: (Doppler) freq. shift for peak correlation magnitude
            for every code period.
        """
        
        assert self.ca_code.sampling_dt == baseband.sampling_dt, 'inconsistent sampling rates.'
        
        doppler_frequencies = np.arange(delta_freq_range[0], delta_freq_range[1], delta_freq_step)

        ca_code = self.ca_code.samples                      # C/A code to be correlated
        ca_code_fft = np.fft.fft(ca_code).conj()            # precalculate FFT for faster correlation
        t = np.arange(0, len(ca_code)) * baseband.sampling_dt # time axis of one code period [s]
        ca_code_rms = np.sqrt(np.sum(np.abs(ca_code**2)))   # root mean square of code

        num_code_periods = int(baseband.duration() / self.ca_code.PERIOD_TIME)
        
        self._scan_data = np.ndarray((len(ca_code), len(doppler_frequencies)))  # full scan data for debugging
            
        acquisition_data = np.ndarray((num_code_periods,), 
                                        dtype=[('time', np.float64),
                                               ('correlator', np.complex128),
                                               ('delay', np.float64),
                                               ('delta_freq', np.float64),
                                              ]
                                       )

        # loop over time, i.e. code periods
        for i in range(num_code_periods):
            # select time window in signal of one C/A code period length (1ms)
            baseband_windowed = baseband.samples[(i) * len(ca_code) : (i+1) * len(ca_code)]
            baseband_windowed_rms = np.sqrt(np.sum(np.abs(baseband_windowed**2)))
            
            # search best Doppler frequency
            corr_peak_magnitude = 0
            for j, doppler_freq in enumerate(doppler_frequencies):
                # correct Doppler shift
                baseband_windowed_corrected = baseband_windowed * np.exp(-1j * 2*np.pi * doppler_freq * t)

                # fast periodic correlation
                corr_amplitude = np.fft.ifft(np.fft.fft(baseband_windowed_corrected) * ca_code_fft) \
                    / ca_code_rms / baseband_windowed_rms

                if i == num_code_periods - 1: # collect full scan for last time period for debug purposes
                    self._scan_data[:, j] = np.abs(corr_amplitude)**2 

                # search best tau at given Doppler freq.
                peak_position = np.argmax(abs(corr_amplitude))
 
                # new magnitude record, i.e. new tau/Doppler optimum found?
                if abs(corr_amplitude[peak_position])**2 > corr_peak_magnitude:  
                    corr_peak_magnitude = abs(corr_amplitude[peak_position])**2
                    corr_peak_amplitude = corr_amplitude[peak_position]
                    corr_peak_offset = peak_position * baseband.sampling_dt
                    corr_peak_doppler = doppler_freq
            
            acquisition_data['time'][i] = i * self.ca_code.PERIOD_TIME  # time [s]
            acquisition_data['correlator'][i] = corr_peak_amplitude
            acquisition_data['delay'][i] = corr_peak_offset
            acquisition_data['delta_freq'][i] = corr_peak_doppler
        
        return acquisition_data
    

def plot_acquisition_data(correlation_data, show_magnitude=True, show_offset=True, show_phase=True, show_doppler=True):
    num_plots = show_magnitude + show_phase + show_offset + show_doppler
    fig, ax = plt.subplots(num_plots, 1, sharex=True, 
                            figsize=(15, 2.5*num_plots), squeeze=False)
    ax = ax[:, 0]
    
    ax[-1].set_xlabel('Receiver Time $t$ [s]') 
    t = correlation_data['time']
    #ax[0].set_title(f'Correlation between Signal and L1 C/A Code of PRN {prn_id}')
    i = 0
    if show_magnitude:
        ax[i].plot(t, abs(correlation_data['correlator'])**2, 'b')
        ax[i].set_ylabel('Correlation Magnitude')
        i += 1
    if show_phase:
        ax[i].plot(t, np.angle(correlation_data['correlator']), 'g')
        ax[i].set_ylabel('Correlation Phase [rad]')
        i += 1
    if show_offset:
        ax[i].plot(t, correlation_data['delay'] * 1e6, 'r')
        ax[i].set_ylabel('Code Delay $\\tau$ [$\mu$s]')
        i += 1
    if show_doppler:
        ax[i].plot(t, correlation_data['delta_freq'], 'm')
        ax[i].set_ylabel('Doppler Shift $\Delta f_{D}$ [Hz]')
    
    return ax
    