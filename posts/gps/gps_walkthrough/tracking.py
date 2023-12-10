"""
GPS L1 signal tracking.


References:
- https://gnss-sdr.org/docs/sp-blocks/tracking/
- https://gssc.esa.int/navipedia/index.php/Digital_Signal_Processing
- https://gssc.esa.int/navipedia/index.php/Tracking_Loops
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import tqdm

from .sdr_wave import SdrWave
from .gps_ca_code import GpsCaCode


class PIController:
    """Implements a real-time PI controller with low-pass filtered input."""

    def __init__(self, p_coeff, ki_coeff, filter_order, filter_bw, sampling_rate) -> None:
        # PI controller coefficients and internal states
        self._ki = ki_coeff
        self._p = p_coeff
        self._integrator = 0
        self._discriminator = 0
        self._discriminator_filtered = 0
        self.output = 0

        # state of a digital filter used for input filtering
        self._filter_b, self._filter_a = \
            scipy.signal.butter(filter_order, filter_bw, btype='low', 
                                analog=False, fs=sampling_rate)
        self._filter_zi = np.zeros([max(len(self._filter_a), len(self._filter_b)) - 1])

    def update(self, discriminator: float) -> float:
        """
        Update PI output one time step with new sample of discriminator input.
        Returns: updated output value of PI controller
        """
        filter_out, self._filter_zi = \
            scipy.signal.lfilter(self._filter_b, self._filter_a, 
                                 [discriminator], zi=self._filter_zi)
        discriminator_filtered = filter_out[0]

        self._discriminator = discriminator
        self._discriminator_filtered = discriminator_filtered

        self._integrator += self._ki * discriminator_filtered
        self.output = self._p * discriminator_filtered + self._integrator  # P + I
        return self.output


class FrequencyLockedLoop():
    def __init__(self, start_frequency, loop_rate,
                 ki, filter_bw) -> None:
        self.frequency = start_frequency
        self.phase = 0
                
        # setup a I controller
        self._controller = PIController(p_coeff=0, ki_coeff=ki, 
                                       filter_order=3, filter_bw=filter_bw,
                                       sampling_rate=loop_rate)
        self._controller._integrator = start_frequency
        self._last_amplitude = 0
        self._loop_rate = loop_rate
        
    def _discriminator(self, amplitude: complex):
        """Determines phase change of amplitude compared to its last value."""
        PI = np.real([self._last_amplitude, amplitude])
        PQ = np.imag([self._last_amplitude, amplitude])
        cross = PI[0]*PQ[1] - PQ[0]*PI[1]  # cross product of phasors
        dot = PI[0]*PI[1] + PQ[0]*PQ[1]    # dot product of phasors

        self._last_amplitude = amplitude

        # change in phase from last to current symbol, yields -180°..180°. 
        delta_phi = np.arctan2(cross, dot)
        if abs(delta_phi) > np.pi/2:  # Ignore phase change due to PLL: 
            delta_phi = 0             # larger than 90° = left half-plane -> probably bit transition
        delta_f = delta_phi * self._loop_rate / (2*np.pi)  # convert phase change to frequency
        return delta_f

    def update(self, sample: complex):
        x = self._discriminator(sample)
        self.frequency = self._controller.update(x)
        self.phase +=  2*np.pi * self.frequency / self._loop_rate  # total accumulated phase
        return self.frequency
    

class PhaseLockedLoop():
    def __init__(self, loop_rate,
                 ki, p, filter_bw) -> None:
        self.phase = 0
        self.controller = PIController(p_coeff=p, ki_coeff=ki, 
                                       filter_order=3, filter_bw=filter_bw,
                                       sampling_rate=loop_rate)
        
    def _discriminator(self, amplitude: complex):
        """Costas loop discriminator, returns -90°..90°. Compatible with BPSK."""
        PI = np.real(amplitude)
        PQ = np.imag(amplitude)
        return - np.arctan( PQ / PI )
 
    def update(self, sample: complex):
        x = self._discriminator(sample)
        self.phase = self.controller.update(x)
        return self.phase
    

class DelayLockedLoop():
    """Delay locked-loop (DLL): Control loop for locking to delay parameter."""

    def __init__(self, start_tau, ca_code: GpsCaCode, loop_rate,
                 early_late_space_chips,  # [chips] tau spacing between early and late correlator 
                 ki, filter_bw) -> None:
        self._early_late_space_chips = early_late_space_chips
        early_late_space_samples = int(np.round(early_late_space_chips * ca_code.chip_samples))
        assert early_late_space_samples > 0, \
            "sampling rate too low for given early/late correlator spacing! Try increase sample rate or spacing."

        self._ca_code_early = np.roll(ca_code.samples, -early_late_space_samples)
        self._ca_code_late = np.roll(ca_code.samples, early_late_space_samples)
        self._ca_code = ca_code.samples
        self._chip_time = ca_code.chip_time

        # gnss-sdr, order 2 bw 2Hz, no oscil for k_I=0.01, loses lock fast for my bad clock
        # alternative set order 2 and 10-20Hz to follow instable clock
        # more stable at order 1 to follow faster with lower filter delay
        # dll_filter = DspFilterIncremental(1, dll_bw, 1/integration_time)
        # prepare C/A codes for correlation
        self.controller = PIController(p_coeff=0, ki_coeff=ki, 
                                       filter_order=1, filter_bw=filter_bw,
                                       sampling_rate=loop_rate)
        self.controller._integrator = start_tau
        self.delay = start_tau
        
    def _discriminator(self, signal_samples: np.ndarray):
        """Estimates position of correlation peak and returns correction of delay value."""
        correlator_prompt = 1 / len(self._ca_code) * sum( signal_samples * self._ca_code)
        correlator_early = 1 / len(self._ca_code) * sum( signal_samples * self._ca_code_early )
        correlator_late = 1 / len(self._ca_code) * sum( signal_samples * self._ca_code_late )
        # measure peak center position (-1 early, 0 prompt, +1 late)
        peak_center = ((-1) * abs(correlator_early)**2  + (1) * abs(correlator_late)**2) \
                        / (abs(correlator_early)**2 + abs(correlator_prompt)**2 + abs(correlator_late**2))
        delay_correction = peak_center * self._early_late_space_chips * self._chip_time  # tau correction value [s]   
        return delay_correction     

    def update(self, signal_samples: np.ndarray, doppler_frequency=None):
        """
        Update DLL with new time step.
        Arguments:
        - signal_samples: samples of next C/A code period
        - doppler_frequency:  current Doppler frequency for Doppler-guided DLL
        Returns: updated delay value.
        """
        x = self._discriminator(signal_samples)

        # derive change in tau shift (= code stretching) from Doppler frequency
        if doppler_frequency:
            doppler_tau = - doppler_frequency / GpsCaCode.CARRIER_FREQUENCY * GpsCaCode.PERIOD_TIME
            self.controller._integrator += doppler_tau

        self.delay = self.controller.update(x)
        return self.delay
    

class Tracking:
    """GPS L1 tracking using frequency-, phase-, and delay-locking using control loops."""

    def __init__(self, prn_id, sampling_dt, 
                delta_freq_start, delay_start,
                fll_ki=0.05, fll_bw=35.0,
                pll_ki=0.03, pll_p=0.5, pll_bw=20.0,
                dll_ki=0.001, dll_bw=2.0, dll_guided=True, 
                dll_early_late_space_chips=0.5) -> None:
        """
        Initialize tracking object
        
        Arguments:
        - prn_id: C/A code PRN id to be tracked
        - sampling_dt: sampling time step [s]
        - delta_freq_start : initial freq. shift to bootstrap tracking (e.g. from acquisition) [Hz]
        - delay_start: initial code delay to bootstrap tracking (e.g. from acquisition) [s]
        - ***_ki: integrator coefficient of DLL/PLL/FLL
        - ***_bw: low-pass bandwidth limit of control loop feedback [Hz]
        - dll_early_late_space_chips: DLL time spacing between early-late correlator [chips]
        """
        self.ca_code = GpsCaCode(prn_id, sampling_dt)

        loop_rate = 1 / GpsCaCode.PERIOD_TIME  # rate of feedback loop iterations [Hz]
        self.fll = FrequencyLockedLoop(delta_freq_start, loop_rate,
                                       ki=fll_ki, filter_bw=fll_bw)
        self.pll = PhaseLockedLoop(loop_rate, ki=pll_ki, p=pll_p, filter_bw=pll_bw)
        self.dll = DelayLockedLoop(delay_start, self.ca_code, loop_rate,
                                   early_late_space_chips=dll_early_late_space_chips,
                                   ki=dll_ki, filter_bw=dll_bw)
        self._dll_guided = dll_guided

    def track(self, baseband: SdrWave, progress=False):
        """
        Track C/A code in baseband signal. Yields for every code period the optimal code
        parameters (delay tau, Doppler freq.) 
        """

        assert baseband.sampling_dt == self.ca_code.sampling_dt, "inconsistent sampling rates."
        sampling_dt = self.ca_code.sampling_dt

        ca_code = self.ca_code.samples
        t = np.arange(0, len(ca_code)) * sampling_dt        # time axis of one code period [s]
        ca_code_rms = np.sqrt(np.sum(np.abs(ca_code**2)))   # root mean square of code

        num_code_periods = int(baseband.duration() / GpsCaCode.PERIOD_TIME)

        # prepare output array, one entry for each time step
        tracking_data = np.zeros((num_code_periods,), 
                                 dtype=[('time_code_start', np.float64), 
                                        ('delay_used', np.float64), 
                                        ('correlator', np.complex128),
                                        ('delay', np.float64),
                                        ('delta_freq', np.float64),
                                        ('phi', np.float64),
                                        ('discriminator_f', np.float64),
                                        ('discriminator_f_filtered', np.float64),
                                        ('discriminator_phi', np.float64),
                                        ('discriminator_phi_filtered', np.float64),
                                        ('discriminator_tau', np.float64),
                                        ('discriminator_tau_filtered', np.float64),
                                       ])  

        # loop over time steps, i.e. code periods
        for i in tqdm.tqdm(range(num_code_periods), disable=not progress):
            last_tau = self.dll.delay
            last_phi = self.pll.phase
            last_delta_freq = self.fll.frequency
            last_delta_freq_phase = self.fll.phase

            time_receiver = i * GpsCaCode.PERIOD_TIME  # local receiver time at current code period
            
            # starting sample index of C/A code in signal, as determined by DLL for last code period
            code_start_sample = int(round((time_receiver + last_tau) / sampling_dt))

            # select time window in signal where on C/A code period is expected
            baseband_windowed = baseband.samples[code_start_sample : (code_start_sample + len(ca_code))]
            if len(baseband_windowed) != len(ca_code):  # end of signal samples
                tracking_data[i:] = np.nan
                break 
            baseband_windowed_rms = np.sqrt(np.sum(np.abs(baseband_windowed**2)))  # root mean square
   
            # correction of Doppler shift as determined by FLL for last code period
            baseband_windowed = baseband_windowed * np.exp(- 1j * 2*np.pi * last_delta_freq * t 
                                                           - 1j * last_delta_freq_phase)

            # calculate RMS normalized correlation of baseband signal with the C/A code
            corr_amplitude = np.sum(baseband_windowed * ca_code) / ca_code_rms / baseband_windowed_rms
            
            # update loops feedbacks: frequency (FLL), delay (DLL) and phase-locked loops (PLL)
            self.fll.update(corr_amplitude)
            self.dll.update(baseband_windowed, self.fll.frequency if self._dll_guided else None)
            self.pll.update(corr_amplitude * np.exp(1j * last_phi))

            # final demodulated/locked phase output: use PLL output of current code period 
            corr_amplitude_phase_locked = corr_amplitude * np.exp(1j * self.pll.phase)

            # store main tracking data
            tracking_data['time_code_start'][i] = time_receiver + self.dll.delay    # tau delay is relative to receiver time
            tracking_data['correlator'][i] = corr_amplitude_phase_locked
            tracking_data['phi'][i] = self.pll.phase
            tracking_data['delay'][i] = self.dll.delay
            tracking_data['delay_used'][i] = code_start_sample * sampling_dt - time_receiver # used tau discretized in samples
            tracking_data['delta_freq'][i] = self.fll.frequency

            # debug and analysis data
            tracking_data['discriminator_tau'][i] = self.dll.controller._discriminator
            tracking_data['discriminator_tau_filtered'][i] = self.dll.controller._discriminator_filtered
            tracking_data['discriminator_f'][i] = self.fll._controller._discriminator
            tracking_data['discriminator_f_filtered'][i] = self.fll._controller._discriminator
            tracking_data['discriminator_phi'][i] = self.pll.controller._discriminator
            tracking_data['discriminator_phi_filtered'][i] = self.pll.controller._discriminator_filtered
            
        return tracking_data
    

def plot_tracking_data(tracking_data, prn=None, show_magnitude=True, show_tau=True, show_phi=True, show_doppler=True, show_discriminators=False,
                    show_qi=True, show_cn0=False):
    # plot correlations
    num_plots = (show_magnitude + show_phi + show_tau + show_doppler) * (1 + show_discriminators) + show_qi + show_cn0
    
    fig, ax = plt.subplots(num_plots, 1, sharex=True, figsize=(15, 3*num_plots), squeeze=False)
    ax = ax[:, 0]
    
    t = tracking_data['time_code_start']
    
    ax[0].set_title(f' CDMA-BPSK Demodulation of L1-C/A Code' + f' of PRN {prn}' if prn else '')
    
    i = 0
    if show_magnitude:
        ax[i].plot(t, abs(tracking_data['correlation_amplitude'])**2, 'b')
        ax[i].set_ylabel('Correlation Magnitude')
        i += 1
        if show_discriminators:
            ax[i].scatter(t, np.angle(tracking_data['correlation_amplitude']), marker='.')
            ax[i].set_ylabel('Correlation Phase [rad]')
            i += 1

    if show_qi:
        ax[i].plot(t, np.real(tracking_data['correlation_amplitude']) 
                    / np.abs(tracking_data['correlation_amplitude']), 'c')
        ax[i].set_ylabel('$Q_I / |Q|$')
        i += 1
    
    if show_tau:
        ax[i].set_ylabel('Code Delay $\\tau$ [$\mu$s]')
        ax[i].plot(t, tracking_data['delay'] * 1e6, 'r', label='DLL estimate')
        ax[i].plot(t, tracking_data['delay_used'] * 1e6, 'b', label='discretized (used)')
        ax[i].plot(t, (tracking_data['delay_used'] + tracking_data['discriminator_tau']) * 1e6, 'g',
                label='discretized + discriminator signal (true value)', alpha=0.3)
        ax[i].legend()
        i += 1
        
        if show_discriminators:
            ax[i].plot(t, tracking_data['discriminator_tau'] * 1e6)
            ax[i].plot(t, tracking_data['discriminator_tau_filtered'] * 1e6)
            ax[i].set_ylabel('Discriminator tau [$\mu$s]')
            i += 1
        
    if show_doppler:
        ax[i].plot(t, tracking_data['delta_freq'], 'm')
        ax[i].set_ylabel('Doppler Shift $\Delta f_{D}$ [Hz]')
        i += 1
        if show_discriminators:
            ax[i].plot(t, tracking_data['discriminator_f'])
            ax[i].plot(t, tracking_data['discriminator_f_filtered'])
            ax[i].set_ylabel('Discriminator f')
            i += 1
        
    if show_phi:
        ax[i].plot(t, tracking_data['phi'])
        ax[i].set_ylabel('Phi')
        i += 1
        if show_discriminators:
            ax[i].plot(t, tracking_data['discriminator_phi'])
            ax[i].plot(t, tracking_data['discriminator_phi_filtered'])
            ax[i].set_ylabel('Discriminator Phi')
            i += 1
    
    ax[-1].set_xlabel('Receiver Time $t$ [s]') 
    
    return ax


def plot_constellation_diagram(data, norm=False, title=None):
    samples = data / (abs(data) if norm else 1)
    
    fig, ax = plt.subplots(1, 2, figsize=(20,5))
    if title:
        fig.suptitle(title)
    ax[0].scatter(np.real(samples), np.imag(samples), marker='.', alpha=0.3, c=range(len(samples)))
    ax[0].set_aspect('equal', 'box')
    ax[0].set_xlabel('In-phase I')
    ax[0].set_ylabel('Quadrature Q')

    #plt.title('Sample Value Histogram') #, PLL $k_P$ = 1, $k_I$ = 0.05')
    ax[1].hist(np.real(data), bins=100, label='In-Phase I')
    ax[1].hist(np.imag(data), bins=100, label='Quadrature Q', alpha=0.3)
    ax[1].set_ylabel('Occurrences')
    ax[1].set_ylabel('Amplitude')

    ax[1].legend()