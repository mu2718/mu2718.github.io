"""
Measurement of GPS observables.

References:
- https://gssc.esa.int/navipedia/index.php/Data_Demodulation_and_Processing
- https://gssc.esa.int/navipedia/index.php?title=GNSS_Basic_Observables
- https://gnss-sdr.org/docs/sp-blocks/observables/
- https://www.gps.gov/technical/icwg/IS-GPS-200L.pdf
"""

import numpy as np

from .gps_ca_code import GpsCaCode
from .ephemeris import sat_clock_correction


# GPS constants
CA_CODE_PERIOD = GpsCaCode.PERIOD_TIME  # [s] duration of one C/A code period

# physics constants as specified by GPS
SPEED_OF_LIGHT = 2.99792458e8   # [m/s] speed of light in vacuum


def pseudo_ranges(telemetry, tracking, observation_interval):
    """
    Measures the pseudo-range, send and receive time for the GPS signal of a satellite (space vehicle SV).

    Arguments:
    - telemetry: SV telemetry data, for accurate time of sending of each C/A code
    - tracking: SV signal tracking data, for accurate time of reception of each C/A code (see Tracking class)
    - observation_interval: Time interval to calculate pseudo-ranges

    Returns: List of send and receive times (seconds) and pseudo-range (meters) for each interval.
      These are yet uncorrected for local receiver time errors.
    """
    
    # get the most recent clock telemetry data
    clock_telemetry = [subframe['clock'] for subframe in telemetry if 'clock' in subframe.keys()][-1]
    
    # get all ephemeris data
    ephemeris_subframes = [subframe['ephemeris'] for subframe in telemetry if 'ephemeris' in subframe.keys()]
    ephemeris = {}
    for subframe in ephemeris_subframes:
        ephemeris.update(subframe)
        
    # determine send time of the first received C/A code in our signal recording:
    # time (TOW) data in a subframe indicates the SV time of the next subframe, i.e. the one being sent 6s later
    # see "20.3.3.2 Handover Word (HOW)", p. 88.
    first_subframe_time_sv = telemetry[0]['time_of_week'] - 6       # [s] send time of first subframe, SV clock
    first_subframe_start_code_index = telemetry[0]['start_symbol']  # index of C/A code corresponding to first subframe start

    # calculating back from first detected subframe to the very first C/A code in our signal, i.e. code at index 0
    first_code_time_sv = first_subframe_time_sv - first_subframe_start_code_index * CA_CODE_PERIOD  # [s] SV send time of first received C/A code 

    observation_point_num = int(max(tracking['time_code_start']) // observation_interval)
    pseudo_ranges = np.zeros((observation_point_num,), 
                             dtype=[('receive_time', np.float64),  # [s] receiver observation time
                                    ('send_time_gps', np.float64),  # [s] GPS send time of the signal received at the observation time
                                    ('pseudo_range', np.float64),            # [m] pseudo-distance 
                                   ])

    for i in range(observation_point_num):
        # define a desired observation time at the receiver, receiver clock
        receive_time = i * observation_interval

        # find a C/A code which contains this observation time
        for code_index, code_start_receive_time in enumerate(tracking['time_code_start']):
            if code_start_receive_time +  CA_CODE_PERIOD >= receive_time:
                break  # found
        else:
            print('C/A code not found.')
            break    

        # determine send time of the start of this found C/A code, in SV clock
        code_send_time_sv = first_code_time_sv + code_index * CA_CODE_PERIOD

        # get the send time of the signal received at the observation time, SV clock
        receive_time_delta = receive_time - code_start_receive_time  # observation time relativ to code start, < 1ms, receiver clock
        receive_code_period = tracking['time_code_start'][code_index+1] - tracking['time_code_start'][code_index]
        send_time_sv_delta = CA_CODE_PERIOD * receive_time_delta / receive_code_period  # interpolate corresponding relative send time
        send_time_sv = code_send_time_sv + send_time_sv_delta   # send time of signal at observation time, SV clock

        # Note: code_start_receive_time, i.e. the reception time of the code, 
        # was determined by the delay lock loop (DLL) with sub-sample accuracy via
        # correlation. code_send_time_sv, on the other hand, is made available by
        # the telemetry messages as explicit value.

        # correct SV clock error to get accurate GPS time
        send_time_gps = sat_clock_correction(send_time_sv, clock_telemetry, ephemeris)

        # Calculate pseudo-range with time difference between sending and receiving
        rho = SPEED_OF_LIGHT * (receive_time - send_time_gps)
        # Note that these values are not actual distances since our receiver clock
        # is not properly set and starts at 0s at the beginning of the recording.
        # This receiver clock bias will be taken care of later on.

        pseudo_ranges[i]['pseudo_range'] = rho
        pseudo_ranges[i]['send_time_gps'] = send_time_gps
        pseudo_ranges[i]['receive_time'] = receive_time
        
    return pseudo_ranges
