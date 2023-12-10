"""
GPS C/A code generation.

References:
- https://natronics.github.io/blag/2014/gps-prn/
- https://en.wikipedia.org/wiki/Code-division_multiple_access
"""


import numpy as np


class GpsCaCode:
    PERIOD_TIME = 1e-3   # [s] duration of one C/A code period
    PERIOD_CHIPS = 1023  # 1023 chips per code period -> 1.023 Mcps
    CARRIER_FREQUENCY = 1.57542e9  # [Hz] GPS L1 carrier frequency, 1575.42 MHz

    def __init__(self, prn_id, sampling_dt) -> None:
        """
        Generates sampling data of C/A code sequence period of PRN prn_id 
        at the specified sampling rate.

        Args:
        - sampling_dt: sampling delta time, ie. time step per sample
        - prn_id: PRN number of required code
        """
        ca_code = np.array(self.generate_ca_code(prn_id))
        ca_code = 2*ca_code - 1  # shift code levels from 0/1 to -1/+1

        self.period_samples = int(self.PERIOD_TIME / sampling_dt)               # samples per period
        self.chip_samples = self.PERIOD_TIME / self.PERIOD_CHIPS / sampling_dt  # samples per chip
        self.chip_time = self.PERIOD_TIME / self.PERIOD_CHIPS                   # time per chip
        self.sampling_dt = sampling_dt

        # In order to get the C/A codes as sampled by the SDR sampling rate, we need to resample the C/A codes for the this rate;
        # within `period_samples``, we sample `PERIOD_CHIPS` chips of the C/A code. We have to scale accordingly:
        sampled_chip_index = np.arange(0, self.period_samples) * self.PERIOD_CHIPS/self.period_samples
        sampled_chip_index = np.mod(np.round(sampled_chip_index).astype('int'), self.PERIOD_CHIPS)

        self.samples = ca_code[sampled_chip_index]

    @staticmethod
    def generate_ca_code(id):
        """
        Generates GPS L1 C/A codes for given PRN id.

        Implemented in https://natronics.github.io/blag/2014/gps-prn/
        See also reference table https://dsp.stackexchange.com/questions/52810/gps-coarse-acquisition-prn-codes
                -> e.g. PRN 20: G2 taps 4+7 -> outputs (1)715

        Returns: C/A code sequence with 1023 chips length
        """

        def shift(register, feedback, output):
            """
            GPS Shift Register.
            
            Parameters:
            - feedback:  which positions to use as feedback (1 indexed)
            - output: which positions are output (1 indexed)
            
            Returns: Output of shift register
            """
        
            # calculate output
            out = [register[i-1] for i in output]
            if len(out) > 1:
                out = sum(out) % 2
            else:
                out = out[0]
                
            # update register
            fb = sum([register[i-1] for i in feedback]) % 2  # get summed feedback taps
            for i in reversed(range(len(register[1:]))):     # shift right
                register[i+1] = register[i]
            register[0] = fb
            
            return out

        # list of feedback taps for G2 register for PRN code nr 1-37
        tap_list = [
            [2,6], [3,7], [4,8], [5,9], [1,9], [2,10], [1,8], [2,9], [3,10], [2,3], [3,4], 
            [5,6], [6,7], [7,8], [8,9], [9,10], [1,4], [2,5], [3,6], [4,7], [5,8], [6,9], 
            [1,3], [4,6], [5,7], [6,8], [7,9], [8,10], [1,6], [2,7], [3,8], [4,9], [5,10], 
            [4,10], [1,7], [2,8], [4,10] ]
        tap = tap_list[id-1]
        
        # init registers with ones
        G1 = [1 for i in range(10)]
        G2 = [1 for i in range(10)]

        # generate full sequence of 1023 chips of C/A code
        ca = []
        for i in range(1023):
            g1 = shift(G1, [3,10], [10])         # feedback 3,10, output 10
            g2 = shift(G2, [2,3,6,8,9,10], tap)  # feedback 2,3,6,8,9,10, e.g. PRN 1 output 2,6 
            ca.append((g1 + g2) % 2)
            
        return ca
    
