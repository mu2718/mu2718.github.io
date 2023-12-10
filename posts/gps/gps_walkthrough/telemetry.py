"""
GPS L1 LNAV navigation message decoder.

References:
- https://gnss-sdr.org/docs/sp-blocks/telemetry-decoder/
- https://gssc.esa.int/navipedia/index.php/Data_Demodulation_and_Processing
- https://gssc.esa.int/navipedia/index.php?title=GPS_Navigation_Message
- https://www.gps.gov/technical/icwg/IS-GPS-200L.pdf
"""

import numpy as np
import re


# GPS L1 LNAV navigation message constants
PSYMBOLS_PER_BIT = 20       # a bit is encoded in 20 repetitions of a pseudo symbol (= phase of code periods)
WORD_SIZE = 30              # a word consists of 30 bits
WORDS_PER_SUBFRAME = 10     # a subframe consists of 10 words
SUBFRAME_SIZE = WORDS_PER_SUBFRAME * WORD_SIZE


def bit_synchronize(pseudosymbols):
    """
    Synchronizes the stream of pseudo-symbols to bits and decodes bits.
    
    Returns: Stream of decoded bits; index of first pseudo-symbol encoding the first fully received bit
    """

    bit_sync_histogram = np.zeros(PSYMBOLS_PER_BIT)
    for bit_start in range(PSYMBOLS_PER_BIT):
        # test symbol error rate for all following bits in symbols data
        for i in range(bit_start, len(pseudosymbols), PSYMBOLS_PER_BIT):
            bit_symbols = pseudosymbols[i: i + PSYMBOLS_PER_BIT] 
            if len(bit_symbols) < PSYMBOLS_PER_BIT:  # bit incomplete?
                break  
            # calculate symbol errors in one bit. All should be equal if correct 
            # start bit was chosen and RF channel is free of noise.
            sym_sum = np.sum(bit_symbols)  
            sym_errors = sym_sum if sym_sum < PSYMBOLS_PER_BIT//2 else PSYMBOLS_PER_BIT - sym_sum  # assume majority is correct
            bit_sync_histogram[bit_start] += sym_errors

    bit_sync_histogram /= len(pseudosymbols) / PSYMBOLS_PER_BIT
    bit_start = np.argmin(bit_sync_histogram)
    print(f'Bits: synced, avg. pseudo-symbol errors per bit: {bit_sync_histogram[bit_start]:.3f}')

    # calculate bits, assume majority of symbols are correct
    psymbols_bit_aligned = pseudosymbols[bit_start:]
    bit_count = len(psymbols_bit_aligned) // PSYMBOLS_PER_BIT
    psymbols_bit_aligned = psymbols_bit_aligned[:bit_count * PSYMBOLS_PER_BIT]
    bit_symbols = np.reshape(psymbols_bit_aligned, (bit_count, PSYMBOLS_PER_BIT))
    bits = np.array(np.round(np.mean(bit_symbols, axis=1)), dtype='int')
    
    return bits, bit_start


def subframe_hamming_decode(subframe_bits):
    """
    Decode every word of the given subframe bits using the used Hamming code, 
    see "20.3.5.2 User Parity Algorithm" in GPS specs, p. 136.

    Arguments:
      - subframe_bits: stream of bits of a subframe, ie. 300 bits

    Returns: Parity check result, decoded subframe
    """

    subframe_bits = subframe_bits[:SUBFRAME_SIZE].copy()
        
    D30_old = D29_old = 0  # subframe always ends with two zero parity bits
    for word_index in range(WORDS_PER_SUBFRAME):
        word = subframe_bits[word_index * WORD_SIZE : (word_index+1) * WORD_SIZE]
        
        word[0:24] ^= D30_old
        D25 = (D29_old + sum(word[[0, 1, 2, 4, 5, 9, 10, 11, 12, 13, 16, 17, 19, 22]])) % 2
        D26 = (D30_old + sum(word[[1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23]])) % 2
        D27 = (D29_old + sum(word[[0, 2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21]])) % 2
        D28 = (D30_old + sum(word[[1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22]])) % 2
        D29 = (D30_old + sum(word[[0, 2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23]])) % 2
        D30 = (D29_old + sum(word[[2, 4, 5, 7, 8, 9, 10, 12, 14, 18, 21, 22, 23]])) % 2
        D30_old = word[29]
        D29_old = word[28]
        
        if not all([D25 == word[24], D26 == word[25], D27 == word[26],
                    D28 == word[27], D29 == word[28], D30 == word[29]]):
            return False, None
    return True, subframe_bits


def subframe_synchronize(bits):
    """
    Synchronizes bit stream with subframes.

    Returns:
     - offset of first subframe in bit array and 
     - the input bit array with corrected polarity.
    """
    
    def subframe_find_start(bits):
        """
        Tries to find the first start of a subframe in the given bit stream and
        the correct polarity of bits.

        Returns: 
          - offset, s.t. bits[offset:] starts at the first identified subframe,
          - number of subframes with valid parity check
        """
     
        # find potential candidates of starting frame: preamble in TLM word
        bits_string = "".join([str(b) for b in bits])
        frame_sync_offsets = np.array([m.start() for m in re.finditer('10001011', bits_string)])  

        offset_max_valid = -1
        counter_max_valid = -1
        # check all potential candidates using Hamming code parity, take best
        for offset in frame_sync_offsets:
            valid_counter = 0
            for subframe_offset in range(offset, len(bits), SUBFRAME_SIZE):
                subframe = bits[subframe_offset : subframe_offset + SUBFRAME_SIZE]
                try:
                    parity_is_valid, _ = subframe_hamming_decode(subframe)
                except:
                    continue
                if parity_is_valid: 
                    valid_counter += 1

            if valid_counter > counter_max_valid:
                offset_max_valid = offset
                counter_max_valid = valid_counter

        return offset_max_valid, counter_max_valid
 
    # since we don't know bit polarity at demodulation stage, we try both
    sync_offset1, valid_subframes1 = subframe_find_start(bits)
    sync_offset2, valid_subframes2 = subframe_find_start(1 - bits)

    if sync_offset1 == sync_offset2 == -1:
        raise Exception("No subframe sync found.")
   
    print(f"Subframes: synced, found {int(len(bits)/SUBFRAME_SIZE)},"
          f" valid {max(valid_subframes1, valid_subframes2)}")
    
    if valid_subframes1 > valid_subframes2:
        sync_offset = sync_offset1
    else:
        bits = bits.copy()
        bits = 1 - bits
        sync_offset = sync_offset2  
        
    return bits, sync_offset


def subframe_dissect(bits, subframe_start_bit_offset, bit_start_psymbol_offset):  
    """
    Dissects the bit stream and returns some of the subframe data as dictionaries. 
    
    Arguments:
      - subframe_sync_bit_offset: index of bit in bits of first subframe.
      - bit_sync_psymbol_offset:  index of pseudo-symbol of first bit in bits array.
          Used to calculate absolute symbol index references for all messages.

    References: [1] GPS specification IS-GPS-200L, https://www.gps.gov/technical/icwg/IS-GPS-200L.pdf
    """
    
    def bits2uint(bits):  
        """Convert bits to unsigned integer."""
        #assert len(bits) in [8, 16, 17, 14, 32, 10, 22,3,24], f'bit len {len(bits)} is strange'
        return int("".join(str(bit) for bit in bits), 2)
    
    def bits2int(bits): 
        """Convert bits to signed integer using two's complement encoding."""
        val = bits2uint(bits)
        val = val - (bits[0] << len(bits))  # perform two's complement if bits[0] == 1
        return val
        
    subframes_decoded = []
    for bit_offset in range(subframe_start_bit_offset, len(bits), SUBFRAME_SIZE):
        # current subframe bits
        subf = bits[bit_offset : bit_offset + SUBFRAME_SIZE]
        
        try:
            parity_ok, subf = subframe_hamming_decode(subf)
        except:
            continue
        if not parity_ok:
            print('found invalid parity. Omitting subframe.')
            continue  # abort current subframe

        # now dissect data structures as given in GPS specification:
        
        # HOW word: see "20.3.3.2 Handover Word (HOW)" in [1], p. 89
        subframe_id = bits2uint(subf[49:52])  
        data = {"start_symbol": int(bit_start_psymbol_offset + bit_offset * PSYMBOLS_PER_BIT),  # index of first symbol in subframe
                "subframe_id":  int(subframe_id),
                "integrity":    int(subf[22]),
                "time_of_week": bits2uint(subf[30:47]) * 6,  # [s] TOW value
               }
        
        if subframe_id == 1:                         
            clock = {
                'week_number': bits2uint(subf[60:70]),   # GPS week number mod 1024
                'sv_health':   int(subf[76]),  # space vehicle health status, 0 = OK 
                
                # space vehicle clock correction information,
                # see "20.3.3.3.3 User Algorithms for Subframe 1 Data", p. 94
                'T_GD':  bits2int(subf[196:204]) * 2**(-31),  # 
                't_oc':  bits2uint(subf[218:234]) * 2**4,  
                'a_f2':  bits2int(subf[240:248]) * 2**(-55),
                'a_f1':  bits2int(subf[248:264]) * 2**(-43),
                'a_f0':  bits2int(subf[270:292]) * 2**(-31),
                }
            data['clock'] = clock   

        elif subframe_id == 2:                     
            ephemeris = {  
                # Ephemeris data of space vehicle.
                # see "20.3.3.4.1 Content of Subframes 2 and 3", p. 99 and 101    
                # and "Figure 20-1. Data Format (sheet 2 of 11)", p. 78
                'C_rs': bits2int(subf[68:84]) * 2**(-5),                                            # harmonic orbit correction of orbit radius, sine
                'dn': bits2int(subf[90:106]) * 2**(-43) * np.pi,                                    # mean angular motion correction
                'M_0': bits2int(np.concatenate((subf[106:114], subf[120:144]))) * 2**(-31) * np.pi, # mean anomaly
                'C_uc': bits2int(subf[150:164]) * 2**(-29),                                         # harmonic orbit correction
                'e': bits2uint(np.concatenate((subf[166:174], subf[180:204]))) * 2**(-33),          # excentrity
                'C_us': bits2int(subf[210:226]) * 2**(-29),                                         # harmonic orbit correction
                'sqrtA': bits2uint(np.concatenate((subf[226:234], subf[240:264]))) * 2**(-19),      # squre-root of semi-major axis
                't_oe': bits2uint(subf[270:286]) * 2**4,                                            # ref. time for ephemeris calc.
                }
            data['ephemeris'] = ephemeris     

        elif subframe_id == 3:
            ephemeris = {  
                # see "20.3.3.4.1 Content of Subframes 2 and 3", p. 99 and 101
                # and "Figure 20-1. Data Format (sheet 3 of 11)", p. 79
                'C_ic': bits2int(subf[60:76]) * 2**(-29),                                           # harmonic orbit correction
                'Omega_0': bits2int(np.concatenate((subf[76:84], subf[90:114]))) * 2**(-31) * np.pi,  # longitude of ascending node of orbit
                'C_is': bits2int(subf[120:136]) * 2**(-29),                                         # harmonic orbit correction
                'i_0': bits2int(np.concatenate((subf[136:144], subf[150:174]))) * 2**(-31) * np.pi, # inclination angle
                'C_rc': bits2int(subf[180:196]) * 2**(-5),                                          # harmonic orbit correction of orbit radius, cosine
                'omega': bits2int(np.concatenate((subf[196:204], subf[210:234]))) * 2**(-31) * np.pi, # arg. of perigee
                'Omega_dot': bits2int(subf[240:264]) * 2**(-43) * np.pi,                            # rate of right ascension
                'I_dot': bits2int(subf[278:292]) * 2**(-43) * np.pi,                                # rate of inclination angle
                }
            data['ephemeris'] = ephemeris        
        
        subframes_decoded += [data]
        
    print(f'Dissection: {len(subframes_decoded)} subframes decoded.')
              
    return subframes_decoded 