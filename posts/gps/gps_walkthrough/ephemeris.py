"""
Calculation of orbital position of satellites using the ephemeris parameters

References:
- https://www.gps.gov/technical/icwg/IS-GPS-200L.pdf
- https://www.telesens.co/2017/07/17/calculating-position-from-raw-gps-data/
"""

import numpy as np
import matplotlib.pyplot as plt


# GPS constants
GPS_WEEK_SECONDS = 604800       # [s] seconds in GPS week

# physics constants as specified by IS-GPS-200L
SPEED_OF_LIGHT = 2.99792458e8   # [m/s] speed of light in vacuum
EARTH_MU = 3.986005e14          # [m^3/s^2] earth gravitation constant WGS84
EARTH_OMEGADOT = 7.2921151467e-5  # [rad/s] earth rotation rate WGS84
EARTH_F = -4.442807633e-10      # Relativistic correction constant [s/m**0.5], see IS-GPS-200
EARTH_RADIUS = 6378137


def sat_position(t, ephemeris, harmonic_correct=True):
    """
    Get satellite position at GPS times t using the telemetry ephemeris data.
     
    Result: Position x,y,z in earth-centered earth-fixed (ECEF) coordinate system.
    """
    
    # here we follow algorithm of IS-GPS-200L, table 20-IV, p. 102
    # we omit the "k" (Kepler) index in our naming

    A = ephemeris['sqrtA']**2
    n0 = np.sqrt(EARTH_MU/A**3)
    
    tk = t - ephemeris['t_oe']
    if tk[0] < -GPS_WEEK_SECONDS/2:  # make tk be in [-GPS_WEEK_SECONDS/2, GPS_WEEK_SECONDS/2]
        tk += GPS_WEEK_SECONDS
    elif tk[0] > GPS_WEEK_SECONDS/2:
        tk -= GPS_WEEK_SECONDS
    
    n = n0 + ephemeris['dn']
    M = ephemeris['M_0'] + n * tk
    e = ephemeris['e']
    
    # solve Kepler's equation
    E = M
    for i in range(10):
        E = E + (M - E + e * np.sin(E)) / (1 - e * np.cos(E))      
    #print(M, '==', E - e * np.sin(E)) # eq. solved?
    
    nu = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
    
    Phi = nu + ephemeris['omega']
    
    # second harmonic orbit corrections
    if harmonic_correct:
        delta_u = ephemeris['C_us'] * np.sin(2*Phi) + ephemeris['C_uc'] * np.cos(2*Phi)
        delta_r = ephemeris['C_rs'] * np.sin(2*Phi) + ephemeris['C_rc'] * np.cos(2*Phi)
        delta_i = ephemeris['C_is'] * np.sin(2*Phi) + ephemeris['C_ic'] * np.cos(2*Phi)
    else:
        delta_u = delta_r = delta_i = 0
    
    u = Phi + delta_u
    r = A * (1 - e * np.cos(E)) + delta_r 
    i = ephemeris['i_0'] + delta_i + ephemeris['I_dot'] * tk
    
    x_prime = r * np.cos(u)
    y_prime = r * np.sin(u)
    
    Omega = ephemeris['Omega_0'] \
            + (ephemeris['Omega_dot'] - EARTH_OMEGADOT) * tk \
            - EARTH_OMEGADOT * ephemeris['t_oe']
    
    x = x_prime * np.cos(Omega) - y_prime * np.sin(Omega) * np.cos(i)
    y = x_prime * np.sin(Omega) + y_prime * np.cos(Omega) * np.cos(i)
    z = y_prime * np.sin(i)
    
    return np.vstack((x, y, z)).transpose()


def sat_clock_correction(t_sv, clock_telemetry, ephemeris=None):
    """
    Correct satellite (space vehicle SV) clock reading t_sv, as sent by telemetry, to get accurate GPS time. 
    Uses correction coefficients from LNAV messages (subframe 1),
    see "20.3.3.3.3.1 User Algorithm for SV Clock Correction", p. 95 in <c

    Args:
    - t_sv: Array of times as reported by satellite in telemetry messages
    - clock_telemetry: Telemetry clock correction data
    - ephemeris_correction: Optional. Ephemeris data for correction of relativistic time dilation

    Returns: Array of corrected time.
    """
    delta_t = t_sv - clock_telemetry['t_oc']
    if delta_t < -GPS_WEEK_SECONDS/2:  # make delta_t be in [-GPS_WEEK_SECONDS/2, GPS_WEEK_SECONDS/2]
        delta_t += GPS_WEEK_SECONDS
    elif delta_t > GPS_WEEK_SECONDS/2:
        delta_t -= GPS_WEEK_SECONDS
        
    delta_t_sv = clock_telemetry['a_f0'] \
        + clock_telemetry['a_f1'] * delta_t + clock_telemetry['a_f2'] * delta_t**2 \
        -  clock_telemetry['T_GD']  # see "20.3.3.3.3.2 L1 or L2 Correction"
    t_gps = t_sv - delta_t_sv
    
    # Relativistic correction (approx 10-50ns)
    if ephemeris:
        A = ephemeris['sqrtA']**2
        n0 = np.sqrt(EARTH_MU/A**3)

        tk = t_gps - ephemeris['t_oe']
        if tk < -GPS_WEEK_SECONDS/2:  # make tk be in [-GPS_WEEK_SECONDS/2, GPS_WEEK_SECONDS/2]
            tk += GPS_WEEK_SECONDS
        elif tk > GPS_WEEK_SECONDS/2:
            tk -= GPS_WEEK_SECONDS

        n = n0 + ephemeris['dn']
        M = ephemeris['M_0'] + n * tk
        e = ephemeris['e']

        # solve Kepler's equation
        E = M
        for i in range(10):
            E = E + (M - E + e * np.sin(E)) / (1 - e * np.cos(E)) 

        delta_t_relativistic = EARTH_F * e * ephemeris['sqrtA'] * np.sin(E)
        t_gps = t_gps - delta_t_relativistic
    
    return t_gps