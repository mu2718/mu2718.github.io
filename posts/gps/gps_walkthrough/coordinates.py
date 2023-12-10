"""
Tools for coordinate transformations.

Reference:
- https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
- https://www.telesens.co/2017/07/17/calculating-position-from-raw-gps-data/
"""

import numpy as np

from .ephemeris import EARTH_OMEGADOT


def ecef_to_ellipsoid(x, y, z):
    """
    From GPS ECEF (earth-centered earth-fixed) coordinates, 
    calculates ellipsoid longitude, latitude and height.
    https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates

    Arguments: GPS ECEF coordinates (x,y,z) in meters
    Returns: longitude, latitude in degrees, height in meters
    """
    
    # WGS ellipsoid parameters
    a = 6378137 # [m]
    f = 1/298.257
    e = np.sqrt(2*f - f**2)
    
    l = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
     
    phi = np.arctan2(z, p * (1 - e**2))
    N = a / (1 - (e * np.sin(phi))**2)**0.5
    h = 0
    h_old = h + 10
    while abs(h - h_old) > 0.01:
        phi = np.arctan2(z, p * (1 - e**2 * (N/(N+h))))
        N = a / (1 - (e * np.sin(phi))**2)**0.5
        h_old = h
        h = p / np.cos(phi) - N
    
    return phi * 180/np.pi, l * 180/np.pi, h


def ecef_to_eci(ecef_point, t):
    """ 
    Eearth-centered earth-fixed ECEF to to earth-centerd inertial ECI frame coordinates at time t.
    """
    sl = np.sin(EARTH_OMEGADOT * t)
    cl = np.cos(EARTH_OMEGADOT * t)
    M = np.matrix([[cl, -sl, 0],
                   [sl, cl, 0],
                   [0, 0, 1]])
            
    return np.matmul(M, ecef_point)


def ellipsoid_to_ecef(phi, l, h):
    """ 
    https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates

    Parameters: longitude, latitude, height
    Returns:   ECEF coordinates (x,y,z)
    """
    
    # WGS ellipsoid parameters
    a = 6378137 # [m]
    f = 1/298.257
    e = np.sqrt(2*f - f**2)
    
    l = l * np.pi/180
    phi = phi * np.pi/180
    
    N = a / (1 - (e * np.sin(phi))**2)**0.5

    x = (N + h) * np.cos(phi) * np.cos(l)
    y = (N + h) * np.cos(phi) * np.sin(l)
    z = (N * (1-e**2) + h) * np.sin(phi)
    
    return x,y,z


def ecef_to_local(ecef_point, phi, l):
    """ 
    ECEF to local (ENU, east-north-up) coordinates. doesn't take into account ellispoid l, phi definition.
    https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU
    """
    
    l = l * np.pi/180
    phi = phi * np.pi/180
    sl = np.sin(l)
    cl = np.cos(l)
    sp = np.sin(phi)
    cp = np.cos(phi)
    M = np.matrix([[-sl, cl, 0],
                   [-sp*cl, -sp*sl, cp],
                   [cp*cl, cp*sl, sp]])
            
    return np.matmul(M, ecef_point)