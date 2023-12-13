"""
Solve for GPS receiver position and time given measured observables.

References:
- https://www.telesens.co/2017/07/17/calculating-position-from-raw-gps-data/
- https://gnss-sdr.org/docs/sp-blocks/pvt/
"""


from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

from .ephemeris import SPEED_OF_LIGHT, EARTH_OMEGADOT
from . import observables
from . import coordinates


def position_fix(pseudo_ranges, sv_positions):
    """
    Given observed pseudo-range and calculated satellite positions, determines
    best fitting position and GPS time.

    Using a least-squared method, position and time is searched to get a minimal
    deviation of the model (calculated pseudo-range) to the observation
    (measured pseudo-range).

    Returns: Position array of (x,y,z,t) space and time (GPS ECEF) coordinates 
    """
    # use first PRN to determine number of time steps
    time_steps = len(pseudo_ranges[list(pseudo_ranges)[0]])

    start_coordinates = [0,0,0,0]  # start search at earth center, we must be close :)
    solution_coordinates = np.zeros((time_steps, 4))
    for ti in range(time_steps):
        o = optimize.least_squares(_position_fix_error_function, 
                                   start_coordinates,
                                   x_scale=(1e7, 1e7, 1e7, 1e5),  # scales are 10000 km and days
                                   args=(ti, pseudo_ranges, sv_positions), 
                                  )
        start_coordinates = o.x
        solution_coordinates[ti] = o.x

    # report GPS receive time, i.e. receive time corrected by clock bias solution
    #solution_coordinates[:,3] += receive_time

    return solution_coordinates


def _position_fix_error_function(x, time_index, pseudo_ranges, satellite_positions, correct_sagnac=True):
    """
    Error function to be optimized by position/time solver. 

    Returns: Array of differences between actually measured pseudo-ranges and the 
    expected ranges based on the position and time estimate x. 
    Optimal estimate x is given if sum(pos_error(x)**2) is minimal.

    Arguments:
      x:             x[0:3] receiver position estimate in meters, 
                     x[3] receiver time estimate in seconds.
      time_index:    index to select point in time of the following arguments,
      pseudo_ranges: time series array of measured pseudo-ranges of all PRNs,
      satellite_positions: time series of sat. positions of all PRNs.
    """    
    receiver_position_estimate = x[0:3]
    
    residuals = np.zeros((len(satellite_positions),))
    for i, prn in enumerate(satellite_positions):
        receiver_clock_bias_estimate = x[3] - pseudo_ranges[prn][time_index]['receive_time']

        # correct receiver clock bias of pseudo range
        pseudo_range = pseudo_ranges[prn][time_index]['pseudo_range'] \
                       - SPEED_OF_LIGHT * receiver_clock_bias_estimate 

        # Correct for Sagnac effect, see Eq.(6) in https://gnss-sdr.org/docs/sp-blocks/pvt/ 
        # -> changes position on ground approx 20m towards west.
        if correct_sagnac:
            pseudo_range += - EARTH_OMEGADOT / SPEED_OF_LIGHT * \
                        ( satellite_positions[prn][time_index][0] * receiver_position_estimate[1] \
                         - satellite_positions[prn][time_index][1] * receiver_position_estimate[0] )
        
        # range as expected by current position estimate = euclidean distance
        distance = np.sqrt(np.sum((satellite_positions[prn][time_index] - receiver_position_estimate)**2))

        # residual = difference between measured vs. expected range
        residuals[i] = pseudo_range - distance
    return residuals


def plot_positions_orbits(satellite_positions, solution_coordinates, pseudo_ranges,
                          ax=None, 
                          telemetry=None, orbit_time=12*3600, inertial_frame=False):
    if not ax: 
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        ax.set_proj_type('ortho')
        
    # plot earth in spherical approximation
    R = 6378137 / 1000 # [km] earth radius
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 25),  # wireframe for every 15°
                       np.linspace(0, np.pi, 13)) 
    x = R * np.cos(u) * np.sin(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(v)
    ax.plot_wireframe(x, y, z, alpha=0.2, color='blue')

    # add equator and null meridian
    u = np.linspace(0,2*np.pi, 25)
    ax.plot(R * np.cos(u), R*np.sin(u), 0*u, color='black')
    ax.plot(R * np.cos(u), 0*u, R*np.sin(u), color='black')

    # draw equatorial plane
    x, y = np.meshgrid(np.linspace(-5*R, 5*R, 20), 
                       np.linspace(-5*R, 5*R, 20))
    z = 0 * y
    ax.plot_surface(x, y, z, alpha=0.2, color='gray')

    # plot our position in red
    ax.stem(*(solution_coordinates[:, 0:3]/1000).transpose(), markerfmt='or', bottom=0, orientation='z')

    # plot satellites position and orbits in green
    for prn in pseudo_ranges:
        if telemetry and orbit_time > 0:
            ephemeris_subframes = [subframe['ephemeris'] for subframe in telemetry[prn] if 'ephemeris' in subframe.keys()]
            ephemeris = {}
            for subframe in ephemeris_subframes:
                ephemeris.update(subframe)
            send_time_gps = pseudo_ranges[prn]['send_time_gps'][0]
            orbit_times = np.linspace(send_time_gps, send_time_gps + orbit_time, 100)  # from one hour before to one after
            satellite_orbits = observables.sat_position_from_ephemeris(orbit_times, ephemeris)
            if inertial_frame:
                satellite_orbits = np.squeeze(np.array([coordinates.ecef_to_eci(pos, time - send_time_gps) 
                                                        for pos, time in zip(satellite_orbits, orbit_times)]))
            ax.plot(*(satellite_orbits/1000).transpose(), '-', color='green', alpha=0.5)
        
        ax.stem(*(satellite_positions[prn][-2:-1]/1000).transpose(), markerfmt='og', bottom=0, orientation='z')
        ax.text(*(satellite_positions[prn][-1]/1000 + [0,0,1000]), str(prn), None)

    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])  # set equal aspect ratio
    ax.set_xlabel(('x [km]'))
    ax.set_ylabel(('y [km]'))
    ax.set_zlabel(('z [km]'))
    return fig, ax