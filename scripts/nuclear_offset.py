# Code by Lachlan Marnoch, 2019
from astropy.coordinates import SkyCoord
import os
import numpy as np

from PyCRAFT import params as p
from PyCRAFT import astrometry as am

bursts = list(filter(lambda f: f[-5:] == '.yaml' and 'template' not in f, os.listdir('param/FRBs/')))
bursts.sort()

for burst in bursts:
    burst = burst[:-5]
    burst_params = p.object_params_frb(burst)

    hg_ra = burst_params['hg_ra']
    hg_dec = burst_params['hg_dec']
    hg_x_err = burst_params['hg_err_x']
    hg_y_err = burst_params['hg_err_y']

    hg_x_err = 2 * np.sqrt(hg_x_err)
    hg_y_err = 2 * np.sqrt(hg_y_err)

    a, b, theta = am.calculate_error_ellipse(burst_params, 'quadrature')

    burst_err = np.sqrt(a ** 2 + b ** 2)

    hg_err = np.sqrt(hg_x_err ** 2 + hg_y_err ** 2)

    burst_ra = burst_params['burst_ra']
    burst_dec = burst_params['burst_dec']

    hg_coord = SkyCoord(f'{hg_ra}d {hg_dec}d')
    burst_coord = SkyCoord(f'{burst_ra}d {burst_dec}d')

    offset = hg_coord.separation(burst_coord).value
    offset_err = hg_err + burst_err
    print(burst)
    print(offset * 3600, 'arcsec +/-', offset_err * 3600)

    ang_size_distance = burst_params['ang_size_distance']
    offset_pc = np.deg2rad(ang_size_distance) * offset
    offset_pc_err = np.deg2rad(ang_size_distance) * offset_err

    print(offset_pc / 1000, 'kpc +/-', offset_pc_err / 1000)
    print(ang_size_distance / 1e6, 'Mpc')
    print()
