import shutil
from typing import Tuple

import astropy.io.fits as fits

import craftutils.params as p

def galfit(output_dir: str):
    shutil.copy(p.path_to_config_galfit(), output_dir)

def galfit_feedme(
        output_dir: str,
        input_file: str = None,
        output_file: str = None,
        sigma_file: str = None,
        psf_file: str = None,
        psf_fine_sampling: int = None,
        mask_file: str = None,
        constraint_file: str = None,
        fitting_region_margins: Tuple[int] = None,
        convolution_size: int = None,

):
    """
    Any unset values will be left as the GALFIT defaults (see param/galfit/galfit.feedme)
    :param output_dir:
    :param input_file:
    :param output_file:
    :param sigma_file:
    :return:
    """
    shutil.copy(p.path_to_config_galfit(), output_dir)


