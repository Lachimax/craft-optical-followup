import os

import numpy as np
import pytest
import astropy.io.fits as fits

import craftutils.wrap.montage as montage
import craftutils.params as p

good_input = os.path.join(p.project_path, "tests", "files", "images", "divided_by_exp_time")
bad_input = os.path.join(p.project_path, "tests", "files", "images", "divided_by_exp_time_bad")
coadded_file = os.path.join(p.project_path, "tests", "files", "images", "coadded", "FRB20180924_1_2018-11-09.fits")


# def test_check_input_images():
#     with pytest.raises(ValueError) as e:
#         montage.check_input_images(input_directory=bad_input)
#
#
# def test_inject_header():
#     montage.inject_header(file_path=coadded_file, input_directory=good_input)
#     hdu = fits.open(coadded_file)
#     header = hdu[0].header
#     assert header["EXPTIME"] == 1.0
#     assert np.round(header["GAIN"]) == np.round(2 * 400 * 2 / 3)
#     assert header["NCOMBINE"] == 2
