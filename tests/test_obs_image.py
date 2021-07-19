import os

import craftutils.observation.image as img
import craftutils.params as p

fors2_normalised_path = os.path.join(p.project_path, "tests", "files", "images", "divided_by_exp_time",
                                     "FRB-180924-Host-FORS2.2018-11-09T01:02:49.490_SCIENCE_REDUCED_IMG_norm.fits")

fors2_normalised_image = img.Image.from_fits(path=fors2_normalised_path)


def test_from_fits():
    assert isinstance(fors2_normalised_image, img.FORS2Image)


def test_extract_frame_type():
    frame_type = fors2_normalised_image.extract_frame_type()
    print(frame_type)
    assert frame_type == "science_reduced"