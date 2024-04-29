from .image import *
from .eso import *
from .imaging import *
from .spectroscopy import *
from .ifu import *

from .imaging.image import _set_class_dict as imaging_set_class_dict
from .imaging.coadded import _set_class_dict as coadded_set_class_dict

imaging_set_class_dict()
coadded_set_class_dict()


def best_for_path(
        image_dict: Dict[ImagingImage],
        exclude: list = ()
):
    from craftutils.observation.filters import Filter
    r_sloan = Filter.from_params("r", "sdss")
    best_score = np.inf * units.angstrom
    best_img = None
    for fil_name, img in image_dict.items():
        if fil_name in exclude:
            continue
        fil = img.filter
        score = r_sloan.compare_wavelength_range(fil)
        if score < best_score:
            best_score = score
            best_img = img
    print(f"Best image for PATH is {best_img.filter.name}")
    return best_img
