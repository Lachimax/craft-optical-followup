from .image import *
from .eso import *
from .imaging import *
from .spectroscopy import *
from .ifu import *

from .imaging.imaging import _set_class_dict as imaging_set_class_dict
from .imaging.coadded import _set_class_dict as coadded_set_class_dict

imaging_set_class_dict()
coadded_set_class_dict()
