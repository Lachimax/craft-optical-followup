import os
from typing import Union

import astropy.io.fits as fits

import craftutils.utils as u
import craftutils.fits_files as ff


class Image:
    def __init__(self, path: str = None):
        self.path = path
        self.filename = os.path.split(self.path)[-1]
        self.hdu = fits.open(path)


class SpecRawImage(Image):
    def pypeit_str(self):
        header = self.hdu[0].header
        string = f"| {self.filename} | "
