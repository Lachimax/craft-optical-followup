import os
from typing import Union

import astropy.io.fits as fits

import craftutils.utils as u
import craftutils.fits_files as ff


class Image:
    def __init__(self, path: str = None):
        self.path = path
        self.filename = os.path.split(self.path)[-1]
        self.hdu = None

    def open(self):
        if self.path is not None:
            self.hdu = fits.open(self.path)
        else:
            print("The FITS file could not be loaded because path has not been set.")

    def close(self):
        if self.hdu is not None:
            self.hdu.close()

    def id(self):
        return self.filename[:self.filename.find(".fits")]


class SpecRawImage(Image):
    def __init__(self, path: str = None):
        super().__init__(path=path)

    @classmethod
    def from_pypeit_line(cls, line: str, pypeit_raw_path: str):
        attributes = line.split('|')
        filename = attributes[0]
        return SpecRawImage(path=os.path.join(pypeit_raw_path, filename))

    # def pypeit_str(self):
    #     header = self.hdu[0].header
    #     string = f"| {self.filename} | "
