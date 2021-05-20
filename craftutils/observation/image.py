import os
from typing import Union

import numpy as np

import astropy.io.fits as fits

import craftutils.utils as u
import craftutils.fits_files as ff


class Image:
    def __init__(self, path: str = None, frame_type: str = None):
        self.path = path
        self.filename = os.path.split(self.path)[-1]
        self.hdu_list = None
        self.frame_type = frame_type

    def open(self):
        if self.path is not None and self.hdu_list is None:
            self.hdu_list = fits.open(self.path)
        elif self.path is None:
            print("The FITS file could not be loaded because path has not been set.")

    def close(self):
        if self.hdu_list is not None:
            self.hdu_list.close()

    def id(self):
        return self.filename[:self.filename.find(".fits")]


class SpecRaw(Image):
    def __init__(self, path: str = None, frame_type: str = None, decker: str = None, binning: str = None):
        super().__init__(path=path, frame_type=frame_type)
        self.pypeit_line = None
        self.decker = decker
        self.binning = binning

    @classmethod
    def from_pypeit_line(cls, line: str, pypeit_raw_path: str):
        attributes = line.split('|')
        attributes = list(map(lambda a: a.replace(" ", ""), attributes))
        inst = SpecRaw(path=os.path.join(pypeit_raw_path, attributes[1]),
                       frame_type=attributes[2],
                       decker=attributes[7],
                       binning=attributes[8])
        inst.pypeit_line = line
        return inst


class Spec1DCoadded(Image):
    def __init__(self, path: str = None):
        super().__init__(path=path)
        self.marz_format_path = None

    def convert_to_marz_format(self, output: str, lambda_min: float = None, lambda_max: float = None):
        """
        Extracts the 1D spectrum from the PypeIt-generated file and rearranges it into the format accepted by Marz.
        :param output:
        :param lambda_min:
        :param lambda_max:
        :return:
        """
        self.open()
        data = self.hdu_list[1].data
        header = self.hdu_list[1].header.copy()
        header.update(self.hdu_list[0].header)
        del header["TTYPE1"]
        del header["TTYPE2"]
        del header["TTYPE3"]
        del header["TTYPE4"]
        del header["TFORM1"]
        del header["TFORM2"]
        del header["TFORM3"]
        del header["TFORM4"]
        del header["TFIELDS"]
        del header['XTENSION']

        i_min = np.abs(lambda_min - data['flux']).argmin()
        i_max = np.abs(lambda_max - data['flux']).argmin()
        data = data[i_min:i_max]

        primary = fits.PrimaryHDU(data['flux'])
        primary.header.update(header)

        variance = fits.ImageHDU(data['ivar'])
        variance.name = 'VARIANCE'

        wavelength = fits.ImageHDU(data['wave'])
        wavelength.name = 'WAVELENGTH'

        new_hdu_list = fits.HDUList([primary, variance, wavelength])

        new_hdu_list.writeto(output)
        self.marz_format_path = output

        self.close()

# def pypeit_str(self):
#     header = self.hdu[0].header
#     string = f"| {self.filename} | "
