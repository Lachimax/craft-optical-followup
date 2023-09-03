import os
import copy

import numpy as np

import astropy.units as units
import astropy.io.fits as fits

from ..__init__ import Image, from_path


class Spectrum(Image):
    def __init__(self, path: str = None, frame_type: str = None, decker: str = None, binning: str = None,
                 grism: str = None):
        super().__init__(path=path, frame_type=frame_type)
        self.decker = decker
        self.binning = binning
        self.grism = grism

        self.lambda_min = None
        self.lambda_max = None

    def get_lambda_range(self):
        if self.epoch is not None:
            self.lambda_min = self.epoch.grisms[self.grism]["lambda_min"]
            self.lambda_max = self.epoch.grisms[self.grism]["lambda_max"]
        else:
            print("self.epoch not set; could not determine lambda range")

    @classmethod
    def select_child_class(cls, instrument_name: str, **kwargs):
        if 'frame_type' in kwargs:
            frame_type = kwargs['frame_type']
            if frame_type == "coadded":
                return Coadded1DSpectrum
            elif frame_type == "raw":
                return RawSpectrum
        else:
            raise KeyError("frame_type is required.")


class RawSpectrum(Spectrum):
    frame_type = "raw"

    def __init__(self, path: str = None, frame_type: str = None, decker: str = None, binning: str = None):
        super().__init__(path=path, frame_type=frame_type, decker=decker, binning=binning)
        self.pypeit_line = None

    @classmethod
    def from_pypeit_line(cls, line: str, pypeit_raw_path: str):
        attributes = line.split('|')
        attributes = list(map(lambda at: at.replace(" ", ""), attributes))
        inst = from_path(
            path=os.path.join(pypeit_raw_path, attributes[0]),
            frame_type=attributes[2],
            decker=attributes[7],
            binning=attributes[8],
            cls=RawSpectrum
        )
        inst.pypeit_line = line
        return inst


class Coadded1DSpectrum(Spectrum):
    def __init__(self, path: str = None, grism: str = None):
        super().__init__(path=path, grism=grism)
        self.marz_format_path = None
        self.trimmed_path = None

    def trim(self, output: str = None, lambda_min: units.Quantity = None, lambda_max: units.Quantity = None):
        if lambda_min is None:
            lambda_min = self.lambda_min
        if lambda_max is None:
            lambda_max = self.lambda_max

        lambda_min = lambda_min.to(units.angstrom)
        lambda_max = lambda_max.to(units.angstrom)

        if output is None:
            output = self.path.replace(".fits", f"_trimmed_{lambda_min.value}-{lambda_max.value}.fits")

        self.open()
        hdu_list = copy.deepcopy(self.hdu_list)
        data = hdu_list[1].data
        i_min = np.abs(lambda_min.to(units.angstrom).value - data['wave']).argmin()
        i_max = np.abs(lambda_max.to(units.angstrom).value - data['wave']).argmin()
        data = data[i_min:i_max]
        hdu_list[1].data = data
        hdu_list.writeto(output, overwrite=True)
        self.trimmed_path = output

    def convert_to_marz_format(self, output: str = None, version: str = "main"):
        """
        Extracts the 1D spectrum from the PypeIt-generated file and rearranges it into the format accepted by Marz.
        :param output:
        :param lambda_min:
        :param lambda_max:
        :return:
        """
        self.get_lambda_range()

        if version == "main":
            path = self.path
        elif version == "trimmed":
            path = self.trimmed_path

        hdu_list = fits.open(path)

        data = hdu_list[1].data
        header = hdu_list[1].header.copy()
        header.update(hdu_list[0].header)

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

        i_min = np.abs(self.lambda_min.to(units.angstrom).value - data['wave']).argmin()
        i_max = np.abs(self.lambda_max.to(units.angstrom).value - data['wave']).argmin()
        data = data[i_min:i_max]

        primary = fits.PrimaryHDU(data['flux'])
        primary.header.update(header)
        primary.header["NAXIS"] = 1
        primary.header["NAXIS1"] = len(data)
        del primary.header["NAXIS2"]

        variance = fits.ImageHDU(data['ivar'])
        variance.name = 'VARIANCE'
        primary.header["NAXIS"] = 1
        primary.header["NAXIS1"] = len(data)

        wavelength = fits.ImageHDU(data['wave'])
        wavelength.name = 'WAVELENGTH'
        primary.header["NAXIS"] = 1
        primary.header["NAXIS1"] = len(data)

        new_hdu_list = fits.HDUList([primary, variance, wavelength])

        if output is None:
            output = self.path.replace(".fits", "_marz.fits")
        new_hdu_list.writeto(output, overwrite=True)
        self.marz_format_path = output
        hdu_list.close()
        self.update_output_file()

    def _output_dict(self):
        outputs = super()._output_dict()
        outputs.update({
            "marz_format_path": self.marz_format_path,
            "trimmed_paths": self.trimmed_path
        })
        return outputs

# def pypeit_str(self):
#     header = self.hdu[0].header
#     string = f"| {self.filename} | "
