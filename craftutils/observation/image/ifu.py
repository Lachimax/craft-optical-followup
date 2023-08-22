import numpy as np

import astropy.io.fits as fits
import astropy.units as units

from .__init__ import ImagingImage


class IFUImage(ImagingImage):
    axis_spec = 0

    def white_light_image(
            self,
            output_path: str = None,
            ext: int = 1,
    ):
        data = self.collapse_spectral(ext=ext)
        unit = data.unit
        header = self.header_2d(ext=ext)
        header["BUNIT"] = str(unit)
        hdu = fits.PrimaryHDU(data=data.value, header=header)
        hdul = fits.HDUList(hdus=[hdu])
        hdul.writeto(output_path, overwrite=True)
        img = ImagingImage(
            path=output_path,
            instrument_name="vlt-muse"
        )
        return img

    def collapse_spectral(
            self,
            ext: int = 1,
    ):
        self.load_data()
        dx = self.extract_dz()
        data = self.data[ext]
        data[np.isnan(data)] = 0. * data.unit
        return np.trapz(data, dx=dx, axis=self.axis_spec)

    def extract_specunit(
            self,
            astropy: bool = False
    ):
        unit = self.extract_header_item("CUNIT3")
        if astropy:
            if unit is not None:
                unit = units.Unit(unit)
            else:
                unit = units.ct / units.angstrom
        return unit

    def extract_dz(
            self,
            ext: int = 1
    ):
        unit = self.extract_specunit(True)
        return self.extract_header_item("CD3_3", ext=ext) * unit

    def header_2d(
            self,
            ext: int = 1
    ):
        header = self.headers[ext].copy()
        for key in ("CTYPE3", "CUNIT3", "CRVAL3", "CRPIX3", "CRDER3", "CD1_3", "CD2_3", "CD3_1", "CD3_2", "CD3_3"):
            if key in header:
                header.pop(key)
        return header


class MUSEImage(IFUImage):
    pass
    # def white_light_image(
    #         self,
    #         ext: 1,
    #         output_path: str = None
    # ):
    #     super().white_light_image()
