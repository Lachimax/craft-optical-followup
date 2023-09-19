import numpy as np

import astropy.io.fits as fits
import astropy.units as units

from craftutils.observation import filters
from .__init__ import ImagingImage


class IFUImage(ImagingImage):
    axis_spec = 0

    def white_light_image(
            self,
            output_path: str = None,
            ext: int = 1,
    ):
        data = self.collapse_spectral(ext=ext)
        return self.image_2d(
            data=data,
            output_path=output_path,
            ext=ext
        )

    def attenuate_by_bandpass(
            self,
            band: filters.Filter,
            ext: int = 1
    ):
        """
        Attenuate the cube flux as a function of wavelength using an imaging bandpass.

        :param band: filter to attenuate by.
        :param ext: FITS extension to apply transformation to.
        :return:
        """
        waves = self.wavelengths(ext=ext)
        # Interpolate the filter profile to the cube's wavelength grid
        fil_interp = band.interp_to_wavelength(waves)
        # This next part uses the top answer from this StackOverflow question.
        # https://stackoverflow.com/questions/30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis
        data = self.data[ext] * 1.
        dim_array = np.ones((1, data.ndim), int).ravel()
        dim_array[0] = -1
        r_reshaped = fil_interp.reshape(dim_array)
        data_atten = data * r_reshaped
        return data_atten

    def image_2d(
            self,
            data: units.Quantity,
            output_path: str = None,
            ext: int = 1,
    ):
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

    def pseudo_image(
            self,
            output_path: str,
            band: filters.Filter,
            ext: int = 1
    ):
        """
        Uses an imaging bandpass to generate a 2D image as though viewed through that bandpass.

        :param output_path:
        :param band:
        :param ext:
        :return:
        """
        data = self.collapse_spectral(
            data=self.attenuate_by_bandpass(band=band),
            ext=ext
        )
        return self.image_2d(
            data=data,
            output_path=output_path,
            ext=ext
        )

    def wavelengths(
            self,
            ext: int = 1
    ):
        self.load_wcs()
        wavelengths = self.wcs[ext].all_pix2world(0, 0, np.arange(self.data[ext].shape[0]), 0)[2] * units.m
        return wavelengths.to("Angstrom")

    def collapse_spectral(
            self,
            ext: int = 1,
            data: units.Quantity = None
    ):
        if data is None:
            self.load_data()
            data = self.data[ext]
        dx = self.extract_dz()
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
        for key in (
        "CTYPE3", "CUNIT3", "CRVAL3", "CRPIX3", "CRDER3", "CD1_3", "CD2_3", "CD3_1", "CD3_2", "CD3_3", "CHECKSUM",
        "DATASUM"):
            if key in header:
                header.pop(key)
        header["EQUINOX"] = 2000.
        return header


class MUSEImage(IFUImage):
    pass
