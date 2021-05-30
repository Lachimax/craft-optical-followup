import os
import warnings
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
import astropy.table as table
import astropy.wcs as wcs
import astropy.units as units

import craftutils.utils as u
import craftutils.fits_files as ff
import craftutils.photometry as ph
import craftutils.params as p
import craftutils.plotting as pl
from craftutils.retrieve import cat_columns


class Image:

    def __init__(self, path: str = None, frame_type: str = None, instrument: str = None):
        self.path = path
        self.output_path = path.replace(".fits", "_outputs.yaml")
        self.data_path, self.filename = os.path.split(self.path)
        self.name = self.get_id()
        self.hdu_list = None
        self.frame_type = frame_type
        self.headers = None
        self.data = None
        self.instrument = instrument

        # Header attributes
        self.exposure_time = None
        self.gain = None
        self.noise_read = None
        self.n_x = None
        self.n_y = None
        self.n_pix = None

    def __eq__(self, other):
        return self.name == other.name

    def open(self):
        if self.path is not None and self.hdu_list is None:
            self.hdu_list = fits.open(self.path)
        elif self.path is None:
            print("The FITS file could not be loaded because path has not been set.")

    def close(self):
        if self.hdu_list is not None:
            self.hdu_list.close()
            self.hdu_list = None

    def load_output_file(self):
        outputs = p.load_output_file(self)
        if outputs is not None:
            if "frame_type" in outputs:
                self.frame_type = outputs["frame_type"]
        return outputs

    def update_output_file(self):
        p.update_output_file(self)

    def _output_dict(self):
        return {"frame_type": self.frame_type,
                }

    def load_headers(self, force: bool = False):
        if self.headers is None or force:
            self.open()
            self.headers = list(map(lambda h: h.header, self.hdu_list))
            self.close()
        else:
            print("Headers already loaded.")

    def load_data(self, force: bool = False):
        if self.data is None or force:
            self.open()
            self.data = list(map(lambda h: h.data, self.hdu_list))
            self.close()
        else:
            print("Data already loaded.")

    def get_id(self):
        return self.filename[:self.filename.find(".fits")]

    def extract_header_item(self, key: str, ext: int = 0):
        self.load_headers()
        if key in self.headers[ext]:
            return self.headers[ext][key]

    def extract_gain(self):
        key = self.header_keys()["gain"]
        self.gain = self.extract_header_item(key) * units.electron / units.ct
        return self.gain

    def extract_exposure_time(self):
        key = self.header_keys()["exposure_time"]
        self.exposure_time = self.extract_header_item(key) * units.second
        return self.exposure_time

    def extract_noise_read(self):
        key = self.header_keys()["noise_read"]
        self.noise_read = self.extract_header_item(key) * units.electron / units.pixel
        return self.noise_read

    def extract_n_pix(self, ext: int = 0):
        self.load_data()
        self.n_y, self.n_x = self.data[ext].shape()
        self.n_pix = self.n_y * self.n_x
        return self.n_pix

    @classmethod
    def header_keys(cls):
        header_keys = {"exposure_time": "EXPTIME",
                       "noise_read": "RON",
                       "gain": "GAIN"}
        return header_keys


class ImagingImage(Image):
    def __init__(self, path: str = None, frame_type: str = None, instrument: str = None):
        super().__init__(path=path, frame_type=frame_type, instrument=instrument)
        self.filter = None
        self.filter_short = None
        self.pixel_scale_ra = None
        self.pixel_scale_dec = None

        self.psfex_path = None
        self.psfex_output = None
        self.source_cat_sextractor_path = None
        self.source_cat_path = None
        self.source_cat = None

        self.fwhm_pix_psfex = None
        self.fwhm_psfex = None

        self.sky_background = None

        self.zeropoints = {}
        self.zeropoint_output_paths = {}

        self.depth = None

        self.load_output_file()

    def source_extraction(self, configuration_file: str,
                          output_dir: str,
                          parameters_file: str = None,
                          catalog_name: str = None,
                          **configs):
        return ph.source_extractor(image_path=self.path,
                                   output_dir=output_dir,
                                   configuration_file=configuration_file,
                                   parameters_file=parameters_file,
                                   catalog_name=catalog_name,
                                   **configs
                                   )

    def psfex(self, catalog: str, output_dir: str):
        self.psfex_path = ph.psfex(catalog=catalog, output_dir=output_dir)
        self.psfex_output = fits.open(self.psfex_path)
        self.extract_pixel_scale()
        pix_scale = self.pixel_scale_dec
        self.fwhm_pix_psfex = self.psfex_output[1].header['PSF_FWHM'] * units.pixel
        self.fwhm_psfex = self.fwhm_pix_psfex.to(units.arcsec, pix_scale)
        self.update_output_file()

    def source_extraction_psf(self, output_dir: str, **configs):
        config = p.path_to_config_sextractor_config_pre_psfex()
        output_params = p.path_to_config_sextractor_param_pre_psfex()
        cat_path = self.source_extraction(configuration_file=config,
                                          output_dir=output_dir,
                                          parameters_file=output_params,
                                          catalog_name=f"{self.name}_psfex.fits",
                                          )
        self.psfex(catalog=cat_path, output_dir=output_dir)
        config = p.path_to_config_sextractor_config()
        output_params = p.path_to_config_sextractor_param()
        self.source_cat_sextractor_path = self.source_extraction(configuration_file=config,
                                                                 output_dir=output_dir,
                                                                 parameters_file=output_params,
                                                                 catalog_name=f"{self.name}_psf-fit.cat",
                                                                 psf_name=self.psfex_path,
                                                                 seeing_fwhm=self.fwhm_psfex.value,
                                                                 **configs
                                                                 )
        print(cat_path)
        self.load_source_cat_sextractor()
        self.update_output_file()

    def load_source_cat_sextractor(self, force: bool = False):
        if force:
            self.source_cat = None
        if self.source_cat is None:
            self.source_cat = table.QTable.read(self.source_cat_sextractor_path, format="ascii.sextractor")

    def load_source_cat(self, force: bool = False):
        if force or self.source_cat is None:
            if self.source_cat_path is not None:
                self.source_cat = table.QTable.read(self.source_cat_path, format="ascii.ecsv")
            elif self.source_cat_sextractor_path is not None:
                self.load_source_cat_sextractor(force=force)
            else:
                warnings.warn("No valid source_cat_path found. Could not load source_table")

    def write_source_cat(self):
        if self.source_cat_path is None:
            self.source_cat_path = self.path.replace(".fits", "_source_cat.ecsv")
        if self.source_cat is None:
            raise ValueError("source_cat not yet loaded.")
        self.source_cat.write(self.source_cat_path, format="ascii.ecsv")

    def extract_pixel_scale(self, layer: int = 0, force: bool = False):
        if force or self.pixel_scale_ra is None or self.pixel_scale_dec is None:
            self.open()
            self.pixel_scale_ra, self.pixel_scale_dec = ff.get_pixel_scale(self.hdu_list, layer=layer,
                                                                           astropy_units=True)
            self.close()
        else:
            warnings.warn("Pixel scale already set.")

    def _output_dict(self):
        outputs = super()._output_dict()
        outputs.update({"filter": self.filter,
                        "psfex_path": self.psfex_path,
                        "source_cat_sextractor_path": self.source_cat_sextractor_path,
                        "source_cat_path": self.source_cat_path,
                        "fwhm_pix_psfex": self.fwhm_pix_psfex,
                        "fwhm_psfex": self.fwhm_psfex,
                        "zeropoints": self.zeropoints,
                        "zeropoint_output_paths": self.zeropoint_output_paths})
        return outputs

    def update_output_file(self):
        p.update_output_file(self)
        if self.source_cat is not None:
            self.write_source_cat()

    def load_output_file(self):
        outputs = super().load_output_file()
        if outputs is not None:
            if "filter" in outputs:
                self.filter = outputs["filter"]
            if "psfex_path" in outputs:
                self.psfex_path = outputs["psfex_path"]
            if "source_cat_sextractor_path" in outputs:
                self.source_cat_sextractor_path = outputs["source_cat_sextractor_path"]
            if "source_cat_path" in outputs:
                self.source_cat_path = outputs["source_cat_path"]
            if "fwhm_psfex" in outputs:
                self.fwhm_psfex = outputs["fwhm_psfex"]
            if "fwhm_psfex" in outputs:
                self.fwhm_pix_psfex = outputs["fwhm_pix_psfex"]
            if "zeropoints" in outputs:
                self.zeropoints = outputs["zeropoints"]
            if "zeropoint_output_paths" in outputs:
                self.zeropoint_output_paths = outputs["zeropoint_output_paths"]
        return outputs

    def zeropoint(self,
                  cat_path: str,
                  output_path: str,
                  cat_name: str = 'Catalogue',
                  cat_zeropoint: units.Quantity = 0.0 * units.mag,
                  cat_zeropoint_err: units.Quantity = 0.0 * units.mag,
                  image_name: str = None,
                  show: bool = False,
                  sex_x_col: str = 'XPSF_IMAGE',
                  sex_y_col: str = 'YPSF_IMAGE',
                  sex_ra_col: str = 'ALPHAPSF_SKY',
                  sex_dec_col: str = 'DELTAPSF_SKY',
                  sex_flux_col: str = 'FLUX_PSF',
                  stars_only: bool = True,
                  star_class_col: str = 'CLASS_STAR',
                  star_class_tol: float = 0.95,
                  mag_range_sex_lower: units.Quantity = -100. * units.mag,
                  mag_range_sex_upper: units.Quantity = 100. * units.mag,
                  dist_tol: units.Quantity = 2. * units.arcsec,
                  ):
        self.signal_to_noise()
        if image_name is None:
            image_name = self.name

        column_names = cat_columns(cat=cat_name, f=self.filter_short)
        cat_ra_col = column_names['ra']
        cat_dec_col = column_names['dec']
        cat_mag_col = column_names['mag_psf']
        cat_type = "csv"

        zp_dict = ph.determine_zeropoint_sextractor(sextractor_cat=self.source_cat,
                                                    image=self.path,
                                                    cat_path=cat_path,
                                                    cat_name=cat_name,
                                                    output_path=output_path,
                                                    image_name=image_name,
                                                    show=show,
                                                    cat_ra_col=cat_ra_col,
                                                    cat_dec_col=cat_dec_col,
                                                    cat_mag_col=cat_mag_col,
                                                    sex_ra_col=sex_ra_col,
                                                    sex_dec_col=sex_dec_col,
                                                    sex_x_col=sex_x_col,
                                                    sex_y_col=sex_y_col,
                                                    dist_tol=dist_tol,
                                                    flux_column=sex_flux_col,
                                                    mag_range_sex_upper=mag_range_sex_upper,
                                                    mag_range_sex_lower=mag_range_sex_lower,
                                                    stars_only=stars_only,
                                                    star_class_tol=star_class_tol,
                                                    star_class_col=star_class_col,
                                                    exp_time=self.exposure_time,
                                                    cat_type=cat_type,
                                                    cat_zeropoint=cat_zeropoint,
                                                    cat_zeropoint_err=cat_zeropoint_err,
                                                    snr_col='SNR'
                                                    )

        zp_dict['airmass'] = 0.0
        self.zeropoints[cat_name.lower()] = zp_dict
        self.zeropoint_output_paths[cat_name.lower()] = output_path
        self.update_output_file()
        return self.zeropoints[cat_name.lower()]

    def aperture_areas(self):
        self.load_source_cat()
        self.extract_pixel_scale()
        self.source_cat["A_IMAGE"] = self.source_cat["A_WORLD"].to(units.pix, self.pixel_scale_dec)
        self.source_cat["B_IMAGE"] = self.source_cat["A_WORLD"].to(units.pix, self.pixel_scale_dec)
        self.source_cat["KRON_AREA_IMAGE"] = self.source_cat["A_IMAGE"] * self.source_cat["B_IMAGE"] * np.pi

    def calibrate_magnitudes(self, zeropoint_name: str, force: bool = False):
        if force or f"MAG_AUTO_ZP_{zeropoint_name}" not in self.source_cat:
            if zeropoint_name not in self.zeropoints:
                raise KeyError(f"Zeropoint {zeropoint_name} does not exist.")
            zp_dict = self.zeropoints[zeropoint_name]
            mag, mag_err_minus, mag_err_plus = ph.magnitude_complete(flux=self.source_cat["FLUX_AUTO"],
                                                                     flux_err=self.source_cat[
                                                                         "FLUXERR_AUTO"],
                                                                     exp_time=self.exposure_time,
                                                                     exp_time_err=0.0 * units.second,
                                                                     zeropoint=zp_dict['zeropoint'],
                                                                     zeropoint_err=zp_dict[
                                                                         'zeropoint_err'],
                                                                     airmass=zp_dict['airmass'],
                                                                     airmass_err=0.0,
                                                                     ext=0.0 * units.mag,
                                                                     ext_err=0.0 * units.mag,
                                                                     colour_term=0.0,
                                                                     colour=0.0 * units.mag,
                                                                     )
            self.source_cat[f"MAG_AUTO_ZP_{zeropoint_name}"] = mag
            self.source_cat[f"MAGERR_AUTO_ZP_{zeropoint_name}"] = np.max([np.abs(mag_err_minus), np.abs(mag_err_plus)])
            self.update_output_file()
        else:
            print(f"Magnitudes already calibrated for {zeropoint_name}")

    def estimate_depth(self, zeropoint_name: str):
        self.load_source_cat()
        self.signal_to_noise()
        self.calibrate_magnitudes(zeropoint_name=zeropoint_name)
        cat_3sigma = self.source_cat[self.source_cat["SNR"] > 3.0]
        self.depth = np.max(cat_3sigma[f"MAG_AUTO_ZP_{zeropoint_name}"])
        self.update_output_file()
        return self.depth

    def signal_to_noise(self):
        self.load_source_cat()
        self.extract_exposure_time()
        self.extract_gain()
        self.aperture_areas()
        flux_target = self.source_cat['FLUX_AUTO']
        rate_target = flux_target / self.exposure_time
        rate_sky = self.estimate_sky_background() / self.exposure_time
        rate_read = self.extract_noise_read()
        n_pix = self.source_cat['KRON_AREA_IMAGE'] / units.pixel

        self.source_cat["SNR"] = ph.signal_to_noise(rate_target=rate_target,
                                                    rate_sky=rate_sky,
                                                    rate_read=rate_read,
                                                    exp_time=self.exposure_time,
                                                    gain=self.gain,
                                                    n_pix=n_pix
                                                    )
        self.update_output_file()
        return self.source_cat["SNR"]

    def estimate_sky_background(self, ext: int = 0, force: bool = False):
        if force or self.sky_background is None:
            self.load_data()
            self.sky_background = np.nanmedian(self.data[ext]) * units.ct / units.pixel
        else:
            print("Sky background already estimated.")
        return self.sky_background

    def plot_apertures(self):
        self.load_source_cat()
        pl.plot_all_params(image=self.path, cat=self.source_cat, kron=True, show=False)
        plt.title(self.filter)
        plt.show()

    @classmethod
    def select_child_class(cls, instrument: str):
        instrument = instrument.lower()
        if instrument == "panstarrs":
            return PanSTARRS1Cutout
        else:
            raise ValueError(f"Unrecognised instrument {instrument}")


class PanSTARRS1Cutout(ImagingImage):

    def __init__(self, path: str = None):
        super().__init__(path=path)
        self.extract_filter()
        self.instrument = "panstarrs1"
        self.exposure_time = None
        self.extract_exposure_time()

    def extract_filter(self):
        key = self.header_keys()["filter"]
        fil_string = self.extract_header_item(key)
        self.filter = fil_string[:fil_string.find(".")]
        self.filter_short = self.filter

    def extract_exposure_time(self):
        self.load_headers()
        exp_time_keys = filter(lambda k: k.startswith("EXP_"), self.headers[0])
        exp_time = 0.
        # exp_times = []
        for key in exp_time_keys:
            exp_time += self.headers[0][key]
        #    exp_times.append(self.headers[0][key])

        self.exposure_time = exp_time * units.second  # np.mean(exp_times)

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({"noise_read": "HIERARCH CELL.READNOISE",
                            "filter": "HIERARCH FPA.FILTERID",
                            "gain": "HIERARCH CELL.GAIN"})
        return header_keys


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
