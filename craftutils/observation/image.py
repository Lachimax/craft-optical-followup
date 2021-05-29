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
        self.instrument = instrument

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

    def get_header_item(self, key: str, ext: int = 0):
        self.load_headers()
        if key in self.headers[ext]:
            return self.headers[ext][key]

    def get_id(self):
        return self.filename[:self.filename.find(".fits")]

    def extract_gain(self):
        self.gain = self.get_header_item("GAIN") * units.electron / units.adu
        return self.gain

    def extract_exposure_time(self):
        self.exposure_time = self.get_header_item("EXPTIME") * units.second

    def extract_noise_read(self):
        self.noise_read = self.get_header_item("HIERARCH CELL.READNOISE") * units




class ImagingImage(Image):
    def __init__(self, path: str = None, frame_type: str = None, instrument: str = None):
        super().__init__(path=path, frame_type=frame_type, instrument=instrument)
        self.filter = None
        self.filter_short = None
        self.pixel_scale_ra = None
        self.pixel_scale_dec = None

        self.psfex_path = None
        self.psfex_output = None
        self.source_cat_path = None
        self.source_cat = None

        self.fwhm_pix_psfex = None
        self.fwhm_psfex = None
        self.exposure_time = None
        self.gain = None

        self.zeropoints = {}
        self.zeropoint_output_paths = {}

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
        self.source_cat_path = self.source_extraction(configuration_file=config,
                                                      output_dir=output_dir,
                                                      parameters_file=output_params,
                                                      catalog_name=f"{self.name}_psf-fit.cat",
                                                      psf_name=self.psfex_path,
                                                      seeing_fwhm=self.fwhm_psfex.value,
                                                      **configs
                                                      )
        print(cat_path)
        self.load_source_cat()
        self.update_output_file()

    def load_source_cat(self, force: bool = False):
        if force:
            self.source_cat = None
        if self.source_cat is None:
            self.source_cat = table.Table.read(self.source_cat_path, format="ascii.sextractor")

    def extract_pixel_scale(self, layer: int = 0):
        if self.pixel_scale_ra is None or self.pixel_scale_dec is None:
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
                        "source_cat_path": self.source_cat_path,
                        "fwhm_pix_psfex": self.fwhm_pix_psfex,
                        "fwhm_psfex": self.fwhm_psfex,
                        "zeropoints": self.zeropoints,
                        "zeropoint_output_paths": self.zeropoint_output_paths})
        return outputs

    def load_output_file(self):
        outputs = super().load_output_file()
        if outputs is not None:
            if "filter" in outputs:
                self.filter = outputs["filter"]
            if "psfex_path" in outputs:
                self.psfex_path = outputs["psfex_path"]
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
                  cat_zeropoint: float = 0.0,
                  cat_zeropoint_err: float = 0.0,
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
                  mag_range_sex_lower: float = -100.,
                  mag_range_sex_upper: float = 100.,
                  pix_tol: float = 5.,
                  ):

        if image_name is None:
            image_name = self.name

        column_names = cat_columns(cat=cat_name, f=self.filter_short)
        cat_ra_col = column_names['ra']
        cat_dec_col = column_names['dec']
        cat_mag_col = column_names['mag_psf']
        cat_type = "csv"

        zp_dict = ph.determine_zeropoint_sextractor(sextractor_cat_path=self.source_cat_path,
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
                                                    pix_tol=pix_tol,
                                                    flux_column=sex_flux_col,
                                                    mag_range_sex_upper=mag_range_sex_upper,
                                                    mag_range_sex_lower=mag_range_sex_lower,
                                                    stars_only=stars_only,
                                                    star_class_tol=star_class_tol,
                                                    star_class_col=star_class_col,
                                                    exp_time=self.exposure_time,
                                                    cat_type=cat_type,
                                                    cat_zeropoint=cat_zeropoint,
                                                    cat_zeropoint_err=cat_zeropoint_err
                                                    )

        self.zeropoints[cat_name.lower()] = zp_dict
        self.zeropoint_output_paths[cat_name.lower()] = output_path

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
        self.get_filter()
        self.instrument = "panstarrs1"
        self.exposure_time = None
        self.extract_exposure_time()

    def get_filter(self):
        self.load_headers()
        fil_string = self.get_header_item("HIERARCH FPA.FILTERID")
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

        self.exposure_time = exp_time  # np.mean(exp_times)

    def extract_gain(self):
        self.gain = self.get_header_item("HIERARCH CELL.GAIN") * units.electron / units.adu
        return self.gain


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
