# Code by Lachlan Marnoch, 2021
import copy
import math
import string
import os
import shutil
import sys
import warnings
from typing import Union, Tuple, List
from copy import deepcopy

import ccdproc
import numpy as np
import matplotlib.pyplot as plt

import astropy
import astropy.io.fits as fits
import astropy.table as table
import astropy.wcs as wcs
import astropy.units as units
from astropy.stats import sigma_clipped_stats, SigmaClip

from astropy.visualization import (
    ImageNormalize, LogStretch, SqrtStretch, ZScaleInterval, MinMaxInterval,
    PowerStretch, wcsaxes)
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import quantity_support

from ccdproc import cosmicray_lacosmic

from astroalign import register

import photutils

import sep

import craftutils.utils as u
import craftutils.astrometry as a
import craftutils.fits_files as ff
import craftutils.photometry as ph
import craftutils.params as p
import craftutils.plotting as pl
from craftutils.stats import gaussian_distributed_point
import craftutils.observation.instrument as inst
import craftutils.wrap.source_extractor as se
import craftutils.wrap.psfex as psfex
import craftutils.observation.log as log
from craftutils.wrap.astrometry_net import solve_field
from craftutils.retrieve import cat_columns

quantity_support()

# This contains the names as in the header as keys and the names as used in this project as values.
instrument_header = {
    "FORS2": "vlt-fors2",
    "HAWKI": "vlt-hawki"
}


# TODO: Make this list all fits files, then write wrapper that eliminates non-science images and use that in scripts.
def fits_table(input_path: str, output_path: str = "", science_only: bool = True):
    """
    Produces and writes to disk a table of .fits files in the given path, with the vital statistics of each. Intended
    only for use with raw ESO data.
    :param input_path:
    :param output_path:
    :param science_only: If True, we are writing a list for a folder that also contains calibration files, which we want
     to ignore.
    :return:
    """

    # If there's no trailing slash in the paths, add one.
    input_path = u.check_trailing_slash(input_path)

    if output_path == "":
        output_path = input_path + "fits_table.csv"
    elif output_path[-4:] != ".csv":
        if output_path[-1] == "/":
            output_path = output_path + "fits_table.csv"
        else:
            output_path = output_path + ".csv"

    print('Writing table of fits files to: \n', output_path)

    files = os.listdir(input_path)
    files.sort()
    files_fits = []

    # Keep only the relevant fits files

    for f in files:
        if f.endswith(".fits") and not f.startswith("M."):
            files_fits.append(f)

    # Create list of dictionaries to be used as the output data
    output = []

    ids = string.ascii_lowercase
    if len(ids) < len(files_fits):
        ids = ids + string.ascii_uppercase
    if len(ids) < len(files_fits):
        ids = ids + string.digits

    for i, f in enumerate(files_fits):
        data = {'identifier': f}
        file_path = os.path.join(input_path, f)
        image = Image.from_fits(path=file_path)
        header = image.load_headers()[0]
        if science_only:
            frame_type = image.extract_frame_type()
            if frame_type not in ("science", "science_reduced"):
                continue
        if len(ids) >= len(files_fits):
            data['id'] = ids[i]
        if "OBJECT" in header:
            data['object'] = header["OBJECT"]
        if "ESO OBS NAME" in header:
            data['obs_name'] = header["ESO OBS NAME"]
        if "EXPTIME" in header:
            data['exp_time'] = header["EXPTIME"]
        if "AIRMASS" in header:
            data['airmass'] = header["AIRMASS"]
        elif "ESO TEL AIRM START" in header and "ESO TEL AIRM END":
            data['airmass'] = (header["ESO TEL AIRM START"] + header["ESO TEL AIRM END"]) / 2
        if "CRVAL1" in header:
            data['ref_ra'] = header["CRVAL1"]
        if "CRVAL2" in header:
            data['ref_dec'] = header["CRVAL2"]
        if "CRPIX1" in header:
            data['ref_pix_x'] = header["CRPIX1"]
        if "CRPIX2" in header:
            data['ref_pix_y'] = header["CRPIX2"]
        if "EXTNAME" in header:
            data['chip'] = header["EXTNAME"]
        elif "ESO DET CHIP1 ID" in header:
            if header["ESO DET CHIP1 ID"] == 'CCID20-14-5-3':
                data['chip'] = 'CHIP1'
            if header["ESO DET CHIP1 ID"] == 'CCID20-14-5-6':
                data['chip'] = 'CHIP2'
        if "GAIN" in header:
            data['gain'] = header["GAIN"]
        if "INSTRUME" in header:
            data['instrument'] = header["INSTRUME"]
        if "ESO TEL AIRM START" in header:
            data['airmass_start'] = header["ESO TEL AIRM START"]
        if "ESO TEL AIRM END" in header:
            data['airmass_end'] = header["ESO TEL AIRM END"]
        if "ESO INS OPTI3 NAME" in header:
            data['collimater'] = header["ESO INS OPTI3 NAME"]
        if "ESO INS OPTI5 NAME" in header:
            data['filter1'] = header["ESO INS OPTI5 NAME"]
        if "ESO INS OPTI6 NAME" in header:
            data['filter2'] = header["ESO INS OPTI6 NAME"]
        if "ESO INS OPTI7 NAME" in header:
            data['filter3'] = header["ESO INS OPTI7 NAME"]
        if "ESO INS OPTI9 NAME" in header:
            data['filter4'] = header["ESO INS OPTI9 NAME"]
        if "ESO INS OPTI10 NAME" in header:
            data['filter5'] = header["ESO INS OPTI10 NAME"]
        if "ESO INS OPTI8 NAME" in header:
            data['camera'] = header["ESO INS OPTI8 NAME"]
        if "NAXIS1" in header:
            data['pixels_x'] = header["NAXIS1"]
        if "NAXIS2" in header:
            data['pixels_y'] = header["NAXIS2"]
        if "SATURATE" in header:
            data['saturate'] = header["SATURATE"]
        if "MJD-OBS" in header:
            data['mjd_obs'] = header["MJD-OBS"]
        output.append(data)

    out_file = table.Table(output)
    out_file.write(output_path, format="ascii.csv", overwrite=True)

    return out_file


def fits_table_all(input_path: str, output_path: str = "", science_only: bool = True):
    """
    Produces and writes to disk a table of .fits files in the given path, with the vital statistics of each. Intended
    only for use with raw ESO data.
    :param input_path:
    :param output_path:
    :param science_only: If True, we are writing a list for a folder that also contains calibration files, which we want
     to ignore.
    :return:
    """

    if output_path == "":
        output_path = os.path.join(input_path, "fits_table.csv")

    if os.path.isdir(output_path):
        output_path = output_path + "fits_table.csv"
    else:
        output_path = u.sanitise_file_ext(filename=output_path, ext="csv")

    print('Writing table of fits files to: \n', output_path)

    files = os.listdir(input_path)
    files.sort()
    files_fits = list(filter(lambda x: x.endswith('.fits'), files))

    # Create list of dictionaries to be used as the output data
    output = []

    for i, f in enumerate(files_fits):
        data = {}
        data["FILENAME"] = f
        file_path = os.path.join(input_path, f)
        image = Image.from_fits(path=file_path)
        if science_only:
            frame_type = image.extract_frame_type()
            if frame_type not in ("science", "science_reduced"):
                continue
        header = image.load_headers()[0]
        for key in header:
            # Remove comments.
            if key not in ["COMMENT", "HISTORY", '']:
                data[key] = header[key]
        if 'ESO TEL AIRM END' in data and 'ESO TEL AIRM START' in data:
            data['AIRMASS'] = (float(data['ESO TEL AIRM END']) + float(data['ESO TEL AIRM START'])) / 2
        output.append(data)
        data["PATH"] = file_path

    out_file = table.Table(output)
    out_file.write(output_path, format="ascii.csv", overwrite=True)

    return out_file


class Image:
    instrument_name = "dummy"

    def __init__(self, path: str, frame_type: str = None, instrument_name: str = None, logg: log.Log = None):
        self.path = path
        self.output_file = path.replace(".fits", "_outputs.yaml")
        self.data_path, self.filename = os.path.split(self.path)
        self.name = self.get_id()
        self.hdu_list = None
        self.frame_type = frame_type
        self.headers = None
        self.data = None
        if instrument_name is not None:
            self.instrument_name = instrument_name
        try:
            u.debug_print(1, f"Image.__init__(): {self}.instrument_name ==", self.instrument_name)
            self.instrument = inst.Instrument.from_params(instrument_name=self.instrument_name)

        except FileNotFoundError:
            u.debug_print(1, f"Image.__init__(): FileNotFoundError")
            self.instrument = None
        u.debug_print(
            1, f"Image.__init__(): {self}.instrument ==", self.instrument,
            self.instrument_name)
        self.epoch = None

        # Header attributes
        self.exposure_time = None
        self.gain = None
        self.noise_read = None
        self.date_obs = None
        self.mjd_obs = None
        self.n_x = None
        self.n_y = None
        self.n_pix = None
        self.object = None
        self.pointing = None
        self.saturate = None

        if logg is None:
            self.log = log.Log()
        u.debug_print(2, f"Image.__init__(): {self}.log.log.keys() ==", self.log.log.keys())

    def __eq__(self, other):
        if not isinstance(other, Image):
            raise TypeError("Can only compare Image instance to another Image instance.")
        return self.path == other.path

    def __str__(self):
        return self.filename

    def add_log(
            self, action: str,
            method=None,
            input_path: str = None,
            output_path: str = None,
            packages: List[str] = None,
            ext: int = 0,
            ancestors: List['Image'] = None
    ):

        if ancestors is not None:
            new_dict = {}
            for img in ancestors:
                new_dict[img.name] = img.log
            ancestors = new_dict

        self.log.add_log(
            action=action,
            method=method,
            input_path=input_path,
            output_path=output_path,
            packages=packages,
            ancestor_logs=ancestors
        )
        self.add_history(note=action, ext=ext)
        # self.update_output_file()

    def open(self, mode: str = "readonly"):
        u.debug_print(1, f"Image.open() 1: {self}.hdu_list:", self.hdu_list)
        if self.path is not None and self.hdu_list is None:
            self.hdu_list = fits.open(self.path, mode=mode)
            u.debug_print(1, f"Image.open() 2: {self}.hdu_list:", self.hdu_list)
        elif self.path is None:
            print("The FITS file could not be loaded because path has not been set.")

    def close(self):
        if self.hdu_list is not None:
            self.hdu_list.close()
            self.hdu_list = None

    def new_image(self, path: str):
        c = self.__class__
        new_image = c(path=path)
        new_image.log = self.log.copy()
        new_image.add_log(
            f"Derived from {self.path}.",
            method=self.new_image,
            input_path=self.path,
            output_path=path
        )
        return new_image

    def copy(self, destination: str):
        """
        A note to future me: do copy, THEN make changes, or be prepared to suffer the consequences.
        :param destination:
        :return:
        """
        u.debug_print(1, "Copying", self.path, "to", destination)
        if os.path.isdir(destination):
            destination = os.path.join(destination, self.filename)
        shutil.copy(self.path, destination)
        new_image = self.new_image(path=destination)
        new_image.log = self.log.copy()
        u.debug_print(2, f"Image.copy(): {new_image}.log.log.keys() ==", new_image.log.log.keys())
        new_image.add_log(
            f"Copied from {self.path} to {destination}.",
            method=self.copy,
            input_path=self.path,
            output_path=destination
        )
        return new_image

    def copy_with_outputs(self, destination: str):
        new_image = self.copy(destination)
        self.update_output_file()
        shutil.copy(self.output_file, destination.replace(".fits", "_outputs.yaml"))
        new_image.load_output_file()
        return new_image

    def load_output_file(self):
        outputs = p.load_output_file(self)
        if outputs is not None:
            if "frame_type" in outputs:
                self.frame_type = outputs["frame_type"]
            if "log" in outputs:
                self.log = log.Log(outputs["log"])
        return outputs

    def update_output_file(self):
        p.update_output_file(self)

    def _output_dict(self):
        return {
            "frame_type": self.frame_type,
            "log": self.log.to_dict(),
        }

    def load_headers(self, force: bool = False, **kwargs):
        if self.headers is None or force:
            self.open()
            self.headers = list(map(lambda h: h.header, self.hdu_list))
            self.close()
        else:
            u.debug_print(2, "Headers already loaded.")
        return self.headers

    def load_data(self, force: bool = False):
        if self.data is None or force:
            unit = self.extract_unit()
            self.open()
            u.debug_print(1, f"Image.load_data() 1: {self}.hdu_list:", self.hdu_list)
            if unit is not None:
                try:
                    unit = units.Unit(unit)
                    self.data = list(map(lambda h: h.data * unit, self.hdu_list))
                # If unit could not be parsed, assume counts
                except ValueError:
                    self.data = list(map(lambda h: h.data * units.ct, self.hdu_list))
            else:
                # If unit does not exist, assume counts
                u.debug_print(1, f"Image.load_data(): {self}.path:", self.path)
                u.debug_print(1, f"Image.load_data() 2: {self}.hdu_list:", self.hdu_list)
                self.data = list(map(lambda h: h.data * units.ct, self.hdu_list))
            self.close()
        else:
            u.debug_print(1, "Data already loaded.")
        return self.data

    def to_ccddata(self, unit: Union[str, units.Unit]):
        if unit is None:
            return ccdproc.CCDData.read(self.path)
        else:
            return ccdproc.CCDData.read(self.path, unit=unit)

    @classmethod
    def from_ccddata(self, ccddata: ccdproc.CCDData, path: str):
        pass

    def get_id(self):
        return self.filename[:self.filename.find(".fits")]

    def set_header_items(self, items: dict, ext: int = 0, write: bool = True):
        for key in items:
            self.set_header_item(
                key=key,
                value=items[key],
                ext=ext,
                write=False
            )
        if write:
            self.write_fits_file()

    def set_header_item(self, key: str, value, ext: int = 0, write: bool = False):
        self.load_headers()
        value = u.dequantify(value)

        if key in self.headers[ext]:
            old_value = self.headers[ext][key]
            action = f"Changed FITS header item {key} from {old_value} to {value} on ext {ext}."
        else:
            action = f"Created new FITS header item {key} with value {value} on ext {ext}."

        u.debug_print(1, "Image.set_header_item(): key, value ==", key, value)
        self.headers[ext][key] = value
        # self.close()

        self.add_log(
            action=action, method=self.set_header_item, ext=ext
        )
        if write:
            self.write_fits_file()

    def add_history(self, note: str, ext: int = 0):
        self.load_headers()
        self.headers[ext]["HISTORY"] = str(Time.now()) + ": " + note
        # self.write_fits_file()

    def _extract_header_item(self, key: str, ext: int = 0):
        self.load_headers()
        if key in self.headers[ext]:
            return self.headers[ext][key]
        else:
            return None

    def extract_header_item(self, key: str, ext: int = 0):
        # Check in the given HDU, then check all headers.
        value = self._extract_header_item(key=key, ext=ext)
        u.debug_print(2, "")
        u.debug_print(2, "Image.extract_header_item():")
        u.debug_print(2, f"\t{self}.path ==", self.path)
        u.debug_print(2, f"\t key ==", key)
        u.debug_print(2, f"\t value ==", value)
        u.debug_print(2, "")
        if value is None:
            for ext in range(len(self.headers)):
                value = self._extract_header_item(key=key, ext=ext)
                if value is not None:
                    return value
            # Then, if we get to the end of the loop, the item clearly doesn't exist.
            return None
        else:
            return value

    def extract_unit(self):
        key = self.header_keys()["unit"]
        return self.extract_header_item(key)

    def extract_gain(self):
        self.gain = self.extract_header_item("GAIN")
        if self.gain is None:
            key = self.header_keys()["gain"]
            u.debug_print(2, f"Image.extract_gain(): type({self})", type(self), key)
            self.gain = self.extract_header_item(key) * units.electron / units.ct
        if self.gain is not None:
            self.gain *= units.electron / units.ct
        return self.gain

    def extract_date_obs(self):
        key = self.header_keys()["date-obs"]
        self.date_obs = self.extract_header_item(key)
        key = self.header_keys()["mjd-obs"]
        self.mjd_obs = self.extract_header_item(key)
        return self.date_obs

    def extract_exposure_time(self):
        key = self.header_keys()["exposure_time"]
        self.exposure_time = self.extract_header_item(key) * units.second
        return self.exposure_time

    def extract_noise_read(self):
        key = self.header_keys()["noise_read"]
        noise = self.extract_header_item(key)
        if noise is not None:
            self.noise_read = self.extract_header_item(key) * units.electron / units.pixel
        else:
            raise KeyError(f"{key} not present in header.")
        return self.noise_read

    def extract_object(self):
        key = self.header_keys()["object"]
        self.object = self.extract_header_item(key)

        return self.object

    def extract_n_pix(self, ext: int = 0):
        self.load_data()
        self.n_y, self.n_x = self.data[ext].shape
        self.n_pix = self.n_y * self.n_x
        return self.n_pix

    def extract_pixel_edges(self):
        """
        Using the FITS convention of origin = 1, 1, returns the pixel coordinates of the edges.
        :return: tuple: left, right, bottom, top
        """
        self.extract_n_pix()
        return 1, self.n_x, 1, self.n_y

    def extract_saturate(self):
        key = self.header_keys()["saturate"]
        saturate = self.extract_header_item(key)
        if saturate is None:
            saturate = 65535
        self.saturate = saturate * units.ct
        return self.saturate

    def write_fits_file(self):
        with fits.open(self.path, mode="update") as file:
            for i in range(len(self.headers)):
                if self.headers is not None:
                    if i >= len(file):
                        file.append(fits.ImageHDU())
                    file[i].header = self.headers[i]
                if self.data is not None:
                    file[i].data = u.dequantify(self.data[i])

    @classmethod
    def header_keys(cls):
        header_keys = {
            "exposure_time": "EXPTIME",
            "noise_read": "RON",
            "gain": "GAIN",
            "date-obs": "DATE-OBS",
            "mjd-obs": "MJD-OBS",
            "object": "OBJECT",
            "instrument": "INSTRUME",
            "unit": "BUNIT",
            "saturate": "SATURATE"}
        return header_keys

    @classmethod
    def from_fits(cls, path: str, mode: str = "imaging"):
        # Load fits file
        hdu_list = fits.open(path)
        # First, check for instrument information in each header.
        instrument = None
        i = 0
        # Will need to add cases to the below instruments as you deal with new instruments.
        while instrument is None and i < len(hdu_list):
            header = hdu_list[i].header
            if "INSTRUME" in header:
                instrument = header["INSTRUME"]
            elif "FPA.INSTRUMENT" in header:
                instrument = "panstarrs1"
            i += 1

        if instrument is None:
            print("Instrument could not be determined from header.")
            child = ImagingImage
        else:
            # Look for standard instrument name in list
            if instrument in instrument_header:
                instrument = instrument_header[instrument]
                child = cls.select_child_class(instrument=instrument, mode=mode)
            else:
                child = ImagingImage
        u.debug_print(2, "Image.from_fits(): instrument ==", instrument)
        img = child(path=path, instrument_name=instrument)
        img.instrument_name = instrument
        return img

    @classmethod
    def select_child_class(cls, instrument: str, **kwargs):
        instrument = instrument.lower()
        if 'mode' in kwargs:
            mode = kwargs['mode']
            if mode == 'imaging':
                return ImagingImage.select_child_class(instrument=instrument, **kwargs)
            elif mode == 'spectroscopy':
                return Spectrum.select_child_class(instrument=instrument, **kwargs)
            else:
                raise ValueError(f"Unrecognised mode {mode}")
        else:
            raise KeyError(f"mode must be provided for {cls}.select_child_class()")


class ESOImage(Image):
    """
    Generic parent class for ESO images, both spectra and imaging
    """

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({"mode": "HIERARCH ESO INS MODE"})
        return header_keys


class ImagingImage(Image):
    def __init__(self, path: str, frame_type: str = None, instrument_name: str = None):
        super().__init__(path=path, frame_type=frame_type, instrument_name=instrument_name)

        self.wcs = None

        self.filter_name = None
        self.filter_short = None
        self.filter = None
        self.pixel_scale_ra = None
        self.pixel_scale_dec = None

        self.psfex_path = None
        self.psfex_output = None
        self.psfex_successful = None
        self.source_cat_sextractor_path = None
        self.source_cat_sextractor_dual_path = None
        self.source_cat_path = None
        self.source_cat_dual_path = None
        self.source_cat = None
        self.source_cat_dual = None
        self.dual_mode_template = None

        self.sep_background = None
        self.pu_background = None
        self.data_sub_bkg = None

        self.synth_cat_path = None
        self.synth_cat = None

        self.fwhm_pix_psfex = None
        self.fwhm_psfex = None

        self.fwhm_max_moffat = None
        self.fwhm_median_moffat = None
        self.fwhm_min_moffat = None
        self.fwhm_sigma_moffat = None
        self.fwhm_rms_moffat = None

        self.psf_stats = {}

        self.fwhm_max_gauss = None
        self.fwhm_median_gauss = None
        self.fwhm_min_gauss = None
        self.fwhm_sigma_gauss = None
        self.fwhm_rms_gauss = None

        self.fwhm_max_sextractor = None
        self.fwhm_median_sextractor = None
        self.fwhm_min_sextractor = None
        self.fwhm_sigma_sextractor = None
        self.fwhm_rms_sextractor = None

        self.astrometry_err = None

        self.sky_background = None

        self.airmass = None

        self.zeropoints = {}
        self.zeropoint_output_paths = {}
        self.zeropoint_best = None

        self.extinction_atmospheric = None
        self.extinction_atmospheric_err = None

        self.depth = {}

        self.astrometry_corrected_path = None
        self.astrometry_stats = {}

        self.extract_filter()

        self.load_output_file()

    def source_extraction(
            self, configuration_file: str,
            output_dir: str,
            parameters_file: str = None,
            catalog_name: str = None,
            template: 'ImagingImage' = None,
            **configs) -> str:
        if template is not None:
            template = template.path
            self.dual_mode_template = template
        self.extract_gain()
        u.debug_print(2, f"ImagingImage.source_extraction(): template ==", template)
        output_path = se.source_extractor(
            image_path=self.path,
            output_dir=output_dir,
            configuration_file=configuration_file,
            parameters_file=parameters_file,
            catalog_name=catalog_name,
            template_image_path=template,
            gain=self.gain.value,
            **configs
        )
        self.add_log(
            action="Sources extracted using Source Extractor.",
            method=self.source_extraction,
            output_path=output_dir,
            packages=["source-extractor"]
        )
        self.update_output_file()
        return output_path

    def psfex(self, output_dir: str, force: bool = False, **kwargs):
        if force or self.psfex_path is None:
            config = p.path_to_config_sextractor_config_pre_psfex()
            output_params = p.path_to_config_sextractor_param_pre_psfex()
            catalog = self.source_extraction(
                configuration_file=config,
                output_dir=output_dir,
                parameters_file=output_params,
                catalog_name=f"{self.name}_psfex.fits",
            )
            self.psfex_path = psfex.psfex(catalog=catalog, output_dir=output_dir, **kwargs)
            self.psfex_output = fits.open(self.psfex_path)
            self.extract_pixel_scale()
            pix_scale = self.pixel_scale_dec
            self.fwhm_pix_psfex = self.psfex_output[1].header['PSF_FWHM'] * units.pixel
            self.fwhm_psfex = self.fwhm_pix_psfex.to(units.arcsec, pix_scale)

            self.add_log(
                action="PSF modelled using psfex.",
                method=self.psfex,
                output_path=output_dir,
                packages=["psfex"]
            )
            self.update_output_file()

        return self.load_psfex_output()

    def load_psfex_output(self, force: bool = False):
        if force or self.psfex_output is None:
            self.psfex_output = fits.open(self.psfex_path)

    def source_extraction_psf(
            self,
            output_dir: str,
            template: 'ImagingImage' = None,
            force: bool = False,
            **configs):
        """
        Uses a PSFEx-generated PSF model in conjunction with Source Extractor to generate a source catalog. The key
        difference with source_extraction is that source_extraction uses only Source Extractor, and does not therefore
        use PSF-fitting (ie, no CLASS_STAR, MAG_PSF or FLUX_PSF columns are written).
        :param output_dir: The directory in which to write the PSFEx and Source Extractor output files.
        :param template: The path to the file to use as template, if dual mode is to be used.
        :param force: If True, performs all functions regardless of whether source_catalogues already exist, and
            overwrites them; if False, checks whether they exist first and skips some steps if so.
        :param configs: A dictionary of Source Extractor arguments to pass to command line.
        :return:
        """
        self.psfex(output_dir=output_dir, force=force)
        config = p.path_to_config_sextractor_config()
        output_params = p.path_to_config_sextractor_param()
        cat_path = self.source_extraction(
            configuration_file=config,
            output_dir=output_dir,
            parameters_file=output_params,
            catalog_name=f"{self.name}_psf-fit.cat",
            psf_name=self.psfex_path,
            seeing_fwhm=self.fwhm_psfex.value,
            template=template,
            **configs
        )
        if template is not None:
            self.source_cat_sextractor_dual_path = cat_path
        else:
            self.source_cat_sextractor_path = cat_path
        self.load_source_cat_sextractor(force=True)
        self.load_source_cat_sextractor_dual(force=True)

        if template is not None:
            cat = self.source_cat_dual
        else:
            cat = self.source_cat

        if len(cat) == 0:
            print()
            print("PSF source extraction was unsuccessful, probably due to lack of viable sources. Trying again without"
                  " PSFEx.")
            print()
            self.psfex_successful = False
            cat_path = self.source_extraction(
                configuration_file=p.path_to_config_sextractor_failed_psfex_config(),
                output_dir=output_dir,
                parameters_file=p.path_to_config_sextractor_failed_psfex_param(),
                catalog_name=f"{self.name}.cat",
                template=template,
                **configs
            )
            if template is not None:
                self.source_cat_sextractor_dual_path = cat_path
            else:
                self.source_cat_sextractor_path = cat_path
            self.load_source_cat_sextractor(force=True)
            self.load_source_cat_sextractor_dual(force=True)
        else:
            self.psfex_successful = True
        self.write_source_cat()
        self.add_log(
            action="Sources extracted using Source Extractor with PSFEx PSF modelling.",
            method=self.source_extraction_psf,
            output_path=output_dir,
            packages=["psfex", "source-extractor"]
        )
        self.update_output_file()

    def _load_source_cat_sextractor(self, path: str):
        self.load_wcs()
        print("Loading source catalogue from", path)
        source_cat = table.QTable.read(path, format="ascii.sextractor")
        source_cat["RA"], source_cat["DEC"] = self.wcs.all_pix2world(
            source_cat["X_IMAGE"],
            source_cat["Y_IMAGE"],
            1
        ) * units.deg

        return source_cat

    def load_data(self, force: bool = False):
        super().load_data()
        self.data_sub_bkg = [None] * len(self.data)
        self.sep_background = [None] * len(self.data)
        self.pu_background = [None] * len(self.data)
        return self.data

    def load_source_cat_sextractor(self, force: bool = False):
        if self.source_cat_sextractor_path is not None:
            if force:
                self.source_cat = None
            if self.source_cat is None:
                self.source_cat = self._load_source_cat_sextractor(path=self.source_cat_sextractor_path)
        else:
            print("source_cat could not be loaded because source_cat_sextractor_path has not been set.")

    def load_source_cat_sextractor_dual(self, force: bool = False):
        if self.source_cat_sextractor_dual_path is not None:
            if force:
                self.source_cat_dual = None
            if self.source_cat_dual is None:
                self.source_cat_dual = self._load_source_cat_sextractor(path=self.source_cat_sextractor_dual_path)
        else:
            print("source_cat_dual could not be loaded because source_cat_sextractor_dual_path has not been set.")

    def load_source_cat(self, force: bool = False):
        u.debug_print(2, f"ImagingImage.load_source_cat(): {self}.name ==", self.name)
        u.debug_print(2, f"ImagingImage.load_source_cat(): {self}.source_cat_path ==", self.source_cat_path)
        if force or self.source_cat is None or self.source_cat_dual is None:
            if self.source_cat_path is not None:
                u.debug_print(1, "Loading source_table from", self.source_cat_path)
                self.source_cat = table.QTable.read(self.source_cat_path, format="ascii.ecsv")
            elif self.source_cat_sextractor_path is not None:
                self.load_source_cat_sextractor(force=force)
            else:
                u.debug_print(1, "No valid source_cat_path found. Could not load source_table.")

            if self.source_cat_dual_path is not None:
                u.debug_print(1, "Loading source_table from", self.source_cat_dual_path)
                self.source_cat_dual = table.QTable.read(self.source_cat_dual_path, format="ascii.ecsv")
            elif self.source_cat_sextractor_dual_path is not None:
                self.load_source_cat_sextractor_dual(force=force)
            else:
                u.debug_print(1, "No valid source_cat_dual_path found. Could not load source_table.")

    def get_source_cat(self, dual: bool, force: bool = False):
        self.load_source_cat(force=force)
        if dual:
            source_cat = self.source_cat_dual
        else:
            source_cat = self.source_cat

        return source_cat

    def _set_source_cat(self, source_cat: table.QTable, dual: bool):
        """
        CAUTION. This will overwrite any saved source_cat both on disk and in memory. Recommended that this only be used when columns
        have been added to the existing source_cat.
        :param source_cat: QTable to make this object's
        :param dual:
        :return:
        """
        if dual:
            self.source_cat_dual = source_cat
        else:
            self.source_cat = source_cat
        self.update_output_file()

    def write_source_cat(self):
        if self.source_cat is None:
            u.debug_print(1, "source_cat not yet loaded.")
        else:
            if self.source_cat_path is None:
                self.source_cat_path = self.path.replace(".fits", "_source_cat.ecsv")
            u.debug_print(1, "Writing source catalogue to", self.source_cat_path)
            self.source_cat.write(self.source_cat_path, format="ascii.ecsv", overwrite=True)

        if self.source_cat_dual is None:
            u.debug_print(1, "source_cat_dual not yet loaded.")
        else:
            if self.source_cat_dual_path is None:
                self.source_cat_dual_path = self.path.replace(".fits", "_source_cat_dual.ecsv")
            u.debug_print(1, "Writing dual-mode source catalogue to", self.source_cat_dual_path)
            self.source_cat_dual.write(self.source_cat_dual_path, format="ascii.ecsv", overwrite=True)

    def load_synth_cat(self, force: bool = False):
        if force or self.synth_cat is None:
            if self.synth_cat_path is not None:
                u.debug_print(2, f"ImagingImage.load_synth_cat(): {self}.synth_cat_path ==", self.synth_cat_path)
                self.synth_cat = table.QTable.read(self.synth_cat_path, format="ascii.ecsv")
            else:
                u.debug_print(1, "No valid synth_cat_path found. Could not load synth_cat.")
            return self.synth_cat

    def write_synth_cat(self):

        if self.synth_cat is None:
            u.debug_print(1, "synth_cat not yet loaded.")
        else:
            if self.synth_cat_path is None:
                self.synth_cat_path = self.path.replace(".fits", "_synth_cat.ecsv")
            u.debug_print(1, "Writing source catalogue to", self.synth_cat_path)
            self.synth_cat.write(self.synth_cat_path, format="ascii.ecsv", overwrite=True)

    def load_wcs(self, ext: int = 0) -> wcs.WCS:
        self.load_headers()
        self.wcs = wcs.WCS(header=self.headers[ext])
        return self.wcs

    def extract_astrometry_err(self):
        key = self.header_keys()["astrometry_err"]
        self.astrometry_err = self.extract_header_item(key)
        if self.astrometry_err is not None:
            self.astrometry_err *= units.arcsec
        return self.astrometry_err

    def extract_rotation_angle(self, ext: int = 0):
        self.load_headers()
        return ff.get_rotation_angle(header=self.headers[ext])

    def extract_wcs_footprint(self):
        """
        Returns the RA & Dec of the corners of the image.
        :return: tuple of SkyCoords, (top_left, top_right, bottom_left, bottom_right)
        """
        self.load_wcs()
        return self.wcs.calc_footprint()

    def extract_pixel_scale(self, layer: int = 0, force: bool = False):
        if force or self.pixel_scale_ra is None or self.pixel_scale_dec is None:
            self.open()
            self.pixel_scale_ra, self.pixel_scale_dec = ff.get_pixel_scale(self.hdu_list, layer=layer,
                                                                           astropy_units=True)
            self.close()
        else:
            u.debug_print(2, "Pixel scale already set.")

        return self.pixel_scale_ra, self.pixel_scale_dec

    def extract_filter(self):
        key = self.header_keys()["filter"]
        self.filter_name = self.extract_header_item(key)
        if self.filter_name is not None:
            self.filter_short = self.filter_name[0]

        if self.filter_name is not None and self.instrument is not None and self.filter_name in self.instrument.filters:
            self.filter = self.instrument.filters[self.filter_name]

        return self.filter_name

    def extract_airmass(self):
        key = self.header_keys()["airmass"]
        self.airmass = self.extract_header_item(key)
        return self.airmass

    def extract_pointing(self):
        key = self.header_keys()["ra"]
        ra = self.extract_header_item(key)
        key = self.header_keys()["dec"]
        dec = self.extract_header_item(key)
        self.pointing = SkyCoord(ra, dec, unit=units.deg)
        return self.pointing

    def extract_ref_pixel(self) -> Tuple[float]:
        """
        Retrieve the coordinates of the "reference pixel" from the header.
        :return: Tuple containing the reference pixel coordinates as (x, y).
        """
        key = self.header_keys()["ref_pix_x"]
        x = self.extract_header_item(key)
        key = self.header_keys()["ref_pix_y"]
        y = self.extract_header_item(key)
        return x, y

    def extract_old_pointing(self):
        key = self.header_keys()["ra_old"]
        ra = self.extract_header_item(key)
        key = self.header_keys()["dec_old"]
        dec = self.extract_header_item(key)
        return SkyCoord(ra, dec, unit=units.deg)

    def _output_dict(self):
        outputs = super()._output_dict()
        outputs.update({
            "astrometry_stats": self.astrometry_stats,
            "extinction_atmospheric": self.extinction_atmospheric,
            "extinction_atmospheric_err": self.extinction_atmospheric_err,
            "filter": self.filter_name,
            "psfex_path": self.psfex_path,
            "source_cat_sextractor_path": self.source_cat_sextractor_path,
            "source_cat_sextractor_dual_path": self.source_cat_sextractor_dual_path,
            "source_cat_path": self.source_cat_path,
            "source_cat_dual_path": self.source_cat_dual_path,
            "synth_cat_path": self.synth_cat_path,
            "fwhm_pix_psfex": self.fwhm_pix_psfex,
            "fwhm_psfex": self.fwhm_psfex,
            "psfex_succesful": self.psfex_successful,
            "zeropoints": self.zeropoints,
            "zeropoint_output_paths": self.zeropoint_output_paths,
            "zeropoint_best": self.zeropoint_best,
            "depth": self.depth,
            "dual_mode_template": self.dual_mode_template,
        })
        return outputs

    def update_output_file(self):
        p.update_output_file(self)
        self.write_source_cat()
        self.write_synth_cat()

    def load_output_file(self):
        outputs = super().load_output_file()
        if outputs is not None:
            if "astrometry_stats" in outputs:
                self.astrometry_stats = outputs["astrometry_stats"]
            if "extinction_atmospheric" in outputs:
                self.extinction_atmospheric = outputs["extinction_atmospheric"]
            if "extinction_atmospheric_err" in outputs:
                self.extinction_atmospheric_err = outputs["extinction_atmospheric_err"]
            if "filter" in outputs:
                self.filter_name = outputs["filter"]
            if "psfex_path" in outputs:
                self.psfex_path = outputs["psfex_path"]
            if "source_cat_sextractor_path" in outputs:
                self.source_cat_sextractor_path = outputs["source_cat_sextractor_path"]
            if "source_cat_sextractor_dual_path" in outputs:
                self.source_cat_sextractor_path = outputs["source_cat_sextractor_dual_path"]
            if "source_cat_path" in outputs:
                self.source_cat_path = outputs["source_cat_path"]
            if "synth_cat_path" in outputs:
                self.synth_cat_path = outputs["synth_cat_path"]
            if "source_cat_dual_path" in outputs:
                self.source_cat_dual_path = outputs["source_cat_dual_path"]
            if "fwhm_psfex" in outputs:
                self.fwhm_psfex = outputs["fwhm_psfex"]
            if "fwhm_psfex" in outputs:
                self.fwhm_pix_psfex = outputs["fwhm_pix_psfex"]
            if "psf_stats" in outputs:
                self.psf_stats = outputs["psf_stats"]
            if "psfex_successful" in outputs:
                self.psfex_successful = outputs["psfex_successful"]
            if "zeropoints" in outputs:
                self.zeropoints = outputs["zeropoints"]
            if "zeropoint_output_paths" in outputs:
                self.zeropoint_output_paths = outputs["zeropoint_output_paths"]
            if "zeropoint_best" in outputs:
                self.zeropoint_best = outputs["zeropoint_best"]
            if "depth" in outputs and outputs["depth"] is not None:
                self.depth = outputs["depth"]
            if "dual_mode_template" in outputs and outputs["dual_mode_template"] is not None:
                self.dual_mode_template = outputs["dual_mode_template"]
        u.debug_print(2, f"ImagingImage.load_output_file(): {self}.source_cat_path ==", self.source_cat_path)
        return outputs

    def select_zeropoint(self, no_user_input: bool = False, preferred: str = None):

        ranking = self.rank_photometric_cat()
        if preferred is not None:
            ranking.insert(0, preferred)
        zps = {}
        best = None
        for cat in ranking:
            if cat in self.zeropoints:
                zp = self.zeropoints[cat]
                zps[f"{cat} {zp['zeropoint']} +/- {zp['zeropoint_err']} {zp['n_matches']}"] = zp
                if best is None:
                    best = cat

        if best is None:
            raise ValueError("No zeropoints are present to select from.")

        zeropoint_best = self.zeropoints[best]
        print(
            f"For {self.name}, we have selected a zeropoint of {zeropoint_best['zeropoint']} +/- {zeropoint_best['zeropoint_err']}, "
            f"from {zeropoint_best['catalogue']}.")
        if not no_user_input:
            select_own = u.select_yn(message="Would you like to select another?", default=False)
            if select_own:
                _, zeropoint_best = u.select_option(message="Select best zeropoint:", options=zps)
                best = zeropoint_best["catalogue"]
        self.zeropoint_best = zeropoint_best

        self.set_header_items(
            items={
                "PHOTZP": zeropoint_best["zeropoint"] - zeropoint_best["extinction"] * zeropoint_best["airmass"],
                "PHOTZPER": np.sqrt(
                    zeropoint_best["zeropoint_err"] ** 2 + u.uncertainty_product(
                        zeropoint_best["extinction"] * zeropoint_best["airmass"],
                        (zeropoint_best["extinction"], zeropoint_best["extinction_err"]),
                        (zeropoint_best["airmass"], zeropoint_best["airmass_err"])
                    ) ** 2
                ),
                "ZPCAT": zeropoint_best["catalogue"]
            },
            ext=0,
            write=False
        )

        self.add_log(
            action=f"Selected best zeropoint as {zeropoint_best['zeropoint']} +/- {zeropoint_best['zeropoint_err']}, from {zeropoint_best['catalogue']}",
            method=self.select_zeropoint
        )
        self.update_output_file()
        self.write_fits_file()
        return self.zeropoint_best, best

    def zeropoint(
            self,
            cat_path: str,
            output_path: str,
            cat_name: str = 'Catalogue',
            cat_zeropoint: units.Quantity = 0.0 * units.mag,
            cat_zeropoint_err: units.Quantity = 0.0 * units.mag,
            image_name: str = None,
            show: bool = False,
            sex_x_col: str = 'XPSF_IMAGE',
            sex_y_col: str = 'YPSF_IMAGE',
            sex_ra_col: str = 'RA',
            sex_dec_col: str = 'DEC',
            sex_flux_col: str = 'FLUX_PSF',
            stars_only: bool = True,
            star_class_col: str = 'CLASS_STAR',
            star_class_tol: float = 0.95,
            mag_range_sex_lower: units.Quantity = -100. * units.mag,
            mag_range_sex_upper: units.Quantity = 100. * units.mag,
            dist_tol: units.Quantity = None,
            snr_cut=100
    ):
        self.signal_to_noise_measure()
        if image_name is None:
            image_name = self.name
        self.extract_filter()
        print("FILTER:", self.filter_short)
        print("CAT_NAME:", cat_name)
        column_names = cat_columns(cat=cat_name, f=self.filter_short)
        print("MAG NAME:", column_names['mag_psf'])
        cat_ra_col = column_names['ra']
        cat_dec_col = column_names['dec']
        cat_mag_col = column_names['mag_psf']
        cat_type = "csv"

        if dist_tol is None:
            self.load_headers()
            self.extract_astrometry_err()
            if self.astrometry_err is not None:
                dist_tol = 2 * self.astrometry_err
            else:
                dist_tol = 2 * units.arcsec

        zp_dict = ph.determine_zeropoint_sextractor(
            sextractor_cat=self.source_cat,
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
            snr_col='SNR_SE',
            snr_cut=snr_cut,
        )

        if zp_dict is None:
            return None
        zp_dict['airmass'] = 0.0
        zp_dict['airmass_err'] = 0.0
        zp_dict['extinction'] = 0.0 * units.mag
        zp_dict['extinction_err'] = 0.0 * units.mag
        self.zeropoints[cat_name.lower()] = zp_dict
        self.zeropoint_output_paths[cat_name.lower()] = output_path
        self.add_log(
            action=f"Calculated zeropoint as {zp_dict['zeropoint']} +/- {zp_dict['zeropoint_err']}, from {zp_dict['catalogue']}.",
            method=self.zeropoint,
            output_path=output_path
        )
        self.update_output_file()
        return self.zeropoints[cat_name.lower()]

    def aperture_areas(self):
        self.load_source_cat()
        self.extract_pixel_scale()

        self.source_cat["A_IMAGE"] = self.source_cat["A_WORLD"].to(units.pix, self.pixel_scale_dec)
        self.source_cat["B_IMAGE"] = self.source_cat["A_WORLD"].to(units.pix, self.pixel_scale_dec)
        self.source_cat["KRON_AREA_IMAGE"] = self.source_cat["A_IMAGE"] * self.source_cat["B_IMAGE"] * np.pi

        if self.source_cat_dual is not None:
            self.source_cat_dual["A_IMAGE"] = self.source_cat_dual["A_WORLD"].to(units.pix, self.pixel_scale_dec)
            self.source_cat_dual["B_IMAGE"] = self.source_cat_dual["A_WORLD"].to(units.pix, self.pixel_scale_dec)
            self.source_cat_dual["KRON_AREA_IMAGE"] = self.source_cat_dual["A_IMAGE"] * self.source_cat_dual[
                "B_IMAGE"] * np.pi

        self.add_log(
            action=f"Calculated area of FLUX_AUTO apertures.",
            method=self.aperture_areas,
        )
        self.update_output_file()

    def calibrate_magnitudes(self, zeropoint_name: str = "best", force: bool = False, dual: bool = False):
        cat = self.get_source_cat(dual=dual, force=True)

        self.extract_exposure_time()

        if force or f"MAG_AUTO_ZP_{zeropoint_name}" not in cat:
            mags = self.magnitude(
                flux=cat["FLUX_AUTO"],
                flux_err=cat["FLUXERR_AUTO"],
                zeropoint_name=zeropoint_name
            )

            cat[f"MAG_AUTO_ZP_{zeropoint_name}"] = mags[0]
            cat[f"MAGERR_AUTO_ZP_{zeropoint_name}"] = mags[1]
            cat[f"MAG_AUTO_ZP_{zeropoint_name}_no_ext"] = mags[2]
            cat[f"MAGERR_AUTO_ZP_{zeropoint_name}_no_ext"] = mags[3]

            mags = self.magnitude(
                flux=cat["FLUX_PSF"],
                flux_err=cat["FLUXERR_PSF"],
                zeropoint_name=zeropoint_name
            )

            cat[f"MAG_PSF_ZP_{zeropoint_name}"] = mags[0]
            cat[f"MAGERR_PSF_ZP_{zeropoint_name}"] = mags[1]
            cat[f"MAG_PSF_ZP_{zeropoint_name}_no_ext"] = mags[2]
            cat[f"MAGERR_PSF_ZP_{zeropoint_name}_no_ext"] = mags[3]

            self._set_source_cat(source_cat=cat, dual=dual)

            self.add_log(
                action=f"Calibrated source catalogue magnitudes using zeropoint {zeropoint_name}.",
                method=self.calibrate_magnitudes,
            )
            self.update_output_file()

        else:
            print(f"Magnitudes already calibrated for {zeropoint_name}")

    def magnitude(
            self,
            flux: units.Quantity,
            flux_err: units.Quantity = 0 * units.ct,
            zeropoint_name: str = 'best'
    ):

        if zeropoint_name == "best":
            zp_dict = self.zeropoint_best
        elif zeropoint_name not in self.zeropoints:
            raise KeyError(f"Zeropoint {zeropoint_name} does not exist.")
        else:
            zp_dict = self.zeropoints[zeropoint_name]

        mag, mag_err = ph.magnitude_complete(
            flux=flux,
            flux_err=flux_err,
            exp_time=self.extract_exposure_time(),
            exp_time_err=0.0 * units.second,
            zeropoint=zp_dict['zeropoint'],
            zeropoint_err=zp_dict['zeropoint_err'],
            airmass=zp_dict['airmass'],
            airmass_err=zp_dict['airmass_err'],
            ext=zp_dict['extinction'],
            ext_err=zp_dict['extinction_err'],
            colour_term=0.0,
            colour=0.0 * units.mag,
        )

        mag_no_ext_corr, mag_no_ext_corr_err = ph.magnitude_complete(
            flux=flux,
            flux_err=flux_err,
            exp_time=self.extract_exposure_time(),
            exp_time_err=0.0 * units.second,
            zeropoint=zp_dict['zeropoint'],
            zeropoint_err=zp_dict['zeropoint_err'],
            airmass=zp_dict['airmass'],
            airmass_err=zp_dict['airmass_err'],
            ext=0.0 * units.mag,
            ext_err=0.0 * units.mag,
            colour_term=0.0,
            colour=0.0 * units.mag,
        )

        return mag, mag_err, mag_no_ext_corr, mag_no_ext_corr_err

    def estimate_depth(self, zeropoint_name: str, dual: bool = False):
        """
        Use various measures of S/N to estimate image depth at a range of sigmas.
        :param zeropoint_name:
        :param dual:
        :return:
        """

        self.signal_to_noise_ccd(dual=dual)
        self.signal_to_noise_measure(dual=dual)
        self.calibrate_magnitudes(zeropoint_name=zeropoint_name, dual=dual)

        source_cat = self.get_source_cat(dual=dual)

        # "max" stores the magnitude of the faintest object with S/N > x sigma
        self.depth["max"] = {}
        # "secure" finds the brightest object with S/N < x sigma, then increments to the
        # overall; thus giving the faintest magnitude at which we can be confident of a detection
        self.depth["secure"] = {}

        # We do this to ensure that, in the "secure" step, object i+1 is the next-brightest in the catalogue
        source_cat.sort("FLUX_AUTO")

        for sigma in range(1, 6):
            for snr_key in ["SNR_CCD", "SNR_MEASURED", "SNR_SE"]:
                u.debug_print(1, "ImagingImage.estimate_depth(): snr_key, sigma ==", snr_key, sigma)
                self.depth["max"][snr_key] = {}
                self.depth["secure"][snr_key] = {}
                # Faintest source at x-sigma:
                u.debug_print(
                    1, f"ImagingImage.estimate_depth(): source_cat[{snr_key}].unit ==",
                    source_cat[snr_key].unit)
                cat_more_xsigma = source_cat[source_cat[snr_key] > sigma]
                print(f"Sources > {sigma}-sigma:", len(cat_more_xsigma))
                self.depth["max"][snr_key][f"{sigma}-sigma"] = np.max(cat_more_xsigma[f"MAG_AUTO_ZP_{zeropoint_name}"])

                # Brightest source less than x-sigma (kind of)
                source_less_sigma = source_cat[source_cat[snr_key] < sigma]
                if len(source_less_sigma) > 0:
                    source_less_sigma = source_less_sigma[source_less_sigma[snr_key] != np.inf]
                    source_less_sigma = source_less_sigma[source_less_sigma[snr_key] != np.nan]
                    i = np.argmax(source_less_sigma["FLUX_AUTO"])
                    i, _ = u.find_nearest(source_cat["NUMBER"], source_less_sigma[i]["NUMBER"])
                    i += 1
                    src_lim = source_cat[i]
                else:
                    src_lim = source_cat[0]
                self.depth["secure"][snr_key][f"{sigma}-sigma"] = src_lim[f"MAG_AUTO_ZP_{zeropoint_name}"]

        source_cat.sort("NUMBER")
        self.add_log(
            action=f"Estimated image depth.",
            method=self.estimate_depth,
        )
        self.update_output_file()
        return self.depth

    def send_column_to_source_cat(self, colname: str, sample: table.Table):
        """
        Takes values from an extra column added to a subset of source_cat.
        Trust me, it comes in handy.
        Assumes that NO entries have been removed from the main source_cat since being produced by Source Extractor,
        as it requires that the relationship source_cat["NUMBER"] = i - 1 holds true.
        (The commented line is for use when that assumption is no longer valid, but is slower).
        :param colname: Name of column to send.
        :param sample:
        :return:
        """
        if colname not in self.source_cat.colnames:
            self.source_cat.add_column(-99 * sample[colname].unit, name=colname)
        self.source_cat.sort("NUMBER")
        for star in sample:
            index = star["NUMBER"]
            # i = self.find_object_index(index, dual=False)
            self.source_cat[index - 1][colname] = star[colname]

    def register(self, target: 'ImagingImage', output_path: str, ext: int = 0, trim: bool = True):
        self.load_data()
        target.load_data()

        data_source = self.data[ext]
        data_source = u.sanitise_endianness(data_source)
        data_target = target.data[ext]
        data_target = u.sanitise_endianness(data_target)
        u.debug_print(1, f"Attempting registration of {self.name} against {target.name}")
        registered, footprint = register(data_source, data_target)

        self.copy(output_path)
        with fits.open(output_path, mode="update") as new_file:
            new_file[0].data = registered
            u.debug_print(1, "Writing registered image to", output_path)
            new_file.writeto(output_path, overwrite=True)

        new_image = self.new_image(path=output_path)
        new_image.transfer_wcs(target)

        if trim:
            frame_value = new_image.detect_frame_value(ext=ext)
            left, right, bottom, top = new_image.detect_edges(frame_value=frame_value)
            new_image.trim(left=left, right=right, bottom=bottom, top=top, output_path=output_path)

        new_image.add_log(
            action=f"Registered and reprojected to footprint of {target} using astroalign.",
            method=self.register,
        )
        new_image.update_output_file()
        return new_image

    def detect_frame_value(self, ext: int = 0):
        self.open()
        frame_value = ff.detect_frame_value(file=self.hdu_list, ext=ext)
        self.close()
        return frame_value

    def detect_edges(self, frame_value: float, ext: int = 0):
        self.open()
        left, right, bottom, top = ff.detect_edges(file=self.hdu_list, value=frame_value, ext=ext)
        self.close()
        return left, right, bottom, top

    def correct_astrometry(self, output_dir: str = None, tweak: bool = True, time_limit: int = None, **kwargs):
        """
        Uses astrometry.net to solve the astrometry of the image. Solved image is output as a separate file.
        :param output_dir: Directory in which to output
        :return: Path of corrected file.
        """
        self.extract_pointing()
        u.debug_print(1, "image.correct_astrometry(): tweak ==", tweak)
        if output_dir is not None:
            u.mkdir_check(output_dir)
        base_filename = f"{self.name}_astrometry"
        success = solve_field(
            image_files=self.path,
            base_filename=base_filename,
            overwrite=True,
            tweak=tweak,
            guess_scale=True,
            search_radius=0.5 * units.arcmin,
            centre=self.pointing,
            time_limit=time_limit
        )
        if not success:
            return None
        new_path = os.path.join(self.data_path, f"{base_filename}.new")
        new_new_path = os.path.join(self.data_path, f"{base_filename}.fits")
        os.rename(new_path, new_new_path)

        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise ValueError(f"Invalid output directory {output_dir}")
            for astrometry_product in filter(lambda f: f.startswith(base_filename), os.listdir(self.data_path)):
                path = os.path.join(self.data_path, astrometry_product)
                shutil.copy(path, output_dir)
                os.remove(path)
        else:
            output_dir = self.data_path
        final_file = os.path.join(output_dir, f"{base_filename}.fits")
        self.astrometry_corrected_path = final_file
        new_image = self.new_image(final_file)
        new_image.add_log(
            "Astrometry corrected using Astrometry.net.",
            method=self.correct_astrometry,
            packages=["astrometry.net"])
        new_image.update_output_file()
        return new_image

    def correct_astrometry_coarse(
            self,
            output_dir: str = None,
            cat: table.Table = None,
            ext: int = 0,
    ):
        self.load_source_cat()
        if self.source_cat is None:
            self.source_extraction_psf(output_dir=output_dir)

        if cat is None:
            if self.epoch is not None:
                cat = self.epoch.epoch_gaia_catalogue()
            else:
                raise ValueError(f"If image epoch is not assigned, cat must be provided.")
        diagnostics = self.astrometry_diagnostics(
            reference_cat=cat,
            offset_tolerance=3 * units.arcsec
        )
        new_path = os.path.join(output_dir, self.filename.replace(".fits", "_astrometry.fits"))
        new = self.copy(new_path)

        new.load_headers()
        if not np.isnan(diagnostics["median_offset_x"].value) and not np.isnan(diagnostics["median_offset_y"].value):
            new.headers[ext]["CRPIX1"] -= diagnostics["median_offset_x"].value
            new.headers[ext]["CRPIX2"] += diagnostics["median_offset_y"].value
            new.add_log(
                "Astrometry corrected using median offsets from reference catalogue.",
                method=self.correct_astrometry_coarse,
                input_path=self.path,
                output_path=new_path,
                ext=ext
            )
            new.write_fits_file()
            return new
        else:
            u.rm_check(new_path)
            return None

    def transfer_wcs(self, other_image: 'ImagingImage', ext: int = 0):
        other_image.load_headers()
        self.load_headers()
        self.headers[ext] = ff.wcs_transfer(header_template=other_image.headers[ext], header_update=self.headers[ext])
        self.add_log(
            f"Changed WCS information to match {other_image}.",
            method=self.transfer_wcs
        )
        self.update_output_file()
        self.write_fits_file()

    def correct_astrometry_from_other(self, other_image: 'ImagingImage', output_dir: str = None) -> 'ImagingImage':
        """
        Uses the header information from an image that has already been corrected by the Astrometry.net code to apply
        the same tweak to this image.
        This assumes that both images had the same astrometry to begin with, and is only really valid for use with an
        image that represents the same exposure but on a different CCD chip.
        :param other_image: Header must contain both _RVAL and CRVAL keywords.
        :param output_dir: Path to write new fits file to.
        :return:
        """
        if not isinstance(other_image, ImagingImage):
            raise ValueError("other_image is not a valid ImagingImage")
        other_header = other_image.load_headers()[0]

        output_path = os.path.join(output_dir, f"{self.name}_astrometry.fits")
        shutil.copyfile(self.path, output_path)

        # TODO: This method works, but does not preserve the order of header keys in the new file.
        # In fact, it makes rather a mess of them. Work out how to do this properly.

        # Take old astrometry info from other header
        start_index = other_header.index("_RVAL1") - 1
        end_index = other_header.index("_D2_2") + 1
        insert = other_header[start_index:end_index]

        # Take new astrometry info from other header
        start_index = other_header.index("WCSAXES") - 5
        end_index = start_index + 269
        insert.update(other_header[start_index:end_index])

        # Calculate offset, in other image and in world coordinates, of the new reference pixel from its old one.
        other_pointing = other_image.extract_pointing()
        other_old_pointing = other_image.extract_old_pointing()
        offset_ra = other_pointing.ra - other_old_pointing.ra
        offset_dec = other_pointing.dec - other_old_pointing.dec

        # Calculate the offset, in the other image and in pixels, of the new reference frame from its old one.
        offset_crpix1 = other_header["CRPIX1"] - other_header["_RPIX1"]
        offset_crpix2 = other_header["CRPIX2"] - other_header["_RPIX2"]

        with fits.open(output_path, "update") as file:
            # Apply the same offsets to this image, while keeping the old values as "_" keys
            insert["_RVAL1"] = file[0].header["CRVAL1"]
            insert["_RVAL2"] = file[0].header["CRVAL2"]
            insert["CRVAL1"] = insert["_RVAL1"] + offset_ra.value
            insert["CRVAL2"] = insert["_RVAL2"] + offset_dec.value

            insert["_RPIX1"] = file[0].header["CRPIX1"]
            insert["_RPIX2"] = file[0].header["CRPIX2"]
            insert["CRPIX1"] = insert["_RPIX1"] + offset_crpix1
            insert["CRPIX2"] = insert["_RPIX2"] + offset_crpix2

            # Insert all other astrometry info as previously extracted.
            file[0].header.update(insert)

        cls = ImagingImage.select_child_class(instrument=self.instrument_name)
        new_image = cls(path=output_path)

        new_image.add_log(
            f"Used WCS info from {other_image} to correct this image.",
            method=self.correct_astrometry_from_other
        )
        new_image.update_output_file()

        return new_image

    def astrometry_diagnostics(
            self,
            reference_cat: Union[str, table.QTable],
            ra_col: str = "ra", dec_col: str = "dec", mag_col: str = "phot_g_mean_mag",
            offset_tolerance: units.Quantity = 0.5 * units.arcsec,
            star_tolerance: float = 0.8,
            local_coord: SkyCoord = None,
            local_radius: units.Quantity = 0.5 * units.arcmin,
            show_plots: bool = False,
            output_path=None
    ):
        """
        Perform diagnostics of astrometric offset of stars in image from catalogue.
        :param reference_cat: Path to reference catalogue.
        :param ra_col:
        :param dec_col:
        :param mag_col:
        :param offset_tolerance: Maximum offset to be matched.
        :param star_tolerance: Minimum CLASS_STAR for object to be considered.
        :param local_coord:
        :param local_radius:
        :param show_plots:
        :param output_path:
        :return:
        """

        # quantity_support()

        if local_coord is None:
            local_coord = self.extract_pointing()

        if output_path is None:
            output_path = self.data_path

        self.load_source_cat()

        if isinstance(reference_cat, str):
            reference_cat = table.QTable.read(reference_cat)

        u.debug_print(2, "ImagingImage.astrometry_diagnostics(): reference_cat ==", reference_cat)
        u.debug_print(2, f"ImagingImage.astrometry_diagnostics(): {self}.source_cat ==", self.source_cat)

        plt.close()

        plt.scatter(self.source_cat["RA"], self.source_cat["DEC"], marker='x')
        plt.xlabel("Right Ascension (Catalogue)")
        plt.ylabel("Declination (Catalogue)")
        # plt.colorbar(label="Offset of measured position from catalogue (\")")
        if show_plots:
            plt.show()
        plt.savefig(os.path.join(output_path, f"{self.name}_sourcecat_sky.pdf"))
        plt.close()

        plt.scatter(reference_cat[ra_col], reference_cat[dec_col], marker='x')
        plt.xlabel("Right Ascension (Catalogue)")
        plt.ylabel("Declination (Catalogue)")
        # plt.colorbar(label="Offset of measured position from catalogue (\")")
        if show_plots:
            plt.show()
        plt.savefig(os.path.join(output_path, f"{self.name}_referencecat_sky.pdf"))
        plt.close()

        self.load_wcs()
        ref_cat_coords = SkyCoord(reference_cat[ra_col], reference_cat[dec_col])
        in_footprint = self.wcs.footprint_contains(ref_cat_coords)

        plt.scatter(self.source_cat["RA"],
                    self.source_cat["DEC"],
                    marker='x')
        plt.scatter(reference_cat[ra_col][in_footprint],
                    reference_cat[dec_col][in_footprint],
                    marker='x')
        plt.xlabel("Right Ascension (Catalogue)")
        plt.ylabel("Declination (Catalogue)")
        # plt.colorbar(label="Offset of measured position from catalogue (\")")
        if show_plots:
            plt.show()
        plt.savefig(os.path.join(output_path, f"{self.name}_bothcats_sky.pdf"))
        plt.close()

        matches_source_cat, matches_ext_cat, distance = self.match_to_cat(
            cat=reference_cat,
            ra_col=ra_col,
            dec_col=dec_col,
            offset_tolerance=offset_tolerance,
            star_tolerance=star_tolerance
        )

        matches_coord = SkyCoord(matches_source_cat["RA"], matches_source_cat["DEC"])

        mean_offset = np.mean(distance)
        median_offset = np.median(distance)
        rms_offset = np.sqrt(np.mean(distance ** 2))

        ref = self.extract_pointing()
        ref_distance = ref.separation(matches_coord)

        local_distance = local_coord.separation(matches_coord)
        distance_local = distance[local_distance <= local_radius]
        u.debug_print(2, distance_local)
        mean_offset_local = np.mean(distance_local)
        median_offset_local = np.median(distance_local)
        rms_offset_local = np.sqrt(np.mean(distance_local ** 2))

        plt.scatter(ref_distance.to(units.arcsec), distance.to(units.arcsec))
        plt.xlabel("Distance from reference pixel (\")")
        plt.ylabel("Offset (\")")
        if show_plots:
            plt.show()
        plt.savefig(os.path.join(output_path, f"{self.name}_astrometry_offset_v_ref.pdf"))
        plt.close()

        plt.hist(distance.to(units.arcsec).value)
        plt.xlabel("Offset (\")")
        if show_plots:
            plt.show()
        plt.savefig(os.path.join(output_path, f"{self.name}_astrometry_offset_hist.pdf"))
        plt.close()

        plt.scatter(matches_ext_cat[ra_col], matches_ext_cat[dec_col], c=distance.to(units.arcsec), marker='x')
        plt.xlabel("Right Ascension (Catalogue)")
        plt.ylabel("Declination (Catalogue)")
        plt.colorbar(label="Offset of measured position from catalogue (\")")
        if show_plots:
            plt.show()
        plt.savefig(os.path.join(output_path, f"{self.name}_astrometry_offset_sky.pdf"))
        plt.close()

        fig = plt.figure(figsize=(12, 12), dpi=1000)
        self.plot_catalogue(
            cat=reference_cat[in_footprint],
            ra_col=ra_col, dec_col=dec_col,
            fig=fig,
            colour_column=mag_col,
            cbar_label=mag_col)
        # fig.savefig(os.path.join(output_path, f"{self.name}_cat_overplot.pdf"))

        self.astrometry_stats["mean_offset"] = mean_offset.to(units.arcsec)
        self.astrometry_stats["median_offset"] = median_offset.to(units.arcsec)
        self.astrometry_stats["rms_offset"] = rms_offset.to(units.arcsec)
        self.astrometry_stats["median_offset_x"] = np.nanmedian(matches_source_cat["X_OFFSET_FROM_REF"])
        self.astrometry_stats["median_offset_y"] = np.nanmedian(matches_source_cat["Y_OFFSET_FROM_REF"])
        self.astrometry_stats["mean_offset_x"] = np.nanmean(matches_source_cat["X_OFFSET_FROM_REF"])
        self.astrometry_stats["mean_offset_y"] = np.nanmean(matches_source_cat["Y_OFFSET_FROM_REF"])

        self.astrometry_stats["mean_offset_local"] = mean_offset_local.to(units.arcsec)
        self.astrometry_stats["median_offset_local"] = median_offset_local.to(units.arcsec)
        self.astrometry_stats["rms_offset_local"] = rms_offset_local.to(units.arcsec)

        self.astrometry_stats["n_matches"] = len(matches_source_cat)
        self.astrometry_stats["n_cat"] = sum(in_footprint)
        self.astrometry_stats["n_local"] = len(distance_local)
        self.astrometry_stats["local_coord"] = local_coord
        self.astrometry_stats["local_tolerance"] = local_radius
        self.astrometry_stats["star_tolerance"] = star_tolerance
        self.astrometry_stats["offset_tolerance"] = offset_tolerance

        self.send_column_to_source_cat(colname="OFFSET_FROM_REF", sample=matches_source_cat)
        self.send_column_to_source_cat(colname="RA_OFFSET_FROM_REF", sample=matches_source_cat)
        self.send_column_to_source_cat(colname="DEC_OFFSET_FROM_REF", sample=matches_source_cat)
        self.send_column_to_source_cat(colname="PIX_OFFSET_FROM_REF", sample=matches_source_cat)
        self.send_column_to_source_cat(colname="X_OFFSET_FROM_REF", sample=matches_source_cat)
        self.send_column_to_source_cat(colname="Y_OFFSET_FROM_REF", sample=matches_source_cat)

        self.add_log(
            action=f"Calculated astrometry offset statistics.",
            method=self.astrometry_diagnostics,
            output_path=output_path
        )
        self.astrometry_err = self.astrometry_stats["rms_offset"]
        if not np.isnan(self.astrometry_err.value):
            self.headers[0]["ASTM_RMS"] = self.astrometry_err.value
        self.write_fits_file()
        self.update_output_file()

        return self.astrometry_stats

    def psf_diagnostics(
            self, star_class_tol: float = 0.95,
            mag_max: float = 0.0 * units.mag, mag_min: float = -7.0 * units.mag,
            match_to: table.Table = None, frame: float = 15):
        self.open()
        self.load_source_cat()
        u.debug_print(2, f"ImagingImage.psf_diagnostics(): {self}.source_cat_path ==", self.source_cat_path)
        stars_moffat, stars_gauss, stars_sex = ph.image_psf_diagnostics(
            hdu=self.hdu_list,
            cat=self.source_cat,
            star_class_tol=star_class_tol,
            mag_max=mag_max,
            mag_min=mag_min,
            match_to=match_to,
            frame=frame
        )

        fwhm_gauss = stars_gauss["GAUSSIAN_FWHM_FITTED"]
        self.fwhm_median_gauss = np.nanmedian(fwhm_gauss)
        self.fwhm_max_gauss = np.nanmax(fwhm_gauss)
        self.fwhm_min_gauss = np.nanmin(fwhm_gauss)
        self.fwhm_sigma_gauss = np.nanstd(fwhm_gauss)
        self.fwhm_rms_gauss = np.sqrt(np.mean(fwhm_gauss ** 2))
        self.send_column_to_source_cat("GAUSSIAN_FWHM_FITTED", stars_gauss)

        fwhm_moffat = stars_moffat["MOFFAT_FWHM_FITTED"]
        self.fwhm_median_moffat = np.nanmedian(fwhm_moffat)
        self.fwhm_max_moffat = np.nanmax(fwhm_moffat)
        self.fwhm_min_moffat = np.nanmin(fwhm_moffat)
        self.fwhm_sigma_moffat = np.nanstd(fwhm_moffat)
        self.fwhm_rms_moffat = np.sqrt(np.mean(fwhm_moffat ** 2))
        self.send_column_to_source_cat("MOFFAT_FWHM_FITTED", stars_moffat)

        fwhm_sextractor = stars_sex["FWHM_WORLD"].to(units.arcsec)
        self.fwhm_median_sextractor = np.nanmedian(fwhm_sextractor)
        self.fwhm_max_sextractor = np.nanmax(fwhm_sextractor)
        self.fwhm_min_sextractor = np.nanmin(fwhm_sextractor)
        self.fwhm_sigma_sextractor = np.nanstd(fwhm_sextractor)
        self.fwhm_rms_sextractor = np.sqrt(np.mean(fwhm_sextractor ** 2))

        self.close()

        results = {
            "n_stars": len(stars_gauss),
            "fwhm_psfex": self.fwhm_psfex.to(units.arcsec),
            "gauss": {
                "fwhm_median": self.fwhm_median_gauss.to(units.arcsec),
                "fwhm_mean": np.nanmean(fwhm_gauss).to(units.arcsec),
                "fwhm_max": self.fwhm_max_gauss.to(units.arcsec),
                "fwhm_min": self.fwhm_min_gauss.to(units.arcsec),
                "fwhm_sigma": self.fwhm_sigma_gauss.to(units.arcsec),
                "fwhm_rms": self.fwhm_rms_gauss.to(units.arcsec)
            },
            "moffat": {
                "fwhm_median": self.fwhm_median_moffat.to(units.arcsec),
                "fwhm_mean": np.nanmean(fwhm_moffat).to(units.arcsec),
                "fwhm_max": self.fwhm_max_moffat.to(units.arcsec),
                "fwhm_min": self.fwhm_min_moffat.to(units.arcsec),
                "fwhm_sigma": self.fwhm_sigma_moffat.to(units.arcsec),
                "fwhm_rms": self.fwhm_rms_moffat.to(units.arcsec)
            },
            "sextractor": {
                "fwhm_median": self.fwhm_median_sextractor.to(units.arcsec),
                "fwhm_mean": np.nanmean(fwhm_sextractor).to(units.arcsec),
                "fwhm_max": self.fwhm_max_sextractor.to(units.arcsec),
                "fwhm_min": self.fwhm_min_sextractor.to(units.arcsec),
                "fwhm_sigma": self.fwhm_sigma_sextractor.to(units.arcsec),
                "fwhm_rms": self.fwhm_rms_sextractor.to(units.arcsec)}
        }
        self.add_log(
            action=f"Calculated PSF FWHM statistics.",
            method=self.psf_diagnostics,
            packages=["source-extractor", "psfex"]
        )
        self.psf_stats = results
        self.update_output_file()
        return results

    def trim(
            self,
            left: Union[int, units.Quantity] = None,
            right: Union[int, units.Quantity] = None,
            bottom: Union[int, units.Quantity] = None,
            top: Union[int, units.Quantity] = None,
            output_path: str = None
    ):
        left = u.dequantify(left, unit=units.pix)
        right = u.dequantify(right, unit=units.pix)
        bottom = u.dequantify(bottom, unit=units.pix)
        top = u.dequantify(top, unit=units.pix)

        if output_path is None:
            output_path = self.path.replace(".fits", "_trimmed.fits")
        ff.trim_file(
            path=self.path,
            left=left, right=right, bottom=bottom, top=top,
            new_path=output_path
        )
        image = self.__class__(path=output_path, instrument_name=self.instrument_name)
        image.log = self.log.copy()

        image.add_log(
            action=f"Trimmed image to margins left={left}, right={right}, bottom={bottom}, top={top}",
            method=self.trim,
            output_path=output_path
        )
        image.update_output_file()

        return image

    def convert_to_cs(self, output_path: str, ext: int = 0):
        new = self.copy(output_path)
        gain = self.extract_gain()
        exp_time = self.extract_exposure_time()
        saturate = self.extract_saturate()

        new.load_data()
        new_data = new.data[ext]
        # new_data *= gain
        new_data /= exp_time
        new.data[ext] = new_data.value
        u.debug_print(1, "Image.concert_to_cs() 2: new_data.unit ==", new_data.unit)

        new.set_header_items(
            items={
                "BUNIT": str(new_data.unit),
                "GAIN": gain * exp_time,
                "OLD_GAIN": gain,
                "EXPTIME": 1.0,
                "OLD_EXPTIME": exp_time.value,
                "OLD_SATURATE": saturate,
                "SATURATE": saturate / exp_time
            },
            ext=ext,
            write=False
        )

        new.add_log(
            action=f"Converted image data on ext {ext} to cts / s, using exptime of {exp_time}.",
            method=self.convert_to_cs,
            output_path=output_path
        )

        new.write_fits_file()
        new.update_output_file()
        return new

    def clean_cosmic_rays(self, output_path: str, ext: int = 0):
        cleaned = self.copy(output_path)
        cleaned.load_data()
        data = cleaned.data[ext]
        gain = cleaned.extract_gain().value

        cleaned_data, mask = cosmicray_lacosmic(
            ccd=data,
            gain_apply=False,
            gain=gain,
            readnoise=cleaned.extract_noise_read().value,
            satlevel=cleaned.extract_saturate().value,
            verbose=True
        )

        cleaned.data[ext] = cleaned_data
        cleaned.write_fits_file()
        cleaned.add_log(
            action="Cleaned cosmic rays using LA cosmic algorithm.",
            method=self.clean_cosmic_rays,
            output_path=output_path,
            input_path=self.path,
        )
        cleaned.update_output_file()

    def reproject(
            self,
            other_image: 'ImagingImage',
            ext: int = 0,
            output_path: str = None,
            include_footprint: bool = False
    ):
        import reproject as rp
        if output_path is None:
            output_path = self.path.replace(".fits", "_reprojected.fits")
        other_image.load_headers()
        print(f"Reprojecting {self.filename} into the pixel space of {other_image.filename}")
        reprojected, footprint = rp.reproject_exact(self.path, other_image.headers[ext], parallel=True)

        if output_path == self.path:
            reprojected_image = self
        else:
            reprojected_image = self.copy(output_path)
        reprojected_image.load_data()
        reprojected_image.data[ext] = reprojected

        if include_footprint:
            new_hdu = fits.ImageHDU()
            reprojected_image.headers.append(new_hdu.header)
            reprojected_image.data.append(footprint)
        reprojected_image.write_fits_file()

        reprojected_image.add_log(
            action=f"Reprojected into pixel space of {other_image}.",
            method=self.reproject,
            output_path=output_path
        )
        reprojected_image.update_output_file()

        reprojected_image.transfer_wcs(other_image=other_image)
        return reprojected_image

    def trim_to_wcs(self, bottom_left: SkyCoord, top_right: SkyCoord, output_path: str = None) -> 'ImagingImage':
        """
        Trims the image to a footprint defined by two RA/DEC coordinates
        :param bottom_left:
        :param top_right:
        :param output_path:
        :return:
        """
        self.load_wcs()
        left, bottom = bottom_left.to_pixel(wcs=self.wcs, origin=0)
        right, top = top_right.to_pixel(wcs=self.wcs, origin=0)
        return self.trim(left=left, right=right, bottom=bottom, top=top, output_path=output_path)

    def match_to_cat(self, cat: Union[str, table.QTable],
                     ra_col: str = "ra", dec_col: str = "dec",
                     offset_tolerance: units.Quantity = 1 * units.arcsec,
                     star_tolerance: float = None,
                     dual: bool = False):

        source_cat = self.get_source_cat(dual=dual)

        ra_scale, dec_scale = self.extract_pixel_scale()

        if star_tolerance is not None:
            source_cat = source_cat[source_cat["CLASS_STAR"] > star_tolerance]

        matches_source_cat, matches_ext_cat, distance = a.match_catalogs(
            cat_1=source_cat,
            cat_2=cat,
            ra_col_1="RA",
            dec_col_1="DEC",
            ra_col_2=ra_col,
            dec_col_2=dec_col,
            tolerance=offset_tolerance)

        self.load_wcs()
        x_cat, y_cat = self.wcs.all_world2pix(matches_ext_cat[ra_col], matches_ext_cat[dec_col], 0)
        matches_ext_cat["x_image"] = x_cat
        matches_ext_cat["y_image"] = y_cat

        matches_source_cat["OFFSET_FROM_REF"] = distance.to(units.arcsec)
        matches_source_cat["RA_OFFSET_FROM_REF"] = matches_source_cat["RA"] - matches_ext_cat[ra_col]
        matches_source_cat["DEC_OFFSET_FROM_REF"] = matches_source_cat["DEC"] - matches_ext_cat[dec_col]

        matches_source_cat["PIX_OFFSET_FROM_REF"] = distance.to(units.pix, dec_scale)

        matches_source_cat["X_OFFSET_FROM_REF"] = matches_source_cat["X_IMAGE"] - x_cat * units.pix
        matches_source_cat["Y_OFFSET_FROM_REF"] = matches_source_cat["Y_IMAGE"] - y_cat * units.pix

        return matches_source_cat, matches_ext_cat, distance

    def signal_to_noise_ccd(self, dual: bool = False):
        self.extract_exposure_time()
        self.extract_gain()
        self.aperture_areas()
        source_cat = self.get_source_cat(dual=dual)

        flux_target = source_cat['FLUX_AUTO']
        rate_target = flux_target / self.exposure_time
        rate_sky = source_cat['BACKGROUND'] / (self.exposure_time * units.pix)
        rate_read = self.extract_noise_read()
        n_pix = source_cat['KRON_AREA_IMAGE'] / units.pixel

        source_cat["SNR_CCD"] = ph.signal_to_noise_ccd_equ(
            rate_target=rate_target,
            rate_sky=rate_sky,
            rate_read=rate_read,
            exp_time=self.exposure_time,
            gain=self.gain,
            n_pix=n_pix
        ).value

        self._set_source_cat(source_cat, dual)

        self.update_output_file()
        print("MEDIAN SNR:", np.median(source_cat["SNR_CCD"]))

        self.add_log(
            action=f"Estimated SNR using CCD Equation.",
            method=self.signal_to_noise_ccd,
        )
        self.update_output_file()

        return source_cat["SNR_CCD"]

    def signal_to_noise_measure(self, dual: bool = False):
        source_cat = self.get_source_cat(dual=dual)

        self.load_data()
        _, scale = self.extract_pixel_scale()
        mask = self.generate_mask(method='sep')
        mask = mask.astype(bool)
        bkg = self.calculate_background(method='sep', mask=mask)
        rms = bkg.rms()

        gain = self.extract_gain() / units.electron

        snrs = []
        snrs_se = []
        sigma_fluxes = []

        for cat_obj in source_cat:
            x = cat_obj["X_IMAGE"].value - 1
            y = cat_obj["Y_IMAGE"].value - 1

            a = cat_obj["A_WORLD"].to(units.pix, scale).value
            b = cat_obj["B_WORLD"].to(units.pix, scale).value

            theta = u.world_angle_se_to_pu(cat_obj["THETA_WORLD"])

            ap = photutils.aperture.EllipticalAperture(
                [x, y],
                a=a,
                b=b,
                theta=theta
            )

            ap_mask = ap.to_mask(method='center')

            flux = cat_obj["FLUX_AUTO"]

            ap_rms = ap_mask.multiply(rms)
            sigma_flux = np.sqrt(ap_rms.sum()) * units.ct
            snr = flux / np.sqrt(sigma_flux ** 2 + flux / gain)

            snr_se = flux / cat_obj["FLUXERR_AUTO"].value

            snrs.append(snr.value)
            sigma_fluxes.append(sigma_flux.value)
            snrs_se.append(snr_se.value)

        source_cat["SNR_MEASURED"] = snrs
        source_cat["NOISE_MEASURED"] = sigma_fluxes
        source_cat["SNR_SE"] = snrs_se

        self._set_source_cat(source_cat=source_cat, dual=dual)

        self.add_log(
            action=f"Estimated SNR using SEP RMS map and Source Extractor uncertainty.",
            method=self.signal_to_noise_measure,
            packages=["source-extractor"]
        )
        self.update_output_file()

    def object_axes(self):
        self.load_source_cat()
        self.extract_pixel_scale()
        self.source_cat["A_IMAGE"] = self.source_cat["A_WORLD"].to(units.pix, self.pixel_scale_dec)
        self.source_cat["B_IMAGE"] = self.source_cat["B_WORLD"].to(units.pix, self.pixel_scale_dec)
        self.source_cat_dual["A_IMAGE"] = self.source_cat_dual["A_WORLD"].to(units.pix, self.pixel_scale_dec)
        self.source_cat_dual["B_IMAGE"] = self.source_cat_dual["B_WORLD"].to(units.pix, self.pixel_scale_dec)

        self.add_log(
            action=f"Created axis columns A_IMAGE, B_IMAGE in pixel units from A_WORLD, B_WORLD.",
            method=self.object_axes,
        )
        self.update_output_file()

    def estimate_sky_background(self, ext: int = 0, force: bool = False):
        """
        Estimates background as a global median. VERY loose estimate.
        :param ext:
        :param force:
        :return:
        """
        if force or self.sky_background is None:
            self.load_data()
            self.sky_background = np.nanmedian(self.data[ext]) * units.ct / units.pixel
        else:
            print("Sky background already estimated.")
        return self.sky_background

    def plot_apertures(self):
        self.load_source_cat()
        pl.plot_all_params(image=self.path, cat=self.source_cat, kron=True, show=False)
        plt.title(self.filter_name)
        plt.show()

    def find_object(self, coord: SkyCoord, dual: bool = True):
        cat = self.get_source_cat(dual=dual)
        u.debug_print(2, f"{self}.find_object(): dual ==", dual)
        u.debug_print(2, f"{self}.find_object(): cat.colnames ==", cat.colnames)
        coord_cat = SkyCoord(cat["RA"], cat["DEC"])
        separation = coord.separation(coord_cat)
        i = np.argmin(separation)
        nearest = cat[i]
        return nearest, separation[i]

    def find_object_index(self, index: int, dual: bool = True):
        """
        Using NUMBER column
        :param index:
        :param dual:
        :return:
        """
        source_cat = self.get_source_cat(dual=dual)
        i, _ = u.find_nearest(source_cat["NUMBER"], index)
        return source_cat[i], i

    def generate_psf_image(self, x: int, y: int, output: str = None):
        """
        Generates an image of the modelled point-spread function of the image.
        :param x:
        :param y:
        :param output:
        :return:
        """
        pass

    def plot_subimage(
            self,
            centre: SkyCoord,
            frame: units.Quantity,
            ext: int = 0,
            fig: plt.Figure = None,
            n: int = 1, n_x: int = 1, n_y: int = 1,
            show_cbar: bool = False,
            show_grid: bool = False,
            ticks: int = None,
            show_coords: bool = True, ylabel: str = None,
            reverse_y=False,
            imshow_kwargs: dict = None,  # Can include cmap
            normalize_kwargs: dict = None,  # Can include vmin, vmax
            output_path: str = None,
            **kwargs,
    ):
        self.load_data()
        self.load_wcs()
        _, scale = self.extract_pixel_scale()
        x, y = self.wcs.all_world2pix(centre.ra.value, centre.dec.value, 0)
        data = self.data[ext]
        u.debug_print(1, "ImagingImage.plot_subimage(): frame ==", frame)
        left, right, bottom, top = u.frame_from_centre(frame.to(units.pix, scale).value, x, y, data)

        if fig is None:
            fig = plt.figure()

        if normalize_kwargs is None:
            normalize_kwargs = {}
        if imshow_kwargs is None:
            imshow_kwargs = {}

        if "stretch" not in normalize_kwargs:
            normalize_kwargs["stretch"] = SqrtStretch()
        elif normalize_kwargs["stretch"] == "sqrt":
            normalize_kwargs["stretch"] = SqrtStretch()
        elif normalize_kwargs["stretch"] == "log":
            normalize_kwargs["stretch"] = LogStretch()

        if "interval" not in normalize_kwargs:
            normalize_kwargs["interval"] = MinMaxInterval()

        if "origin" not in imshow_kwargs:
            imshow_kwargs["origin"] = "lower"

        plot = fig.add_subplot(n_x, n_y, n, projection=self.wcs)
        plot.imshow(
            data,
            norm=ImageNormalize(
                data[bottom:top, left:right],
                **normalize_kwargs
            ),
            **imshow_kwargs
        )
        plot.set_xlim(left, right)
        plot.set_ylim(bottom, top)
        plot.set_xlabel("Right Ascension (J2000)")
        plot.set_ylabel("Declination (J2000)")

        if output_path is not None:
            fig.savefig(output_path)

        other_args = {}
        other_args["x"] = x
        other_args["y"] = y

        return plot, fig, other_args

    def nice_frame(
            self,
            row: table.Row,
            frame: units.Quantity = 10 * units.pix,
    ):
        self.extract_pixel_scale()
        u.debug_print(1, "ImagingImage.nice_frame(): row['KRON_RADIUS'], row['A_WORLD'] ==", row['KRON_RADIUS'],
                      row['A_WORLD'].to(units.arcsec))
        kron_a = row['KRON_RADIUS'] * row['A_WORLD']
        u.debug_print(1, "ImagingImage.nice_frame(): kron_a ==", kron_a)
        pix_scale = self.pixel_scale_dec
        u.debug_print(1, "ImagingImage.nice_frame(): self.pixel_scale_dec ==", self.pixel_scale_dec)
        this_frame = max(
            kron_a.to(units.pixel, pix_scale), frame)  # + 5 * units.pix,
        u.debug_print(1, "ImagingImage.nice_frame(): this_frame ==", this_frame)
        return this_frame

    def plot_source_extractor_object(
            self,
            row: table.Row,
            ext: int = 0,
            frame: units.Quantity = 10 * units.pix,
            output: str = None,
            show: bool = False, title: str = None):

        self.load_headers()
        kron_a = row['KRON_RADIUS'] * row['A_WORLD']
        kron_b = row['KRON_RADIUS'] * row['B_WORLD']
        kron_theta = row['THETA_WORLD']
        # kron_theta = -kron_theta + ff.get_rotation_angle(
        #     header=self.headers[ext],
        #     astropy_units=True)
        this_frame = self.nice_frame(row=row, frame=frame)
        mid_x = row["X_IMAGE"]
        mid_y = row["Y_IMAGE"]
        left = mid_x - this_frame
        right = mid_x + this_frame
        bottom = mid_y - this_frame
        top = mid_y + this_frame
        self.open()
        image_cut = ff.trim(hdu=self.hdu_list, left=left, right=right, bottom=bottom, top=top)
        norm = pl.nice_norm(image=image_cut[ext].data)
        title = u.latex_sanitise(title)
        plt.imshow(image_cut[0].data, origin='lower', norm=norm)
        # theta =
        pl.plot_gal_params(
            hdu=image_cut,
            ras=[row["RA"].value],
            decs=[row["DEC"].value],
            a=[row["A_WORLD"].value],
            b=[row["B_WORLD"].value],
            theta=[row["THETA_WORLD"].value],
            world=True,
            show_centre=True
        )
        pl.plot_gal_params(
            hdu=image_cut,
            ras=[row["RA"].value],
            decs=[row["DEC"].value],
            a=[kron_a.value],
            b=[kron_b.value],
            theta=[row["THETA_WORLD"].value],
            world=True,
            show_centre=True
        )
        if title is None:
            title = self.name
        plt.title(title)
        plt.savefig(os.path.join(output))
        if show:
            plt.show()
        self.close()
        return

    def plot(self, fig: plt.Figure = None, ext: int = 0, **kwargs):
        if fig is None:
            fig = plt.figure(figsize=(12, 12), dpi=1000)
        ax, fig = self.wcs_axes(fig=fig)
        self.load_data()
        data = self.data[ext]
        ax.imshow(
            u.dequantify(data), **kwargs,
            norm=ImageNormalize(
                u.dequantify(data),
                interval=MinMaxInterval(),
                stretch=SqrtStretch(),
                vmin=np.median(data),
            ),
            origin='lower',
        )
        return ax, fig

    def wcs_axes(self, fig: plt.Figure = None):
        if fig is None:
            fig = plt.figure(figsize=(12, 12), dpi=1000)
        ax = fig.add_subplot(
            projection=self.load_wcs()
        )
        return ax, fig

    def plot_catalogue(
            self,
            cat: table.QTable,
            ra_col: str = "ra",
            dec_col: str = "dec",
            colour_column: str = None,
            fig: plt.Figure = None,
            ext: int = 0,
            cbar_label: str = None,
            **kwargs
    ):
        if fig is None:
            fig = plt.figure()
        if colour_column is not None:
            c = u.dequantify(cat[colour_column])
        else:
            c = "red"

        ax, fig = self.plot(fig=fig, ext=ext, zorder=0, **kwargs)
        x, y = self.wcs.all_world2pix(cat[ra_col], cat[dec_col], 0)
        pcm = plt.scatter(x, y, c=c, cmap="plasma", marker="x", zorder=10)
        if colour_column is not None:
            fig.colorbar(pcm, ax=ax, label=cbar_label)

        u.debug_print(2, f"{self}.plot_catalogue(): len(cat):", len(cat))

        return ax, fig

    def insert_synthetic_sources(
            self,
            x: np.float64, y: np.float64,
            mag: np.float64,
            output: str,
            overwrite: bool = True,
            world_coordinates: bool = False,
            extra_values: table.Table = None,
            model: str = "psfex"
    ):
        if self.psfex_path is None:
            raise ValueError(f"{self.name}.psfex_path has not been set.")
        if self.zeropoint_best is None:
            raise ValueError(f"{self.name}.zeropoint_best has not been set.")
        output_cat = output.replace('.fits', '_synth_cat.ecsv')

        self.extract_pixel_scale()

        if model == "gaussian":
            file, sources = ph.insert_point_sources_to_file(
                file=self.path,
                x=x, y=y, mag=mag,
                zeropoint=self.zeropoint_best["zeropoint"],
                airmass=self.extract_airmass(),
                extinction=self.zeropoint_best["extinction"],
                exp_time=self.extract_exposure_time(),
                world_coordinates=world_coordinates,
                extra_values=extra_values,
                output=output,
                output_cat=output_cat,
                overwrite=overwrite,
                fwhm=self.fwhm_psfex.to(units.pix, self.pixel_scale_dec)
            )
        elif model == "psfex":
            file, sources = ph.insert_point_sources_to_file(
                file=self.path,
                x=x, y=y, mag=mag,
                psf_model=self.psfex_path,
                zeropoint=self.zeropoint_best["zeropoint"],
                airmass=self.extract_airmass(),
                extinction=self.zeropoint_best["extinction"],
                exp_time=self.extract_exposure_time(),
                world_coordinates=world_coordinates,
                extra_values=extra_values,
                output=output,
                output_cat=output_cat,
                overwrite=overwrite
            )
        else:
            raise ValueError(f"Model {model} not recognised.")

        inserted = self.new_image(output)
        u.debug_print(1, "ImagingImage.insert_synthetic_sources: output_cat", output_cat)
        inserted.synth_cat_path = output_cat
        inserted.add_log(
            action=f"Injected synthetic point-sources, defined in output catalogue at {output_cat}.",
            method=self.insert_synthetic_sources,
            output_path=output
        )
        inserted.update_output_file()
        return inserted, sources

    def insert_synthetic_range(
            self,
            x: float = None, y: float = None,
            mag_min: units.Quantity = 20.0 * units.mag,
            mag_max: units.Quantity = 30.0 * units.mag,
            interval: units.Quantity = 0.1 * units.mag,
            output_dir: str = None,
            filename: str = None,
            model: str = "psfex",
            positioning: str = "inplace",
            scale: units.Quantity = 10 * units.arcsec
    ):
        x = float(x)
        y = float(y)

        x_ref, y_ref = self.extract_ref_pixel()
        if x is None:
            x = x_ref
        if y is None:
            y = y_ref

        inserted = []
        cats = []
        if filename is None:
            filename = f"{self.name}_insert"
        for mag in np.linspace(mag_min, mag_max, int((mag_max - mag_min) / interval + 1)):
            u.debug_print(1, f"INSERTING SOURCE {mag}")

            if positioning == 'inplace':
                x_synth = x
                y_synth = y
            elif positioning == 'gaussian':
                self.extract_pixel_scale()
                scale.to(units.pix, self.pixel_scale_dec)
                x_synth = -1
                y_synth = -1
                self.extract_n_pix()
                x_max, y_max = self.n_x, self.n_y
                while not (0 < x_synth < x_max) or not (0 < y_synth < y_max):
                    x_synth, y_synth = gaussian_distributed_point(x_0=x, y_0=y, sigma=scale.value)
            else:
                raise ValueError(f"positioning {positioning} not recognised.")

            file, sources = self.insert_synthetic_sources(
                x=x_synth, y=y_synth, mag=mag.value,
                output=os.path.join(output_dir, filename + f"_mag_{np.round(mag.value, 1)}.fits"),
                model=model
            )
            file.source_extraction_psf(output_dir=output_dir)
            file.zeropoint_best = self.zeropoint_best
            file.calibrate_magnitudes()
            file.signal_to_noise_ccd()
            inserted.append(file)
            cat = file.check_synthetic_sources()
            cats.append(cat)

        cat_all = table.vstack(cats)
        cat_all["distance_from_ref"] = np.sqrt(
            (cat_all["x_inserted"] - x) ** 2 + (cat_all["y_inserted"] - y) ** 2) * units.pix
        return cat_all

    def check_synthetic_sources(self):
        """
        Checks on the fidelity of inserted sources against catalogue.
        :return:
        """

        self.signal_to_noise_measure()
        self.signal_to_noise_ccd()

        self.load_synth_cat()
        if self.synth_cat is None:
            raise ValueError("No synth_cat present.")

        matches_source_cat, matches_synth_cat, distance = self.match_to_cat(
            cat=self.synth_cat,
            ra_col='ra_inserted',
            dec_col='dec_inserted',
            offset_tolerance=1.0 * units.arcsec,
            # star_tolerance=0.7,
        )

        aperture_radius = 2 * self.fwhm_pix_psfex

        self.synth_cat["flux_sep"], self.synth_cat["flux_sep_err"], _ = self.sep_aperture_photometry(
            aperture_radius=aperture_radius,
            x=self.synth_cat["x_inserted"],
            y=self.synth_cat["y_inserted"]
        )

        self.synth_cat["mag_sep"], self.synth_cat["mag_sep_err"], _, _ = self.magnitude(
            flux=self.synth_cat["flux_sep"],
            flux_err=self.synth_cat["flux_sep_err"]
        )

        self.synth_cat["delta_mag_sep"] = self.synth_cat["mag_sep"] - self.synth_cat["mag_inserted"]
        self.synth_cat["fraction_flux_recovered_sep"] = self.synth_cat["flux_sep"] / self.synth_cat["flux_inserted"]
        self.synth_cat["snr_sep"] = self.synth_cat["flux_sep"] / self.synth_cat["flux_sep_err"]
        self.synth_cat["aperture_radius"] = aperture_radius

        matches_source_cat["matching_dist"] = distance.to(units.arcsec)
        matches_source_cat["fraction_flux_recovered_auto"] = matches_source_cat["FLUX_AUTO"] / matches_synth_cat[
            "flux_inserted"]
        matches_source_cat["fraction_flux_recovered_psf"] = matches_source_cat["FLUX_PSF"] / matches_synth_cat[
            "flux_inserted"]
        matches_source_cat["delta_mag_auto"] = matches_source_cat["MAG_AUTO_ZP_best"] - matches_synth_cat[
            "mag_inserted"]
        matches_source_cat["delta_mag_psf"] = matches_source_cat["MAG_PSF_ZP_best"] - matches_synth_cat["mag_inserted"]

        if len(matches_source_cat) > 0:
            self.synth_cat = table.hstack([self.synth_cat, matches_source_cat])

        self.add_log(
            action=f"Created catalogue of synthetic sources and their measurements.",
            method=self.check_synthetic_sources,
            output_path=self.synth_cat_path
        )
        self.update_output_file()

        return self.synth_cat

    def test_limit_location(
            self,
            coord: SkyCoord,
            ap_radius: units.Quantity = 2 * units.arcsec,
            ext: int = 0,
            sigma_min: int = 1,
            sigma_max: int = 10,
            **kwargs
    ):

        self.load_wcs()
        _, pix_scale = self.extract_pixel_scale()
        x, y = self.wcs.all_world2pix(coord.ra, coord.dec, 0)
        ap_radius_pix = ap_radius.to(units.pix, pix_scale).value

        mask = self.generate_mask(method='sep')
        mask = mask.astype(bool)

        self.calculate_background(method="sep", mask=mask, ext=ext, **kwargs)
        rms = self.sep_background[ext].rms()

        # plt.imshow(rms)
        # plt.colorbar()
        # plt.show()

        flux, _, _ = sep.sum_circle(rms, [x], [y], ap_radius_pix)
        sigma_flux = np.sqrt(flux)

        limits = {}
        for i in range(sigma_min, sigma_max + 1):
            n_sigma_flux = sigma_flux * i
            limit, _, _, _ = self.magnitude(flux=n_sigma_flux)
            limits[f"{i}-sigma"] = {
                "flux": n_sigma_flux,
                "mag": limit
            }
        return limits

    def test_limit_synthetic(
            self,
            coord: SkyCoord = None,
            output_dir: str = None,
            positioning: str = "inplace",
            mag_min: units.Quantity = 20.0 * units.mag,
            mag_max: units.Quantity = 30.0 * units.mag,
            interval: units.Quantity = 0.1 * units.mag,
    ):

        if output_dir is None:
            output_dir = os.path.join(self.data_path, f"{self.name}_lim_test")
        u.mkdir_check(output_dir)
        if SkyCoord is None:
            coord = self.extract_pointing()
        self.load_wcs()
        x, y = self.wcs.all_world2pix(coord.ra, coord.dec, 0)
        sources = self.insert_synthetic_range(
            x=x, y=y,
            mag_min=mag_min,
            mag_max=mag_max,
            interval=interval,
            output_dir=output_dir,
            positioning=positioning
        )

        plt.scatter(sources["mag_inserted"], sources["fraction_flux_recovered_psf"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Fraction of flux recovered")
        plt.savefig(os.path.join(output_dir, "flux_recovered_psf.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["fraction_flux_recovered_auto"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Fraction of flux recovered")
        plt.savefig(os.path.join(output_dir, "flux_recovered_auto.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["fraction_flux_recovered_sep"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Fraction of flux recovered")
        plt.savefig(os.path.join(output_dir, "flux_recovered_sep.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["delta_mag_psf"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Mag psf - mag inserted")
        plt.savefig(os.path.join(output_dir, "delta_mag_psf.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["delta_mag_auto"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Mag auto - mag inserted")
        plt.savefig(os.path.join(output_dir, "delta_mag_auto.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["delta_mag_sep"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Mag auto - mag inserted")
        plt.savefig(os.path.join(output_dir, "delta_mag_sep.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["CLASS_STAR"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Class star")
        plt.savefig(os.path.join(output_dir, "class_star.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["matching_dist"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Matching distance (arcsec)")
        plt.savefig(os.path.join(output_dir, "matching_dist.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["snr_sep"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("S/N, measured by SEP")
        plt.savefig(os.path.join(output_dir, "matching_dist.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["SNR_SE"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("S/N, measured by SEP")
        plt.savefig(os.path.join(output_dir, "matching_dist.png"))
        plt.close()

        # TODO: S/N measure and plot

        ax, fig = self.plot_catalogue(cat=sources, ra_col="ra_inserted", dec_col="dec_inserted")
        fig.savefig(os.path.join(output_dir, "inserted_overplot.png"))

        sources.write(os.path.join(output_dir, "synth_cat_all.ecsv"), format="ascii.ecsv")

        self.add_log(
            action=f"Created catalogue of synthetic sources from range of insertions and their measurements.",
            method=self.test_limit_synthetic,
            output_path=output_dir
        )
        self.update_output_file()

        return sources

    def calculate_background(
            self, ext: int = 0,
            box_size: int = 64,
            filter_size: int = 3,
            method: str = "sep",
            **back_kwargs
    ):
        self.load_data()

        if method == "sep":
            data = u.sanitise_endianness(self.data[ext])
            bkg = self.sep_background[ext] = sep.Background(
                data,
                bw=box_size, bh=box_size,
                fw=filter_size, fh=filter_size,
                **back_kwargs
            )
            if isinstance(data, units.Quantity):
                back = bkg.back() * data.unit
            else:
                back = bkg.back()
            self.data_sub_bkg[ext] = (data - back)

        elif method == "photutils":
            data = self.data[ext]
            sigma_clip = SigmaClip(sigma=3.)
            bkg_estimator = photutils.MedianBackground()
            bkg = self.pu_background[ext] = photutils.Background2D(
                data, box_size,
                filter_size=filter_size,
                sigma_clip=sigma_clip,
                bkg_estimator=bkg_estimator,
                **back_kwargs
            )
            self.data_sub_bkg[ext] = (data - bkg.background)

        else:
            raise ValueError(f"Unrecognised method {method}.")
        return bkg

    def generate_segmap(
            self,
            ext: int = 0,
            threshold: float = 4,
            method="sep",
            margins: tuple = (None, None, None, None),
            min_area: int = 5,
            **background_kwargs
    ):
        """
        Generate a segmentation map of the image in which the image is broken into segments according to detected sources.
        Each source is assigned an integer, and the segmap has the same spatial dimensions as the input image.
        :param ext:
        :param threshold:
        :param method:
        :param margins:
        :return:
        """
        self.load_data()
        data = self.data[ext]
        left, right, bottom, top = u.check_margins(data=data, margins=margins)

        bkg = self.calculate_background(method=method, ext=ext, **background_kwargs)

        if method == "photutils":
            data_trim = u.trim_image(
                data=data,
                margins=margins
            )
            u.debug_print(2, f"{self}.generate_segmap(): data_trim.shape ==", data_trim.shape)
            threshold = photutils.segmentation.detect_threshold(
                data_trim,
                threshold,
                background=u.trim_image(bkg.background, margins=margins),
                error=u.trim_image(bkg.background_rms, margins=margins)
            )
            u.debug_print(2, f"{self}.generate_segmap(): threshold ==", threshold)
            segmap = photutils.detect_sources(data_trim, threshold, npixels=min_area)

        elif method == "sep":
            # The copying is done here to avoid 'C-contiguous' errors in SEP.
            data_trim = u.sanitise_endianness(
                u.trim_image(self.data_sub_bkg[ext], margins=margins)
            ).copy()
            err = u.trim_image(bkg.rms(), margins=margins).copy()
            u.debug_print(2, f"{self}.generate_segmap(): type(err) ==", type(err), "err.shape ==", err.shape)
            objects, segmap = sep.extract(
                data_trim,
                err=err,
                thresh=threshold,
                deblend_cont=True, clean=False,
                segmentation_map=True, minarea=min_area
            )

        else:
            raise ValueError(f"Unrecognised method {method}.")
        segmap_full = np.zeros(data.shape)
        u.debug_print(2, f"{self}.generate_segmap(): segmap_full ==", segmap_full)
        u.debug_print(2, f"{self}.generate_segmap(): segmap ==", segmap)
        segmap_full[bottom:top + 1, left:right + 1] = segmap.data
        return segmap_full

    def generate_mask(
            self,
            unmasked: SkyCoord = (),
            ext: int = 0,
            threshold: float = 4,
            method: str = "sep",
            obj_value=1,
            back_value=0,
            margins: tuple = (None, None, None, None)
    ):
        """
        Uses a segmentation map to produce a
        :param unmasked: SkyCoord list of objects to keep unmasked; if any
        :param ext:
        :param threshold:
        :param method:
        :param obj_value: For GALFIT masks, should be 1.
        :param back_value: For GALFIT masks, should be 0.
        :param margins: If only part of the image is to be masked, provide (left, right, bottom, top) in pixel
            coordinates as tuple.
        :return:
        """
        data = self.load_data()[ext]
        segmap = self.generate_segmap(
            ext=ext, threshold=threshold, method=method,
            margins=margins)
        self.load_wcs()

        unmasked = u.check_iterable(unmasked)

        if segmap is None:
            mask = np.zeros(data.shape)
        else:
            # Loop over the given coordinates and eliminate those segments from the mask.
            mask = np.ones(data.shape, dtype=bool)
            # This sets all the background pixels to False
            mask[segmap == 0] = False
            for coord in unmasked:
                x_unmasked, y_unmasked = self.wcs.all_world2pix(coord.ra, coord.dec, 0)
                # obj_id is the integer representing that object in the segmap
                obj_id = segmap[int(np.round(y_unmasked)), int(np.round(x_unmasked))]
                # If obj_id is zero, then our work here is already done (ie, the segmap routine read it as background anyway)
                if obj_id != 0:
                    mask[segmap == obj_id] = False

        # Convert to integer (from bool)
        mask = mask.astype(int)
        mask[mask > 0] = obj_value
        mask[mask == 0] = back_value

        return mask

    def masked_data(
            self,
            mask: np.ndarray = None,
            ext: int = 0,
            mask_type: str = 'zeroed-out',
            **generate_mask_kwargs
    ):
        self.load_data()
        if mask is None:
            mask = self.generate_mask(**generate_mask_kwargs)

        # if mask_type == 'zeroed-out'

        return np.ma.MaskedArray(self.data[ext].copy(), mask=mask)

    def write_mask(
            self, output_path: str,
            ext: int = 0,
            **mask_kwargs
    ):
        """
        Not yet implemented.
        :param path:
        :return:
        """

        mask_file = self.copy(output_path)
        mask_file.load_data()
        mask_file.data[ext] = self.generate_mask(ext=ext, **mask_kwargs)
        mask_file.write_fits_file()

        mask_file.add_log(
            action="Converted image to source mask.",
            method=self.source_extraction,
            output_path=output_path,
        )
        mask_file.update_output_file()

    def sep_aperture_photometry(
            self, x: float, y: float,
            aperture_radius: units.Quantity = 2.0 * units.arcsec,
            ext: int = 0
    ):
        self.calculate_background(ext=ext)
        self.extract_pixel_scale()
        pixel_radius = aperture_radius.to(units.pix, self.pixel_scale_dec)
        flux, fluxerr, flag = sep.sum_circle(
            self.data_sub_bkg,
            x, y,
            pixel_radius.value,
            err=self.sep_background.rms(),
            gain=self.extract_gain().value)
        return flux, fluxerr, flag

    @classmethod
    def select_child_class(cls, instrument: str, **kwargs):
        if instrument is None:
            return ImagingImage
        instrument = instrument.lower()
        if instrument == "panstarrs1":
            return PanSTARRS1Cutout
        elif instrument == "vlt-fors2":
            return FORS2Image
        elif instrument == "vlt-hawki":
            return HAWKICoaddedImage
        elif instrument == "gs-aoi":
            return GSAOIImage
        elif "hst" in instrument:
            return HubbleImage
        else:
            raise ValueError(f"Unrecognised instrument {instrument}")

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "filter": "FILTER",
            "ra": "CRVAL1",
            "dec": "CRVAL2",
            "ref_pix_x": "CRPIX1",
            "ref_pix_y": "CRPIX2",
            "ra_old": "_RVAL1",
            "dec_old": "_RVAL2",
            "airmass": "AIRMASS",
            "astrometry_err": "ASTM_RMS"
        })
        return header_keys

    @classmethod
    def count_exposures(cls, image_paths: list):
        return len(image_paths)

    @classmethod
    def rank_photometric_cat(cls):
        """
        Gives the ranking of photometric catalogues available for calibration, ranked by similarity to filter set.
        :return:
        """

        return [
            "calib_pipeline",
            "instrument_archive",
            "des",
            "delve",
            "panstarrs1",
            "sdss",
            "skymapper"
        ]


class CoaddedImage(ImagingImage):
    def __init__(self, path: str, frame_type: str = None, instrument_name: str = None, area_file: str = None):
        super().__init__(path=path, frame_type=frame_type, instrument_name=instrument_name)
        self.area_file = area_file  # string
        if self.area_file is None:
            self.area_file = self.path.replace(".fits", "_area.fits")

    def trim(
            self,
            left: Union[int, units.Quantity] = None,
            right: Union[int, units.Quantity] = None,
            bottom: Union[int, units.Quantity] = None,
            top: Union[int, units.Quantity] = None,
            output_path: str = None
    ):
        # Let the super method take care of this image
        trimmed = super().trim(left=left, right=right, bottom=bottom, top=top, output_path=output_path)
        # Trim the area file in the same way.
        new_area_path = output_path.replace(".fits", "_area.fits")
        ff.trim_file(
            path=self.area_file,
            left=left, right=right, bottom=bottom, top=top,
            new_path=new_area_path
        )
        trimmed.area_file = new_area_path
        trimmed.add_log(
            action=f"Trimmed area file to margins left={left}, right={right}, bottom={bottom}, top={top}",
            method=self.trim,
            output_path=new_area_path
        )
        trimmed.update_output_file()
        return trimmed

    def trim_from_area(self, output_path: str = None):
        left, right, bottom, top = ff.detect_edges_area(self.area_file)
        trimmed = self.trim(left=left, right=right, bottom=bottom, top=top, output_path=output_path)
        return trimmed

    def _output_dict(self):
        outputs = super()._output_dict()
        outputs.update({
            "area_file": self.area_file
        })
        return outputs

    def load_output_file(self):
        outputs = super().load_output_file()
        if outputs is not None:
            if "area_file" in outputs:
                self.area_file = outputs["area_file"]
        return outputs

    def correct_astrometry(self, output_dir: str = None, tweak: bool = True, **kwargs):
        new_image = super().correct_astrometry(
            output_dir=output_dir,
            tweak=tweak,
            **kwargs
        )
        new_image.area_file = self.area_file
        return new_image

    def copy(self, destination: str):
        new_image = super().copy(destination)
        new_image.area_file = self.area_file
        new_image.update_output_file()
        return new_image

    @classmethod
    def select_child_class(cls, instrument: str, **kwargs):
        if not isinstance(instrument, str):
            instrument = str(instrument)
        if instrument is None:
            return CoaddedImage
        elif instrument == "vlt-fors2":
            return FORS2CoaddedImage
        else:
            return CoaddedImage
            # raise ValueError(f"Unrecognised instrument {instrument}")


class PanSTARRS1Cutout(ImagingImage):
    instrument_name = "panstarrs1"

    def __init__(self, path: str):
        super().__init__(path=path)
        self.extract_filter()
        self.instrument_name = "panstarrs1"
        self.exposure_time = None
        self.extract_exposure_time()

    def extract_filter(self):
        key = self.header_keys()["filter"]
        fil_string = self.extract_header_item(key)
        self.filter_name = fil_string[:fil_string.find(".")]
        self.filter_short = self.filter_name
        return self.filter_name

    def extract_exposure_time(self):
        self.load_headers()
        exp_time_keys = filter(lambda k: k.startswith("EXP_"), self.headers[0])
        exp_time = 0.
        # exp_times = []
        for key in exp_time_keys:
            exp_time += self.headers[0][key]
        #    exp_times.append(self.headers[0][key])

        self.exposure_time = exp_time * units.second  # np.mean(exp_times)
        return self.exposure_time

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({"noise_read": "HIERARCH CELL.READNOISE",
                            "filter": "HIERARCH FPA.FILTERID",
                            "gain": "HIERARCH CELL.GAIN"})
        return header_keys


class ESOImagingImage(ImagingImage, ESOImage):
    def extract_frame_type(self):
        obj = self.extract_object()
        category = self.extract_header_item("ESO DPR CATG")
        if category is None:
            category = self.extract_header_item("ESO PRO CATG")
        if obj == "BIAS":
            self.frame_type = "bias"
        elif "FLAT" in obj:
            self.frame_type = "flat"
        elif obj == "STD":
            self.frame_type = "standard"
        elif category == "SCIENCE":
            self.frame_type = "science"
        elif category == "SCIENCE_REDUCED_IMG":
            self.frame_type = "science_reduced"
        return self.frame_type

    def extract_airmass(self):
        key = self.header_keys()["airmass"]
        self.airmass = self.extract_header_item(key)
        u.debug_print(1, f"{self.name}.airmass", self.airmass)
        if self.airmass is None:
            airmass_start = self.extract_header_item("ESO TEL AIRM START")
            airmass_end = self.extract_header_item("ESO TEL AIRM END")
            self.airmass = (airmass_start + airmass_end) / 2
        u.debug_print(1, f"{self.name}.airmass", self.airmass)
        return self.airmass

    @classmethod
    def header_keys(cls) -> dict:
        header_keys = super().header_keys()
        header_keys.update(ESOImage.header_keys())
        header_keys.update({"noise_read": "HIERARCH ESO DET OUT1 RON",
                            "filter": "HIERARCH ESO INS FILT1 NAME",
                            "gain": "HIERARCH ESO DET OUT1 GAIN",
                            })
        return header_keys

    @classmethod
    def count_exposures(cls, image_paths: list):
        # Counts only chip 1 images
        n = 0
        for path in image_paths:
            image = cls(path=path)
            chip = image.extract_chip_number()
            if chip == 1:
                n += 1
        return n


class HAWKICoaddedImage(ESOImagingImage):
    instrument_name = "vlt-hawki"

    def zeropoint(
            self
    ):
        self.zeropoint_best = {
            "zeropoint": self.extract_header_item("PHOTZP"),
            "zeropoint_err": self.extract_header_item("PHOTZPER"),
            "extinction": 0.0 * units.mag,
            "extinction_err": 0.0 * units.mag,
            "airmass": 0.0,
            "airmass_err": 0.0,
            "catalogue": "2MASS"
        }

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "gain": "GAIN"
        })
        return header_keys


class FORS2Image(ESOImagingImage):
    instrument_name = "vlt-fors2"

    def __init__(self, path: str, frame_type: str = None, **kwargs):
        super().__init__(path=path, frame_type=frame_type, instrument_name=self.instrument_name)
        self.other_chip = None
        self.chip_number = None

    def extract_chip_number(self):
        chip_string = self.extract_header_item(key='HIERARCH ESO DET CHIP1 ID')
        chip = 0
        if chip_string == 'CCID20-14-5-3':
            chip = 1
        elif chip_string == 'CCID20-14-5-6':
            chip = 2
        self.chip_number = chip
        return chip

    def _output_dict(self):
        outputs = super()._output_dict()
        if self.other_chip is not None:
            other_chip = self.other_chip.path
        else:
            other_chip = None
        outputs.update({
            "other_chip": other_chip,
        })
        return outputs

    def load_output_file(self):
        outputs = super().load_output_file()
        if outputs is not None:
            if "other_chip" in outputs:
                self.other_chip = outputs["other_chip"]
        return outputs

    @classmethod
    def rank_photometric_cat(cls):
        """
        Gives the ranking of photometric catalogues available for calibration, ranked by similarity to filter set.
        :return:
        """

        return [
            "instrument_archive",
            "des",
            "panstarrs1",
            "sdss",
            "skymapper"]


class FORS2CoaddedImage(CoaddedImage):
    instrument_name = "vlt-fors2"

    def __init__(
            self,
            path: str,
            frame_type: str = None,
            area_file: str = None,
            **kwargs
    ):
        super().__init__(
            path=path,
            frame_type=frame_type,
            instrument_name=self.instrument_name,
            area_file=area_file
        )

    def zeropoint(
            self,
            cat_path: str,
            output_path: str,
            cat_name: str = 'Catalogue',
            cat_zeropoint: units.Quantity = 0.0 * units.mag,
            cat_zeropoint_err: units.Quantity = 0.0 * units.mag,
            image_name: str = None,
            show: bool = False,
            sex_x_col: str = 'XPSF_IMAGE',
            sex_y_col: str = 'YPSF_IMAGE',
            sex_ra_col: str = 'RA',
            sex_dec_col: str = 'DEC',
            sex_flux_col: str = 'FLUX_PSF',
            stars_only: bool = True,
            star_class_col: str = 'CLASS_STAR',
            star_class_tol: float = 0.95,
            mag_range_sex_lower: units.Quantity = -100. * units.mag,
            mag_range_sex_upper: units.Quantity = 100. * units.mag,
            dist_tol: units.Quantity = 2. * units.arcsec,
            snr_cut=100
    ):
        super().zeropoint(
            cat_path=cat_path,
            output_path=output_path,
            cat_name=cat_name,
            cat_zeropoint=cat_zeropoint,
            cat_zeropoint_err=cat_zeropoint_err,
            image_name=image_name,
            show=show,
            sex_x_col=sex_x_col,
            sex_y_col=sex_y_col,
            sex_ra_col=sex_ra_col,
            sex_dec_col=sex_dec_col,
            sex_flux_col=sex_flux_col,
            stars_only=stars_only,
            star_class_col=star_class_col,
            star_class_tol=star_class_tol,
            mag_range_sex_lower=mag_range_sex_lower,
            mag_range_sex_upper=mag_range_sex_upper,
            dist_tol=dist_tol,
            snr_cut=snr_cut
        )
        self.calibration_from_qc1()

    def calibration_from_qc1(self):
        """
        Use the FORS2 QC1 archive to retrieve calibration parameters.
        :return:
        """
        self.extract_filter()
        u.debug_print(
            1, f"FORS2CoaddedImage.calibration_from_qc1(): {self}.instrument ==", self.instrument,
            self.instrument_name)
        fil = self.instrument.filters[self.filter_name]
        fil.retrieve_calibration_table()
        if fil.calibration_table is not None:
            self.extract_date_obs()
            row = fil.get_nearest_calib_row(mjd=self.mjd_obs)

            if self.epoch is not None and self.epoch.airmass_err is not None:
                airmass_err = self.epoch.airmass_err[self.filter_name]
            else:
                airmass_err = 0.0

            zp_dict = {
                "zeropoint": row["zeropoint"],
                "zeropoint_err": row["zeropoint_err"],
                "airmass": self.extract_airmass(),
                "airmass_err": airmass_err,
                "extinction": row["extinction"],
                "extinction_err": row["extinction_err"],
                "mjd_measured": row["mjd_obs"],
                "delta_t": row["mjd_obs"] - self.mjd_obs,
                "n_matches": "n/a",
                "catalogue": "fors2_qc1_archive"
            }

            self.zeropoints["instrument_archive"] = zp_dict
            self.zeropoint_best = zp_dict

            self.extinction_atmospheric = row["extinction"]
            self.extinction_atmospheric_err = row["extinction_err"]

            self.add_log(
                action=f"Retrieved calibration values from ESO QC1 archive.",
                method=self.calibration_from_qc1,
            )
            self.update_output_file()

            return self.zeropoints["instrument_archive"]
        else:
            return None


class GSAOIImage(ImagingImage):
    instrument_name = "gs-aoi"

    def extract_pointing(self):
        # GSAOI images keep the WCS information in the second HDU header.
        key = self.header_keys()["ra"]
        ra = self.extract_header_item(key, 1)
        key = self.header_keys()["dec"]
        dec = self.extract_header_item(key, 1)
        self.pointing = SkyCoord(ra, dec, unit=units.deg)
        return self.pointing


class HubbleImage(ImagingImage):
    instrument_name = "hst-dummy"

    def extract_exposure_time(self):
        self.exposure_time = 1.0 * units.second
        return self.exposure_time

    def zeropoint(self, **kwargs):
        """
        Returns the AB magnitude zeropoint of the image, according to
        https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
        :return:
        """
        photflam = self.extract_header_item("PHOTFLAM")
        photplam = self.extract_header_item("PHOTPLAM")

        zeropoint = (-2.5 * math.log10(photflam) - 5 * math.log10(photplam) - 2.408) * units.mag

        self.zeropoint_best = {
            "zeropoint": zeropoint,
            "zeropoint_err": 0.0 * units.mag,
            "airmass": 0.0,
        }
        self.update_output_file()
        self.add_log(
            action=f"Calculated zeropoint {zeropoint} from PHOTFLAM and PHOTPLAM header keys.",
            method=self.trim,
        )
        self.update_output_file()
        return self.zeropoint_best

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({"gain": "CCDGAIN"})
        return header_keys


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
    def select_child_class(cls, instrument: str, **kwargs):
        if 'frame_type' in kwargs:
            frame_type = kwargs['frame_type']
            if frame_type == "coadded":
                return Spec1DCoadded
            elif frame_type == "raw":
                return SpecRaw
        else:
            raise KeyError("frame_type is required.")


class SpecRaw(Spectrum):
    frame_type = "raw"

    def __init__(self, path: str = None, frame_type: str = None, decker: str = None, binning: str = None):
        super().__init__(path=path, frame_type=frame_type, decker=decker, binning=binning)
        self.pypeit_line = None

    @classmethod
    def from_pypeit_line(cls, line: str, pypeit_raw_path: str):
        attributes = line.split('|')
        attributes = list(map(lambda at: at.replace(" ", ""), attributes))
        inst = SpecRaw(path=os.path.join(pypeit_raw_path, attributes[1]),
                       frame_type=attributes[2],
                       decker=attributes[7],
                       binning=attributes[8])
        inst.pypeit_line = line
        return inst


class Spec1DCoadded(Spectrum):
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
        hdu_list = deepcopy(self.hdu_list)
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


def deepest(
        img_1: ImagingImage, img_2: ImagingImage, sigma: int = 5, depth_type: str = "secure",
        snr_type: str = "SNR_MEASURED"):
    if img_1.depth[depth_type][snr_type][f"{sigma}-sigma"] > \
            img_2.depth[depth_type][snr_type][f"{sigma}-sigma"]:
        return img_1
    else:
        return img_2

# def pypeit_str(self):
#     header = self.hdu[0].header
#     string = f"| {self.filename} | "
