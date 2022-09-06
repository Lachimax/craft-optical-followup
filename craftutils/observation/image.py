# Code by Lachlan Marnoch, 2021
import math
import string
import os
import shutil
from typing import Union, Tuple, List
from copy import deepcopy

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import astropy.io.fits as fits
import astropy.table as table
import astropy.wcs as wcs
import astropy.units as units
from astropy.stats import SigmaClip

from astropy.visualization import (
    ImageNormalize, LogStretch, SqrtStretch, MinMaxInterval)
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import quantity_support

import photutils

try:
    import sep
except ModuleNotFoundError:
    print("sep not installed; some photometry-related functionality will be unavailable.")

import craftutils.utils as u
import craftutils.astrometry as astm
import craftutils.fits_files as ff
import craftutils.photometry as ph
import craftutils.params as p
import craftutils.plotting as pl
import craftutils.observation.log as log
import craftutils.observation.objects as objects
from craftutils.stats import gaussian_distributed_point
import craftutils.observation.instrument as inst
import craftutils.wrap.source_extractor as se
import craftutils.wrap.psfex as psfex
import craftutils.wrap.galfit as galfit
from craftutils.wrap.astrometry_net import solve_field
from craftutils.retrieve import cat_columns, cat_instruments

quantity_support()

# This contains the names as in the header as keys and the names as used in this project as values.
instrument_header = {
    "FORS2": "vlt-fors2",
    "HAWKI": "vlt-hawki"
}

active_images = {}


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
        instrument = detect_instrument(file_path)
        cls = ImagingImage.select_child_class(instrument)
        image = cls.from_fits(path=file_path)
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
        instrument = detect_instrument(file_path, fail_quietly=True)
        if instrument is None:
            print(f"Instrument could not be detected for {file_path}.")
            continue
        cls = ImagingImage.select_child_class(instrument=instrument)
        image = cls.from_fits(path=file_path)
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


def detect_instrument(path: str, ext: int = 0, fail_quietly: bool = False):
    try:
        with fits.open(path) as file:
            if "INSTRUME" in file[ext].header:
                inst_str = file[ext].header["INSTRUME"]
                if "WFC3" in inst_str:
                    det_str = file[ext].header["DETECTOR"]
                    if "UVIS" in det_str:
                        return "hst-wfc3_uvis2"
                    elif "IR" in det_str:
                        return "hst-wfc3_ir"
                if "FORS2" in inst_str:
                    return "vlt-fors2"
                elif "HAWKI" in inst_str:
                    return "vlt-hawki"
            elif "FPA.TELESCOPE" in file[ext].header:
                inst_str = file[ext].header["FPA.TELESCOPE"]
                if "PS1" in inst_str:
                    return "panstarrs1"
            else:
                if not fail_quietly:
                    raise ValueError(f"Could not establish instrument from file header on {path}.")
                else:
                    return None
    except OSError:
        if fail_quietly:
            print(f"The file {path} is missing the SIMPLE card and may be corrupt.")
            return None
        else:
            raise OSError(f"The file {path} is missing the SIMPLE card and may be corrupt.")


def from_path(path: str, cls: type = None, **kwargs):
    """
    To be used when there may already be an image instance for this path floating around in memory, and it's okay
    (or better) to access this one instead of creating a new instance.
    When the image may have overwritten a previous file, instantiating the image directly is better.
    :param path:
    :param cls:
    :param kwargs:
    :return:
    """
    u.debug_print(3, "image.from_path(): path ==", path)
    if path in active_images:
        return active_images[path]
    if cls is not None:
        return cls(path, **kwargs)


class Image:
    instrument_name = "dummy"
    num_chips = 1

    def __init__(
            self,
            path: str,
            frame_type: str = None,
            instrument_name: str = None,
            logg: log.Log = None,
    ):

        self.path = path
        active_images[path] = self
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"The image file file {path} does not exist.")
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
        self.chip_number = None
        self.airmass = None
        self.airmass_err = 0.0

        if logg is None:
            self.log = log.Log()
        u.debug_print(2, f"Image.__init__(): {self}.log.log.keys() ==", self.log.log.keys())

    def __eq__(self, other):
        if not isinstance(other, Image):
            raise TypeError("Can only compare Image instance to another Image instance.")
        return self.path == other.path

    def __str__(self):
        return self.filename

    def copy_headers(
            self,
            other: 'Image'
    ):
        other.load_headers()
        self.headers = other.headers
        self.write_fits_file()

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
        u.mkdir_check_nested(destination)
        shutil.copy(self.path, destination)
        new_image = self.new_image(path=destination)
        new_image.load_headers(force=True)
        new_image.load_data(force=True)
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
            unit = self.extract_units()
            self.open()
            u.debug_print(1, f"Image.load_data() 1: {self}.hdu_list:", self.hdu_list)

            self.data = []
            for i, h in enumerate(self.hdu_list):
                if unit[i] is not None:
                    try:
                        this_unit = units.Unit(unit[i])
                    except ValueError:
                        this_unit = units.ct
                else:
                    this_unit = units.ct
                if h.data is not None:
                    try:
                        self.data.append(h.data * this_unit)
                    except TypeError or ValueError:
                        # If unit could not be parsed, assume counts
                        self.data.append(h.data * units.ct)
                else:
                    self.data.append(None)

            self.close()
        else:
            u.debug_print(1, "Data already loaded.")
        return self.data

    def to_ccddata(self, unit: Union[str, units.Unit]):
        import ccdproc
        if unit is None:
            return ccdproc.CCDData.read(self.path)
        else:
            return ccdproc.CCDData.read(self.path, unit=unit)

    # @classmethod
    # def from_ccddata(self, ccddata: 'ccdproc.CCDData', path: str):
    #     pass

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
        self.headers[ext]["HISTORY"] = str(Time.now().strftime("%Y-%m-%dT%H:%M:%S")) + ": " + note
        # self.write_fits_file()

    def _extract_header_item(self, key: str, ext: int = 0):
        self.load_headers()
        if key in self.headers[ext]:
            return self.headers[ext][key]
        else:
            return None

    def extract_header_item(self, key: str, ext: int = 0, accept_absent: bool = False):
        # Check in the given HDU, then check all headers.
        value = self._extract_header_item(key=key, ext=ext)
        u.debug_print(2, "")
        u.debug_print(2, "Image.extract_header_item():")
        u.debug_print(2, f"\t{self}.path ==", self.path)
        u.debug_print(2, f"\t key ==", key)
        u.debug_print(2, f"\t value ==", value)
        u.debug_print(2, "")
        if value is None and not accept_absent:
            for ext in range(len(self.headers)):
                value = self._extract_header_item(key=key, ext=ext)
                if value is not None:
                    return value
            # Then, if we get to the end of the loop, the item clearly doesn't exist.
            return None
        else:
            return value

    def extract_chip_number(self):
        chip = 1
        self.chip_number = chip
        return chip

    def extract_unit(self, astropy: bool = False):
        key = self.header_keys()["unit"]
        unit = self.extract_header_item(key)
        if astropy:
            if unit is not None:
                unit = units.Unit(unit)
            else:
                unit = units.ct
        return unit

    def extract_units(self):
        key = self.header_keys()["unit"]
        un = []
        for i, header in enumerate(self.headers):
            un.append(self.extract_header_item(key, ext=i, accept_absent=True))
        return un

    def extract_program_id(self):
        key = self.header_keys()["program_id"]
        return str(self.extract_header_item(key))

    def extract_gain(self):
        self.gain = self.extract_header_item("GAIN")
        if self.gain is None:
            key = self.header_keys()["gain"]
            u.debug_print(2, f"Image.extract_gain(): type({self})", type(self), key)
            self.gain = self.extract_header_item(key) * units.electron / units.ct
        if self.gain is not None:
            self.gain = u.check_quantity(self.gain, units.electron / units.ct)
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

    def remove_extra_extensions(self, ext: int = 0):
        self.load_headers()
        self.load_data()
        self.headers = [self.headers[ext]]
        self.data = [self.data[ext]]
        self.write_fits_file()

    def write_fits_file(self):
        self.open()
        for i in range(len(self.headers)):
            if i >= len(self.hdu_list):
                self.hdu_list.append(fits.ImageHDU())
            if self.data is not None:
                unit = self.data[i].unit
                self.hdu_list[i].data = u.dequantify(self.data[i])
                self.set_header_item(
                    key=self.header_keys()["unit"],
                    value=str(unit),
                    ext=i
                )
            if self.headers is not None:
                self.hdu_list[i].header = self.headers[i]

        while len(self.hdu_list) > len(self.headers):
            self.hdu_list.pop(-1)

        self.hdu_list.writeto(self.path, overwrite=True)

        self.close()

    @classmethod
    def header_keys(cls):
        header_keys = {
            "integration_time": "INTTIME",
            "exposure_time": "EXPTIME",
            "exposure_time_old": "OLD_EXPTIME",
            "noise_read": "RON",
            "noise_read_old": "OLD_RON",
            "gain": "GAIN",
            "gain_old": "OLD_GAIN",
            "date-obs": "DATE-OBS",
            "mjd-obs": "MJD-OBS",
            "object": "OBJECT",
            "instrument": "INSTRUME",
            "unit": "BUNIT",
            "saturate": "SATURATE",
            "saturate_old": "OLD_SATURATE",
            "program_id": "PROG_ID"
        }
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

    def split_fits(self, output_dir: str = None):
        if output_dir is None:
            output_dir = self.data_path
        self.open()
        new_files = {}
        for hdu in self.hdu_list:
            new_hdu_list = fits.HDUList(fits.PrimaryHDU(hdu.data, hdu.header))
            new_path = os.path.join(output_dir, self.filename.replace(".fits", f"_{hdu.name}.fits"))
            new_hdu_list.writeto(
                new_path,
                overwrite=True
            )
            new_img = self.__class__(new_path)
            new_files[hdu.name] = new_img
        self.close()
        return new_files


class ESOImage(Image):
    """
    Generic parent class for ESO images, both spectra and imaging
    """

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "mode": "HIERARCH ESO INS MODE",
        })
        return header_keys


class ImagingImage(Image):
    def __init__(
            self,
            path: str,
            frame_type: str = None,
            instrument_name: str = None,
            load_outputs: bool = True
    ):
        super().__init__(path=path, frame_type=frame_type, instrument_name=instrument_name)

        self.wcs = None

        self.filter_name = None
        self.filter_short = None
        self.filter = None
        self.pixel_scale_x = None
        self.pixel_scale_y = None

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

        self.sky_background = None

        self.zeropoints = {}
        self.zeropoint_output_paths = {}
        self.zeropoint_best = None

        self.extinction_atmospheric = None
        self.extinction_atmospheric_err = None

        self.depth = {}

        self.astrometry_err = None
        self.ra_err = None
        self.dec_err = None
        self.astrometry_corrected_path = None
        self.astrometry_stats = {}

        self.extract_filter()

        if load_outputs:
            self.load_output_file()

    def source_extraction(
            self,
            configuration_file: str,
            output_dir: str,
            parameters_file: str = None,
            catalog_name: str = None,
            template: 'ImagingImage' = None,
            **configs
    ) -> str:
        if template is not None:
            template = template.path
            self.dual_mode_template = template
        self.extract_gain()
        u.debug_print(2, f"ImagingImage.source_extraction(): template ==", template)
        if not self.do_subtract_background():
            configs["BACK_TYPE"] = "MANUAL"
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

    def psfex(
            self,
            output_dir: str,
            force: bool = False,
            set_attributes: bool = True,
            se_kwargs: dict = {},
            **kwargs
    ):
        """
        Run PSFEx on this image to obtain a PSF model.
        :param output_dir: path to directory to write PSFEx outputs to.
        :param force: If False, and this object already has a PSF model, we just return the one that already exists.
        :param se_kwargs: arguments to pass to Source Extractor.
        :param kwargs: arguments to pass to PSFEx.
        :param set_attributes: If True, this Image's psfex_path, psfex_output, fwhm_pix_psfex and fwhm_psfex will be set
            according to the PSFEx output.
        :return: HDUList representing the PSF model FITS file.
        """
        psfex_output = None

        if force or self.psfex_path is None or not os.path.isfile(self.psfex_path):
            # Set up a list of photometric apertures to pass to SE as a string.
            _, scale = self.extract_pixel_scale()
            aper_arcsec = [
                              4.87,
                              3.9,
                              2.92
                          ] * units.arcsec
            phot_aper = aper_arcsec.to(units.pix, scale).value
            phot_aper_str = ""
            for a in phot_aper:
                phot_aper_str += f"{a},"
            phot_aper_str = phot_aper_str[:-1]
            se_kwargs["PHOT_APERTURES"] = phot_aper_str
            kwargs["PHOTFLUX_KEY"] = '"FLUX_APER(1)"'
            kwargs["PHOTFLUXERR_KEY"] = '"FLUXERR_APER(1)"'

            config = p.path_to_config_sextractor_config_pre_psfex()
            output_params = p.path_to_config_sextractor_param_pre_psfex()
            catalog = self.source_extraction(
                configuration_file=config,
                output_dir=output_dir,
                parameters_file=output_params,
                catalog_name=f"{self.name}_psfex.fits",
                **se_kwargs
            )

            psfex_path = psfex.psfex(
                catalog=catalog,
                output_dir=output_dir,
                **kwargs
            )
            psfex_output = fits.open(psfex_path)

            if not psfex.check_successful(psfex_output):
                print(f"PSFEx did not converge. Retrying with PHOTFLUX_KEY==FLUX_AUTO")

                kwargs["PHOTFLUX_KEY"] = "FLUX_AUTO"
                kwargs["PHOTFLUXERR_KEY"] = "FLUXERR_AUTO"

                psfex_path = psfex.psfex(
                    catalog=catalog,
                    output_dir=output_dir,
                    **kwargs
                )
                psfex_output = fits.open(psfex_path)

            i = 1
            while not psfex.check_successful(psfex_output) and i < len(aper_arcsec):
                print(f"PSFEx did not converge. Retrying with smaller PHOTFLUX apertures.")
                kwargs["PHOTFLUX_KEY"] = f'"FLUX_APER({i + 1})"'
                kwargs["PHOTFLUXERR_KEY"] = f'"FLUXERR_APER({i + 1})"'

                catalog = self.source_extraction(
                    configuration_file=config,
                    output_dir=output_dir,
                    parameters_file=output_params,
                    catalog_name=f"{self.name}_psfex.fits",
                    **se_kwargs
                )

                psfex_path = psfex.psfex(
                    catalog=catalog,
                    output_dir=output_dir,
                    **kwargs
                )

                i += 1

            if set_attributes:
                self.psfex_path = psfex_path
                self.extract_pixel_scale()
                pix_scale = self.pixel_scale_y
                self.fwhm_pix_psfex = psfex_output[1].header['PSF_FWHM'] * units.pixel
                self.fwhm_psfex = self.fwhm_pix_psfex.to(units.arcsec, pix_scale)

            self.add_log(
                action="PSF modelled using psfex.",
                method=self.psfex,
                output_path=output_dir,
                packages=["psfex"]
            )
            self.update_output_file()

        if set_attributes:
            return self.load_psfex_output()
        else:
            return psfex_output

    # def _psfex(self):

    def load_psfex_output(self, force: bool = False):
        if force or self.psfex_output is None:
            self.psfex_output = fits.open(self.psfex_path)
        return self.psfex_output

    def psf_image(self, x: float, y: float, match_pixel_scale: bool = True):
        if match_pixel_scale:
            return psfex.load_psfex(model_path=self.psfex_path, x=x, y=y)
        else:
            return psfex.load_psfex_oversampled(model=self.psfex_path, x=x, y=y)

    def source_extraction_psf(
            self,
            output_dir: str,
            template: 'ImagingImage' = None,
            force: bool = False,
            **configs
    ):
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

        psf = self.psfex(
            output_dir=output_dir,
            force=force,
        )

        if psfex.check_successful(psf):
            cat_path = self.source_extraction(
                configuration_file=p.path_to_config_sextractor_config(),
                output_dir=output_dir,
                parameters_file=p.path_to_config_sextractor_param(),
                catalog_name=f"{self.name}_psf-fit.cat",
                psf_name=self.psfex_path,
                seeing_fwhm=self.fwhm_psfex.value,
                template=template,
                **configs
            )
        else:
            cat_path = self.source_extraction(
                configuration_file=p.path_to_config_sextractor_failed_psfex_config(),
                parameters_file=p.path_to_config_sextractor_failed_psfex_param(),
                output_dir=output_dir,
                catalog_name=f"{self.name}_failed-psf-fit.cat",
                template=template,
                **configs
            )
        dual = False
        if template is not None:
            dual = True
        if dual:
            self.source_cat_sextractor_dual_path = cat_path
            cat = self.load_source_cat_sextractor_dual(force=True)
        else:
            self.source_cat_sextractor_path = cat_path
            cat = self.load_source_cat_sextractor(force=True)

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
            if dual:
                self.source_cat_sextractor_dual_path = cat_path
                cat = self.load_source_cat_sextractor_dual(force=True)
            else:
                self.source_cat_sextractor_path = cat_path
                cat = self.load_source_cat_sextractor(force=True)
        else:
            self.psfex_successful = True

        u.debug_print(2, "dual, template:", dual, template)

        self.write_source_cat()

        self.plot_apertures()
        self.add_log(
            action="Sources extracted using Source Extractor with PSFEx PSF modelling.",
            method=self.source_extraction_psf,
            output_path=output_dir,
            packages=["psfex", "source-extractor"]
        )
        self.signal_to_noise_measure(dual=dual)
        print()
        self.update_output_file()

    def _load_source_cat_sextractor(self, path: str):
        self.load_wcs()
        print("Loading source catalogue from", path)
        source_cat = table.QTable.read(path, format="ascii.sextractor")
        if "SPREAD_MODEL" in source_cat.colnames:
            source_cat = u.classify_spread_model(source_cat)
        source_cat["RA"], source_cat["DEC"] = self.wcs.all_pix2world(
            source_cat["X_IMAGE"],
            source_cat["Y_IMAGE"],
            1
        ) * units.deg
        self.extract_astrometry_err()
        if self.ra_err is not None:
            source_cat["RA_ERR"] = np.sqrt(
                source_cat["ERRX2_WORLD"].to(units.arcsec ** 2) + self.ra_err ** 2)
        else:
            source_cat["RA_ERR"] = np.sqrt(
                source_cat["ERRX2_WORLD"].to(units.arcsec ** 2))
        if self.dec_err is not None:
            source_cat["DEC_ERR"] = np.sqrt(
                source_cat["ERRY2_WORLD"].to(units.arcsec ** 2) + self.dec_err ** 2)
        else:
            source_cat["DEC_ERR"] = np.sqrt(
                source_cat["ERRY2_WORLD"].to(units.arcsec ** 2))

        return source_cat

    def world_to_pixel(self, coord: SkyCoord, origin: int = 0) -> np.ndarray:
        """
        Turns a sky coordinate into image pixel coordinates;
        :param coord: SkyCoord object to convert to pixel coordinates; essentially a wrapper for SkyCoord.to_pixel()
        :param origin: Do you want pixel indexing that starts at 1 (FITS convention) or 0 (numpy convention)?
        :return: xp, yp: numpy.ndarray, the pixel coordinates.
        """
        self.load_wcs()
        return coord.to_pixel(self.wcs, origin=origin)

    def pixel_to_world(
            self,
            x: Union[float, np.ndarray, units.Quantity],
            y: Union[float, np.ndarray, units.Quantity],
            origin: int = 0
    ) -> SkyCoord:
        """
        Uses the image's wcs to turn pixel coordinates into sky; essentially a wrapper for SkyCoord.from_pixel().
        :param x: Pixel x-coordinate. Can be provided as an astropy Quantity with units pix, or as a raw number.
        :param y: Pixel y-coordinate. Can be provided as an astropy Quantity with units pix, or as a raw number.
        :param origin: Do you want pixel indexing that starts at 1 (FITS convention) or 0 (numpy convention)?
        :return coord: SkyCoord reflecting the sky coordinates.
        """
        self.load_wcs()
        x = u.dequantify(x, unit=units.pix)
        y = u.dequantify(y, unit=units.pix)
        return SkyCoord.from_pixel(x, y, wcs=self.wcs, origin=origin)

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
            print("source_cat could not be loaded from SE file because source_cat_sextractor_path has not been set.")

        return self.source_cat

    def load_source_cat_sextractor_dual(self, force: bool = False):
        if self.source_cat_sextractor_dual_path is not None:
            if force:
                self.source_cat_dual = None
            if self.source_cat_dual is None:
                self.source_cat_dual = self._load_source_cat_sextractor(path=self.source_cat_sextractor_dual_path)
        else:
            print(
                "source_cat_dual could not be loaded from SE file because source_cat_sextractor_dual_path has not been set.")

        return self.source_cat_dual

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

    def push_source_cat(self, dual: bool = True):
        source_cat = self.get_source_cat(dual=dual)
        for i, row in enumerate(source_cat):
            print(f"Pushing row {i} of {len(source_cat)}")
            obj = objects.Object(row=row, field=self.epoch.field)
            if "SNR_PSF" in self.depth["secure"]:
                depth = self.depth["secure"]["SNR_PSF"][f"5-sigma"]
            else:
                depth = self.depth["secure"]["SNR_AUTO"][f"5-sigma"]
            obj.add_photometry(
                instrument=self.instrument_name,
                fil=self.filter_name,
                epoch_name=self.epoch.name,
                mag=row['MAG_AUTO_ZP_best'],
                mag_err=row[f'MAGERR_AUTO_ZP_best'],
                snr=row[f'SNR_AUTO'],
                ellipse_a=row['A_WORLD'],
                ellipse_a_err=row["ERRA_WORLD"],
                ellipse_b=row['B_WORLD'],
                ellipse_b_err=row["ERRB_WORLD"],
                ellipse_theta=row['THETA_J2000'],
                ellipse_theta_err=row['ERRTHETA_J2000'],
                ra=row['RA'],
                ra_err=np.sqrt(row["ERRX2_WORLD"]),
                dec=row['DEC'],
                dec_err=np.sqrt(row["ERRY2_WORLD"]),
                kron_radius=row["KRON_RADIUS"],
                separation_from_given=None,
                epoch_date=str(self.epoch.date.isot),
                class_star=row["CLASS_STAR"],
                spread_model=row["SPREAD_MODEL"],
                spread_model_err=row["SPREADERR_MODEL"],
                class_flag=row["CLASS_FLAG"],
                mag_psf=row["MAG_PSF_ZP_best"],
                mag_psf_err=row["MAGERR_PSF_ZP_best"],
                snr_psf=row["FLUX_PSF"] / row["FLUXERR_PSF"],
                image_depth=depth,
                image_path=self.path,
                do_mask=self.mask_nearby(),
                zeropoint=row["ZP_best_ATM_CORR"]
            )
            obj.push_to_table(select=False)

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
        key = self.header_keys()["ra_err"]
        self.ra_err = self.extract_header_item(key)
        key = self.header_keys()["dec_err"]
        self.dec_err = self.extract_header_item(key)
        if self.astrometry_err is not None:
            self.astrometry_err *= units.arcsec
        if self.ra_err is not None:
            self.ra_err *= units.arcsec
        if self.dec_err is not None:
            self.dec_err *= units.arcsec
        return self.astrometry_err

    def extract_rotation_angle(self, ext: int = 0):
        self.load_headers()
        return ff.get_rotation_angle(header=self.headers[ext], astropy_units=True)

    def extract_wcs_footprint(self):
        """
        Returns the RA & Dec of the corners of the image.
        :return: tuple of SkyCoords, (top_left, top_right, bottom_left, bottom_right)
        """
        self.load_wcs()
        return self.wcs.calc_footprint()

    def _pixel_scale(self, ext: int = 0):
        self.load_wcs(ext=ext)
        return wcs.utils.proj_plane_pixel_scales(
            self.wcs
        ) * units.deg

    def extract_pixel_scale(self, ext: int = 0, force: bool = False):
        if force or self.pixel_scale_x is None or self.pixel_scale_y is None:
            x, y = self._pixel_scale(ext=ext)
            self.pixel_scale_x = units.pixel_scale(x / units.pix)
            self.pixel_scale_y = units.pixel_scale(y / units.pix)
        else:
            u.debug_print(2, "Pixel scale already set.")

        return self.pixel_scale_x, self.pixel_scale_y

    def extract_world_scale(self, ext: int = 0, force: bool = False):
        x, y = self._pixel_scale(ext=ext)
        dec = self.extract_pointing().dec.to(units.rad)
        ra_scale = units.pixel_scale((x / np.cos(dec)) / units.pix)
        dec_scale = units.pixel_scale(y / units.pix)
        return ra_scale, dec_scale

    def extract_filter(self):
        key = self.header_keys()["filter"]
        self.filter_name = self.extract_header_item(key)
        if self.filter_name is not None:
            self.filter_short = self.filter_name[0]

        self._filter_from_name()

        return self.filter_name

    def _filter_from_name(self):
        if self.filter_name is not None and self.instrument is not None and self.filter_name in self.instrument.filters:
            self.filter = self.instrument.filters[self.filter_name]

    def extract_airmass(self):
        key = self.header_keys()["airmass"]
        self.airmass = self.extract_header_item(key)
        key = self.header_keys()["airmass_err"]
        self.airmass_err = self.extract_header_item(key)
        if self.airmass_err is None:
            self.airmass_err = 0.0
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
            "psf_stats": self.psf_stats,
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

        if not self.zeropoints:
            return None, None

        ranking, diff = self.rank_photometric_cat(cats=self.zeropoints)
        if preferred is not None:
            ranking.insert(0, preferred)

        zps = []
        for i, cat in enumerate(ranking):
            if cat in self.zeropoints:
                zps_cat = []
                for img_name in self.zeropoints[cat]:
                    zp = self.zeropoints[cat][img_name]
                    zps_cat.append(zp)
                zps_cat.sort(key=lambda z: z["zeropoint_img_err"])
                zps.extend(zps_cat)

        zp_tbl = table.QTable(zps)
        if len(zp_tbl) > 0:
            # zp_tbl.sort("zeropoint_img_err")
            zp_tbl.write(
                os.path.join(self.data_path, f"{self.name}_zeropoints.ecsv"), format="ascii.ecsv",
                overwrite=True
            )
            best_row = zp_tbl[0]
            best_cat = best_row["catalogue"]
            best_img = best_row["image_name"]

            if best_cat is None:
                raise ValueError("No zeropoints are present to select from.")

            zeropoint_best = self.zeropoints[best_cat][best_img]
            print(
                f"For {self.name}, we have selected a zeropoint of {zeropoint_best['zeropoint_img']} "
                f"+/- {zeropoint_best['zeropoint_img_err']}, "
                f"from {zeropoint_best['catalogue']} on {zeropoint_best['image_name']}.")
            if not no_user_input:
                select_own = u.select_yn(message="Would you like to select another?", default=False)
                if select_own:
                    zps = {}
                    for i, row in enumerate(zp_tbl):
                        pick_str = f"{row['catalogue']} {row['zeropoint_img']} +/- {row['zeropoint_img_err']}, " \
                                   f"{row['n_matches']} stars, " \
                                   f"from {row['image_name']}"
                        zps[pick_str] = self.zeropoints[row['catalogue']][row['image_name']]
                    _, zeropoint_best = u.select_option(message="Select best zeropoint:", options=zps)
                    best_cat = zeropoint_best["catalogue"]
            self.zeropoint_best = zeropoint_best

            self.set_header_items(
                items={
                    "ZP": zeropoint_best["zeropoint_img"],
                    "ZP_ERR": zeropoint_best["zeropoint_img_err"],
                    "ZPCAT": str(zeropoint_best["catalogue"]),
                },
                ext=0,
                write=False
            )

            self.add_log(
                action=f"Selected best zeropoint as {zeropoint_best['zeropoint']} +/- {zeropoint_best['zeropoint_err']}, from {zeropoint_best['catalogue']}",
                method=self.select_zeropoint
            )
        else:
            best_cat = None

        self.update_output_file()
        self.write_fits_file()
        return self.zeropoint_best, best_cat

    def zeropoint(
            self,
            cat_path: str,
            output_path: str,
            cat_name: str,
            cat_zeropoint: units.Quantity = 0.0 * units.mag,
            cat_zeropoint_err: units.Quantity = 0.0 * units.mag,
            image_name: str = None,
            show: bool = False,
            phot_type: str = "PSF",
            sex_ra_col: str = "RA",
            sex_dec_col: str = "DEC",
            stars_only: bool = True,
            star_class_tol: float = 0.95,
            mag_range_sex_lower: units.Quantity = -100. * units.mag,
            mag_range_sex_upper: units.Quantity = 100. * units.mag,
            dist_tol: units.Quantity = None,
            snr_cut=3.,
            iterate_uncertainty: bool = True,
            do_x_shift: bool = True
    ):
        print(f"\nEstimating photometric zeropoint for {self.name}, {type(self)}\n")

        sex_flux_col = f"FLUX_{phot_type}"

        if phot_type == "PSF":
            sex_x_col = "XPSF_IMAGE"
            sex_y_col = "YPSF_IMAGE"
        else:
            sex_x_col = "X_IMAGE"
            sex_y_col = "Y_IMAGE"

        self.signal_to_noise_measure()
        if image_name is None:
            image_name = self.name
        self.extract_filter()
        column_names = cat_columns(cat=cat_name, f=self.filter_short)
        cat_ra_col = column_names['ra']
        cat_dec_col = column_names['dec']
        cat_mag_col = column_names['mag_psf']
        cat_mag_col_err = column_names['mag_psf_err']
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
            cat_mag_col_err=cat_mag_col_err,
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
            exp_time=self.extract_exposure_time(),
            cat_type=cat_type,
            cat_zeropoint=cat_zeropoint,
            cat_zeropoint_err=cat_zeropoint_err,
            snr_col=f'SNR_{phot_type}',
            snr_cut=snr_cut,
            iterate_uncertainty=iterate_uncertainty,
            do_x_shift=do_x_shift
        )

        if zp_dict is None:
            return None

        zp_dict = self.add_zeropoint(
            # catalogue=cat_name,
            # zeropoint=zp_dict["zeropoint"],
            # zeropoint_err=zp_dict["zeropoint_err"],
            extinction=0.0 * units.mag,
            extinction_err=0.0 * units.mag,
            # airmass=0.0,
            airmass_err=0.0,
            # n_matches=zp_dict["n_matches"],
            image_name="self",
            **zp_dict
        )
        self.zeropoint_output_paths[cat_name.lower()] = output_path
        self.add_log(
            action=f"Calculated zeropoint as {zp_dict['zeropoint_img']} +/- {zp_dict['zeropoint_img_err']}, from {zp_dict['catalogue']}.",
            method=self.zeropoint,
            output_path=output_path
        )
        self.update_output_file()
        return zp_dict

    def get_zeropoint(
            self,
            cat_name: str,
            img_name: str = 'self'
    ):
        if cat_name == "best":
            zp_dict = self.zeropoint_best
        elif cat_name not in self.zeropoints:
            raise KeyError(f"Zeropoint {cat_name} does not exist.")
        elif img_name not in self.zeropoints[cat_name]:
            raise KeyError(f"Zeropoint from {img_name} does not exist")
        else:
            zp_dict = self.zeropoints[cat_name][img_name]

        return zp_dict

    def add_zeropoint(
            self,
            catalogue: str,
            zeropoint: Union[float, units.Quantity],
            zeropoint_err: Union[float, units.Quantity],
            extinction: Union[float, units.Quantity],
            extinction_err: Union[float, units.Quantity],
            airmass: float,
            airmass_err: float,
            n_matches: int = None,
            image_name: str = "self",
            **kwargs
    ):
        zp_dict = kwargs.copy()
        if extinction is None:
            extinction = extinction_err = 0. * units.mag
        elif extinction_err is None:
            extinction_err = 0. * units.mag

        if airmass is None:
            airmass = airmass_err = 0.
        elif airmass_err is None:
            airmass_err = 0.
        zp_dict.update({
            "zeropoint": u.check_quantity(zeropoint, units.mag),
            "zeropoint_err": u.check_quantity(zeropoint_err, units.mag),
            "extinction": u.check_quantity(extinction, units.mag),
            "extinction_err": u.check_quantity(extinction_err, units.mag),
            "airmass": airmass,
            "airmass_err": airmass_err,
            "catalogue": catalogue.lower(),
            "n_matches": n_matches,
            "image_name": image_name
        })
        print(f"Adding zeropoint to image {self.path}"
              f"\nfrom {catalogue} "
              f"\non {image_name}:")
        print(f"\t {zeropoint=} +/- {zeropoint_err}")
        print(f"\t {airmass=} +/- {airmass_err}")
        print(f"\t {extinction=} +/- {extinction_err}")
        zp_dict["zeropoint_img"] = zp_dict["zeropoint"] - zp_dict["extinction"] * zp_dict["airmass"]
        zp_dict['zeropoint_img_err'] = np.sqrt(
            zp_dict["zeropoint_err"] ** 2 + u.uncertainty_product(
                zp_dict["extinction"] * zp_dict["airmass"],
                (zp_dict["extinction"], zp_dict["extinction_err"]),
                (zp_dict["airmass"], zp_dict["airmass_err"])
            ) ** 2
        )

        img_key = image_name

        cat_key = catalogue.lower()
        if cat_key not in self.zeropoints:
            self.zeropoints[cat_key] = {}
        self.zeropoints[cat_key][img_key] = zp_dict
        self.update_output_file()
        return zp_dict

    def add_zeropoint_from_other(self, other: 'ImagingImage'):
        if other.filter_name != self.filter_name:
            raise ValueError(
                f"Zeropoints must come from images with the same filter; other filter {other.filter_name} does not match this filter {self.filter_name}.")
        if other.instrument_name != self.instrument_name:
            raise ValueError(
                f"Zeropoints must come from images with the same instrument; other instrument {other.instrument_name} does not match this instrument {self.instrument_name}.")

        airmass = self.extract_airmass()
        airmass_err = self.airmass_err

        airmass_other = other.extract_airmass()
        airmass_other_err = other.airmass_err

        delta_airmass = airmass - airmass_other
        delta_airmass_err = u.uncertainty_sum(airmass_err, airmass_other_err)

        for source in other.zeropoints:
            if 'self' in other.zeropoints[source]:
                zeropoint = other.zeropoints[source]['self']
                zeropoint.update({
                    "airmass": delta_airmass,
                    "airmass_err": delta_airmass_err,
                    "image_name": other.path,
                    "extinction": self.extinction_atmospheric,
                    "extinction_err": self.extinction_atmospheric_err
                })
                self.add_zeropoint(
                    **zeropoint
                )
        self.update_output_file()

    def aperture_areas(self):
        self.load_source_cat()
        self.extract_pixel_scale()

        self.source_cat["A_IMAGE"] = self.source_cat["A_WORLD"].to(units.pix, self.pixel_scale_y)
        self.source_cat["B_IMAGE"] = self.source_cat["A_WORLD"].to(units.pix, self.pixel_scale_y)
        self.source_cat["KRON_AREA_IMAGE"] = self.source_cat["A_IMAGE"] * self.source_cat["B_IMAGE"] * np.pi

        if self.source_cat_dual is not None:
            self.source_cat_dual["A_IMAGE"] = self.source_cat_dual["A_WORLD"].to(units.pix, self.pixel_scale_y)
            self.source_cat_dual["B_IMAGE"] = self.source_cat_dual["A_WORLD"].to(units.pix, self.pixel_scale_y)
            self.source_cat_dual["KRON_AREA_IMAGE"] = self.source_cat_dual["A_IMAGE"] * self.source_cat_dual[
                "B_IMAGE"] * np.pi

        self.add_log(
            action=f"Calculated area of FLUX_AUTO apertures.",
            method=self.aperture_areas,
        )
        self.update_output_file()

    def calibrate_magnitudes(self, zeropoint_name: str = "best", force: bool = True, dual: bool = False):
        cat = self.get_source_cat(dual=dual, force=True)
        if cat is None:
            raise ValueError(f"Catalogue ({dual=}) could not be loaded.")

        self.extract_exposure_time()

        zp_dict = self.get_zeropoint(cat_name=zeropoint_name)

        if force or f"MAG_AUTO_ZP_{zeropoint_name}" not in cat.colnames:
            mags = self.magnitude(
                flux=cat["FLUX_AUTO"],
                flux_err=cat["FLUXERR_AUTO"],
                cat_name=zeropoint_name
            )
            cat[f"ZP_{zeropoint_name}"] = zp_dict["zeropoint"]
            cat[f"ZPERR_{zeropoint_name}"] = zp_dict["zeropoint_err"]
            cat[f"AIRMASS_{zeropoint_name}"] = zp_dict["airmass"]
            cat[f"AIRMASSERR_{zeropoint_name}"] = zp_dict["airmass_err"]
            cat["EXT_ATM"] = zp_dict["extinction"]
            cat["EXT_ATMERR"] = zp_dict["extinction_err"]
            cat[f"ZP_{zeropoint_name}_ATM_CORR"] = zp_dict["zeropoint_img"]
            cat[f"ZP_{zeropoint_name}_ATM_CORRERR"] = zp_dict["zeropoint_img_err"]

            cat[f"MAG_AUTO_ZP_{zeropoint_name}"] = mags[0]
            cat[f"MAGERR_AUTO_ZP_{zeropoint_name}"] = mags[1]
            cat[f"MAG_AUTO_ZP_{zeropoint_name}_no_ext"] = mags[2]
            cat[f"MAGERR_AUTO_ZP_{zeropoint_name}_no_ext"] = mags[3]

            if "FLUX_PSF" in cat.colnames:
                mags = self.magnitude(
                    flux=cat["FLUX_PSF"],
                    flux_err=cat["FLUXERR_PSF"],
                    cat_name=zeropoint_name
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
            cat_name: str = 'best',
            img_name: str = 'self'
    ):

        zp_dict = self.get_zeropoint(cat_name=cat_name, img_name=img_name)

        if zp_dict is None:
            raise ValueError(f"The {cat_name} zeropoint on {img_name}, for {self.name}, does not appear to exist.")

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
            airmass=0.0,
            airmass_err=0.0,
            ext=0.0 * units.mag,
            ext_err=0.0 * units.mag,
            colour_term=0.0,
            colour=0.0 * units.mag,
        )

        return mag, mag_err, mag_no_ext_corr, mag_no_ext_corr_err

    def estimate_depth(
            self,
            zeropoint_name: str = "best",
            dual: bool = False,
            star_tolerance: int = 1,
            do_magnitude_calibration: bool = True
    ):
        """
        Use various measures of S/N to estimate image depth at a range of sigmas.
        :param zeropoint_name:
        :param dual:
        :return:
        """

        # self.signal_to_noise_ccd(dual=dual)

        self.signal_to_noise_measure(dual=dual)
        if do_magnitude_calibration:
            self.calibrate_magnitudes(zeropoint_name=zeropoint_name, dual=dual)

        source_cat = self.get_source_cat(dual=dual)

        # "max" stores the magnitude of the faintest object with S/N > x sigma
        self.depth = {"max": {}, "secure": {}}
        # "secure" finds the brightest object with S/N < x sigma, then increments to the
        # overall; thus giving the faintest magnitude at which we can be confident of a detection

        stars = u.trim_to_class(cat=source_cat, modify=True, allowed=np.arange(0, star_tolerance + 1))

        if stars is None or len(stars) < 10:
            stars = source_cat[source_cat["CLASS_STAR"] >= 0.9]
        source_cat = stars

        for snr_key in ["PSF", "AUTO"]:  # ["SNR_CCD", "SNR_MEASURED", "SNR_SE"]:
            # We do this to ensure that, in the "secure" step, object i+1 is the next-brightest in the catalogue
            if f"FLUX_{snr_key}" not in source_cat.colnames:
                continue
            source_cat.sort(f"FLUX_{snr_key}")
            self.depth["max"][f"SNR_{snr_key}"] = {}
            self.depth["secure"][f"SNR_{snr_key}"] = {}
            # Dispose of the infinite SNRs and mags
            source_cat = source_cat[np.invert(np.isinf(source_cat[f"MAG_{snr_key}"]))]
            source_cat = source_cat[np.invert(np.isinf(source_cat[f"SNR_{snr_key}"]))]
            source_cat.sort(f"FLUX_{snr_key}")
            for sigma in range(1, 6):
                u.debug_print(1, "ImagingImage.estimate_depth(): snr_key, sigma ==", snr_key, sigma)
                # Faintest source at x-sigma:
                u.debug_print(
                    1, f"ImagingImage.estimate_depth(): source_cat[SNR_{snr_key}].unit ==",
                    source_cat[f"SNR_{snr_key}"].unit)
                cat_more_xsigma = source_cat[source_cat[f"SNR_{snr_key}"] > sigma]
                self.depth["max"][f"SNR_{snr_key}"][f"{sigma}-sigma"] = np.max(
                    cat_more_xsigma[f"MAG_{snr_key}_ZP_{zeropoint_name}"])

                # Brightest source less than x-sigma (kind of)
                # Get the sources with SNR less than x-sigma
                source_less_sigma = source_cat[source_cat[f"SNR_{snr_key}"] < sigma]
                if len(source_less_sigma) > 0:
                    # Get the source with the greatest flux
                    i = np.argmax(source_less_sigma[f"FLUX_{snr_key}"])
                    # Find its counterpart in the full catalogue
                    i, _ = u.find_nearest(source_cat["NUMBER"], source_less_sigma[i]["NUMBER"])
                    # Get the source that is next up in brightness (being brighter)
                    i += 1
                    src_lim = source_cat[i]

                else:
                    src_lim = source_cat[source_cat[f"SNR_{snr_key}"].argmin()]

                self.depth["secure"][f"SNR_{snr_key}"][f"{sigma}-sigma"] = src_lim[f"MAG_{snr_key}_ZP_{zeropoint_name}"]
                self.update_output_file()

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

    def register(
            self,
            target: 'ImagingImage',
            output_path: str,
            ext: int = 0,
            ext_target: int = 0,
            trim: bool = True,
            **kwargs
    ):
        from astroalign import register
        self.load_data()
        target.load_data()

        data_source = self.data[ext]
        data_source = u.sanitise_endianness(data_source)
        data_target = target.data[ext_target]
        data_target = u.sanitise_endianness(data_target)
        u.debug_print(0,
                      f"Attempting registration of {self.name} (Chip {self.extract_chip_number()}) against {target.name} (Chip {target.extract_chip_number()})")
        registered, footprint = register(data_source, data_target, **kwargs)

        self.copy(output_path)
        with fits.open(output_path, mode="update") as new_file:
            new_file[0].data = registered
            u.debug_print(1, "Writing registered image to", output_path)
            new_file.writeto(output_path, overwrite=True)

        new_image = self.new_image(path=output_path)
        new_image.transfer_wcs(target)

        if trim:
            frame_value = new_image.detect_frame_value(ext=ext)
            if frame_value is not None:
                left, right, bottom, top = new_image.detect_edges(frame_value=frame_value)
                trimmed = new_image.trim(left=left, right=right, bottom=bottom, top=top)
                new_image = trimmed

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

    def correct_astrometry(
            self,
            output_dir: str = None,
            tweak: bool = True,
            time_limit: int = None,
            am_flags: list = (),
            am_params: dict = None,
            **kwargs
    ):
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
        if "search_radius" not in kwargs:
            kwargs["search_radius"] = 4.0 * units.arcmin
        success = solve_field(
            image_files=self.path,
            base_filename=base_filename,
            overwrite=True,
            tweak=tweak,
            guess_scale=True,
            centre=self.pointing,
            time_limit=time_limit,
            am_flags=am_flags,
            am_params=am_params,
            **kwargs
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
        new_image.set_header_item("GAIA", True)
        new_image.add_log(
            "Astrometry corrected using Astrometry.net.",
            method=self.correct_astrometry,
            packages=["astrometry.net"])
        new_image.update_output_file()
        new_image.write_fits_file()
        return new_image

    def correct_astrometry_coarse(
            self,
            output_dir: str = None,
            cat: table.Table = None,
            ext: int = 0,
            cat_name: str = None
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

        ra_scale, dec_scale = self.extract_world_scale(ext=ext)

        new.load_headers()
        if not np.isnan(diagnostics["median_offset_x"].value) and not np.isnan(diagnostics["median_offset_y"].value):

            new.shift_wcs(
                delta_ra=diagnostics["median_offset_x"].to(units.deg, ra_scale).value,
                delta_dec=diagnostics["median_offset_y"].to(units.deg, dec_scale).value
            )

            new.add_log(
                "Astrometry corrected using median offsets from reference catalogue.",
                method=self.correct_astrometry_coarse,
                input_path=self.path,
                output_path=new_path,
                ext=ext
            )
            if cat_name.lower() == 'gaia':
                new.set_header_item("GAIA", True)
            new.write_fits_file()
            return new
        else:
            u.rm_check(new_path)
            return None

    def shift_wcs(self, delta_ra: units.Quantity, delta_dec: units.Quantity, ext: int = 0):
        delta_ra = u.dequantify(delta_ra, unit=units.deg)
        delta_dec = u.dequantify(delta_dec, unit=units.deg)
        self.headers[ext]["CRVAL1"] += delta_ra
        self.headers[ext]["CRVAL2"] += delta_dec
        self.add_log(
            f"Shifted WCS coordinate reference value by RA+={delta_ra}, DEC+={delta_dec}.",
            method=self.shift_wcs,
            ext=ext
        )

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

        if "GAIA" in other_header:
            insert["GAIA"] = other_header["GAIA"]

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
            # star_tolerance: float = 1,
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
        :param star_tolerance: Maximum CLASS_FLAG for object to be considered.
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

        with quantity_support():
            plt.scatter(self.source_cat["RA"].value, self.source_cat["DEC"].value, marker='x')
            plt.xlabel("Right Ascension (Catalogue)")
            plt.ylabel("Declination (Catalogue)")
            # plt.colorbar(label="Offset of measured position from catalogue (\")")
            if show_plots:
                plt.show()
            plt.savefig(os.path.join(output_path, f"{self.name}_sourcecat_sky.pdf"))
            plt.close()

            plt.scatter(reference_cat[ra_col].value, reference_cat[dec_col].value, marker='x')
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

            plt.scatter(
                self.source_cat["RA"],
                self.source_cat["DEC"],
                marker='x'
            )
            plt.scatter(
                reference_cat[ra_col][in_footprint],
                reference_cat[dec_col][in_footprint],
                marker='x'
            )
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
                # star_tolerance=star_tolerance
            )
            if len(matches_source_cat) < 1:
                self.astrometry_err = -99 * units.arcsec
                self.ra_err = -99 * units.arcsec
                self.dec_err = -99 * units.arcsec
                self.headers[0]["ASTM_RMS"] = self.astrometry_err.value
                self.headers[0]["RA_RMS"] = self.ra_err.value
                self.headers[0]["DEC_RMS"] = self.dec_err.value
                self.write_fits_file()
                self.update_output_file()
                return -99.0

            matches_coord = SkyCoord(matches_source_cat["RA"], matches_source_cat["DEC"])

            sigma_clip = SigmaClip(sigma=3.)
            distance_clipped = sigma_clip(distance, masked=False)
            distance_clipped_masked = sigma_clip(distance, masked=True)
            mask = ~distance_clipped_masked.mask

            offset_ra = matches_source_cat["RA"][mask] - matches_ext_cat[ra_col][mask]
            offset_dec = matches_source_cat["DEC"][mask] - matches_ext_cat[dec_col][mask]

            mean_offset = np.mean(distance_clipped)
            median_offset = np.median(distance_clipped)
            rms_offset = np.sqrt(np.mean(distance_clipped ** 2))
            rms_offset_ra = np.sqrt(np.mean(offset_ra ** 2))
            rms_offset_dec = np.sqrt(np.mean(offset_dec ** 2))

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

            plt.hist(
                distance.to(units.arcsec).value,
                bins=int(np.sqrt(len(distance))),
                label="Full sample"
            )
            plt.hist(
                distance_clipped.to(units.arcsec).value,
                edgecolor='black',
                linewidth=1.2,
                label="Sigma-clipped",
                fc=(0, 0, 0, 0),
                bins=int(np.sqrt(len(distance_clipped)))
            )
            plt.xlabel("Offset (\")")
            plt.legend()
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
        self.astrometry_stats["rms_offset_ra"] = rms_offset_ra.to(units.arcsec)
        self.astrometry_stats["rms_offset_dec"] = rms_offset_dec.to(units.arcsec)
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
        # self.astrometry_stats["star_tolerance"] = star_tolerance
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
        self.ra_err = self.astrometry_stats["rms_offset_ra"]
        self.dec_err = self.astrometry_stats["rms_offset_dec"]

        self.source_cat["RA_ERR"] = np.sqrt(
            self.source_cat["ERRX2_WORLD"].to(units.arcsec ** 2) + self.ra_err ** 2)
        self.source_cat["DEC_ERR"] = np.sqrt(
            self.source_cat["ERRY2_WORLD"].to(units.arcsec ** 2) + self.dec_err ** 2)

        if not np.isnan(self.astrometry_err.value):
            self.headers[0]["ASTM_RMS"] = self.astrometry_err.value
            self.headers[0]["RA_RMS"] = self.ra_err.value
            self.headers[0]["DEC_RMS"] = self.dec_err.value
        self.write_fits_file()
        self.update_output_file()

        return self.astrometry_stats

    def psf_diagnostics(
            self,
            mag_max: float = 0.0 * units.mag,
            mag_min: float = -50. * units.mag,
            match_to: table.Table = None,
            star_class_tol: int = 0,
            frame: float = None,
            ext: int = 0,
            target: SkyCoord = None,
            near_radius: units.Quantity = 1 * units.arcmin,
            output_path: str = None
    ):
        self.open()
        self.load_source_cat()
        if frame is None:
            _, scale = self.extract_pixel_scale()
            frame = (4 * units.arcsec).to(units.pix, scale).value
        u.debug_print(2, f"ImagingImage.psf_diagnostics(): {self}.source_cat_path ==", self.source_cat_path)
        if output_path is None:
            output_path = self.data_path
        stars_moffat, stars_gauss, stars_sex = ph.image_psf_diagnostics(
            hdu=self.hdu_list,
            cat=self.source_cat,
            mag_max=mag_max,
            mag_min=mag_min,
            match_to=match_to,
            frame=frame,
            near_centre=target,
            near_radius=near_radius,
            output=output_path,
            plot_file_prefix=self.name,
            ext=ext,
            star_class_tol=star_class_tol,
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
            "target": target,
            "radius": near_radius,
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
        self.headers[ext]["PSF_FWHM"] = self.fwhm_median_gauss.to(units.arcsec).value
        self.headers[ext]["PSF_FWHM_ERR"] = self.fwhm_sigma_gauss.to(units.arcsec).value
        self.add_log(
            action=f"Calculated PSF FWHM statistics.",
            method=self.psf_diagnostics,
            packages=["source-extractor", "psfex"]
        )
        self.psf_stats = results
        self.update_output_file()
        self.write_fits_file()
        return results, stars_moffat, stars_gauss, stars_sex

    def trim(
            self,
            left: Union[int, units.Quantity] = None,
            right: Union[int, units.Quantity] = None,
            bottom: Union[int, units.Quantity] = None,
            top: Union[int, units.Quantity] = None,
            output_path: str = None,
            ext: int = 0
    ):
        left = u.dequantify(left, unit=units.pix)
        right = u.dequantify(right, unit=units.pix)
        bottom = u.dequantify(bottom, unit=units.pix)
        top = u.dequantify(top, unit=units.pix)

        if output_path is None:
            output_path = self.path.replace(".fits", "_trimmed.fits")
        image = self.copy_with_outputs(output_path)

        image.load_headers()
        image.load_data()

        header = image.headers[ext]

        trimmed_data, margins = u.trim_image(
            data=image.data[ext],
            left=left, right=right, bottom=bottom, top=top,
            return_margins=True
        )

        crpix1 = header['CRPIX1'] - left
        crpix2 = header['CRPIX2'] - bottom

        # Move reference pixel to account for trim; this should keep the same sky coordinate at the ref pix
        image.set_header_items(
            items={
                'CRPIX1': crpix1,
                'CRPIX2': crpix2
            },
            ext=ext,
            write=False
        )

        image.data[ext] = trimmed_data

        image.add_log(
            action=f"Trimmed image to margins left={left}, right={right}, bottom={bottom}, top={top}",
            method=self.trim,
            output_path=output_path
        )
        image.update_output_file()
        image.write_fits_file()

        return image

    def convert_from_cs(self, output_path: str, ext: int = 0):
        """
        NOT IMPLEMENTED.
        Assuming units of counts / second, converts the image back to total counts.
        :param output_path: Path to write converted file to.
        :param ext: FITS extension to modify.
        :return new: ImagingImage object representing the modified file.
        """
        pass

    def convert_to_cs(self, output_path: str, ext: int = 0):
        """
        Converts the image to flux (units of counts per second) and writes to a new file.
        :param output_path: Path to write converted file to.
        :param ext: FITS extension to modify.
        :return new: ImagingImage object representing the modified file.
        """
        new = self.copy(output_path)
        gain = self.extract_gain()
        exp_time = self.extract_exposure_time()
        saturate = self.extract_saturate()
        read_noise = self.extract_noise_read()

        new.load_data()
        new_data = new.data[ext]
        # new_data *= gain
        new_data /= exp_time
        new.data[ext] = new_data

        u.debug_print(1, "Image.concert_to_cs() 2: new_data.unit ==", new_data.unit)

        header_keys = self.header_keys()
        new.set_header_items(
            items={
                header_keys["noise_read"]: str(new_data.unit),
                header_keys["gain"]: gain * exp_time,
                header_keys["gain_old"]: gain,
                header_keys["exposure_time"]: 1.0,
                header_keys["exposure_time_old"]: exp_time.value,
                header_keys["saturate_old"]: saturate,
                header_keys["saturate"]: saturate / exp_time.value,
                header_keys["noise_read"]: read_noise / exp_time.value,
                header_keys["noise_read_old"]: read_noise,
                header_keys["integration_time"]: exp_time.value
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
        from ccdproc import cosmicray_lacosmic
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

    def scale_to_jansky(
            self,
            ext: int = 0,
            *args
    ):
        self.load_data()
        self.load_output_file()
        data = self.data[ext].value
        zp = self.zeropoint_best["zeropoint_img"].value
        exptime = self.extract_exposure_time().value
        data[data <= 0.] = np.min(data[data > 0.])
        data_scaled = 3631 * units.Jansky * (data / exptime) * 10 ** (zp / -2.5)
        extra_vals = []
        for v in args:
            if v is not None:
                v = u.dequantify(v)
                extra_vals.append(3631 * units.Jansky * (v / exptime) * 10 ** (zp / -2.5))
            else:
                extra_vals.append(v)
        if extra_vals:
            return data_scaled, extra_vals
        else:
            return data_scaled

    def reproject(
            self,
            other_image: 'ImagingImage',
            ext: int = 0,
            output_path: str = None,
            include_footprint: bool = False,
            write_footprint: bool = True,
            method: str = 'exact',
            mask_mode: bool = False
    ):
        import reproject as rp
        if output_path is None:
            output_path = self.path.replace(".fits", "_reprojected.fits")
        other_image.load_headers(force=True)
        print(f"Reprojecting {self.filename} into the pixel space of {other_image.filename}")
        if method == 'exact':
            reprojected, footprint = rp.reproject_exact(self.path, other_image.headers[ext])  # , parallel=True)
        elif method == 'adaptive':
            reprojected, footprint = rp.reproject_adaptive(self.path, other_image.headers[ext])
        elif method in ['interp', 'interpolate', 'interpolation']:
            reprojected, footprint = rp.reproject_interp(self.path, other_image.headers[ext])
        else:
            raise ValueError(f"Reprojection method {method} not recognised.")

        # if not mask_mode:
        reprojected *= other_image.extract_unit(astropy=True)
        footprint *= units.pix
        if mask_mode:
            reprojected = np.round(reprojected)

        if output_path == self.path:
            reprojected_image = self
        else:
            reprojected_image = self.copy(output_path)
        reprojected_image.load_data(force=True)
        reprojected_image.data[ext] = reprojected

        if include_footprint:
            new_hdu = fits.ImageHDU()
            reprojected_image.headers.append(new_hdu.header)
            reprojected_image.data.append(footprint)

        if write_footprint:
            footprint_file = self.copy_with_outputs(output_path.replace(".fits", "_footprint.fits"))
            footprint_file.data[0] = footprint
            footprint_file.write_fits_file()

        reprojected_image.add_log(
            action=f"Reprojected into pixel space of {other_image}.",
            method=self.reproject,
            output_path=output_path
        )
        reprojected_image.update_output_file()
        reprojected_image.transfer_wcs(other_image=other_image)
        # reprojected_image.write_fits_file()

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

    def match_to_cat(
            self,
            cat: Union[str, table.QTable],
            ra_col: str = "ra",
            dec_col: str = "dec",
            offset_tolerance: units.Quantity = 1 * units.arcsec,
            star_tolerance: float = None,
            dual: bool = False
    ):

        source_cat = self.get_source_cat(dual=dual)

        _, scale = self.extract_pixel_scale()

        if star_tolerance is not None:
            source_cat = u.trim_to_class(
                cat=source_cat,
                modify=True,
                allowed=np.arange(0, star_tolerance + 1)
            )

        u.debug_print(2, "len(source_cat) match_catalogs:", len(source_cat))
        matches_source_cat, matches_ext_cat, distance = astm.match_catalogs(
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

        matches_source_cat["PIX_OFFSET_FROM_REF"] = distance.to(units.pix, scale)

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

        self.add_log(
            action=f"Estimated SNR using CCD Equation.",
            method=self.signal_to_noise_ccd,
        )
        self.update_output_file()

        return source_cat["SNR_CCD"]

    def signal_to_noise_measure(self, dual: bool = False):
        print("Measuring signal-to-noise of sources...")

        source_cat = self.get_source_cat(dual=dual)
        source_cat["SNR_AUTO"] = source_cat["FLUX_AUTO"] / source_cat["FLUXERR_AUTO"]
        if "FLUX_PSF" in source_cat.colnames:
            source_cat["SNR_PSF"] = source_cat["FLUX_PSF"] / source_cat["FLUXERR_PSF"]

        # self.load_data()
        # _, scale = self.extract_pixel_scale()
        # mask = self.generate_mask(method='sep')
        # mask = mask.astype(bool)
        # bkg = self.calculate_background(method='sep', mask=mask)
        # rms = bkg.rms()
        #
        # gain = self.extract_gain() / units.electron
        #
        # snrs = []
        # snrs_se = []
        # sigma_fluxes = []
        #
        # for cat_obj in source_cat:
        #     x = cat_obj["X_IMAGE"].value - 1
        #     y = cat_obj["Y_IMAGE"].value - 1
        #
        #     a = cat_obj["A_WORLD"].to(units.pix, scale).value
        #     b = cat_obj["B_WORLD"].to(units.pix, scale).value
        #
        #     theta = u.world_angle_se_to_pu(
        #         cat_obj["THETA_WORLD"],
        #         rot_angle=self.extract_rotation_angle()
        #     )
        #
        #     ap = photutils.aperture.EllipticalAperture(
        #         [x, y],
        #         a=a,
        #         b=b,
        #         theta=theta
        #     )
        #
        #     ap_mask = ap.to_mask(method='center')
        #
        #     flux = cat_obj["FLUX_AUTO"]
        #
        #     ap_rms = ap_mask.multiply(rms)
        #     sigma_flux = np.sqrt(ap_rms.sum()) * units.ct
        #     snr = flux / np.sqrt(sigma_flux ** 2 + flux / gain)
        #
        #     snr_se = flux / cat_obj["FLUXERR_AUTO"].value
        #
        #     snrs.append(snr.value)
        #     sigma_fluxes.append(sigma_flux.value)
        #     snrs_se.append(snr_se.value)
        #
        # source_cat["SNR_MEASURED"] = snrs
        # source_cat["NOISE_MEASURED"] = sigma_fluxes
        # source_cat["SNR_PSF"] = snrs_se

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
        self.source_cat["A_IMAGE"] = self.source_cat["A_WORLD"].to(units.pix, self.pixel_scale_y)
        self.source_cat["B_IMAGE"] = self.source_cat["B_WORLD"].to(units.pix, self.pixel_scale_y)
        self.source_cat_dual["A_IMAGE"] = self.source_cat_dual["A_WORLD"].to(units.pix, self.pixel_scale_y)
        self.source_cat_dual["B_IMAGE"] = self.source_cat_dual["B_WORLD"].to(units.pix, self.pixel_scale_y)

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

    def plot_apertures(self, dual=True, output: str = None, show: bool = False):
        cat = self.get_source_cat(dual=dual)

        if cat is not None:
            pl.plot_all_params(image=self.path, cat=cat, kron=True, show=False)
            plt.title(self.filter_name)
            if output is None:
                output = os.path.join(self.data_path, f"{self.name}_source_cat_dual-{dual}.pdf")
            plt.savefig(output)
            if show:
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

    def plot_subimage(
            self,
            centre: SkyCoord = None,
            frame: units.Quantity = None,
            corners: Tuple[SkyCoord] = None,
            ext: int = 0,
            fig: plt.Figure = None,
            ax: plt.Axes = None,
            n: int = 1, n_x: int = 1, n_y: int = 1,
            show_cbar: bool = False,
            show_grid: bool = False,
            ticks: int = None,
            show_coords: bool = True,
            ylabel: str = None,
            reverse_y=False,
            imshow_kwargs: dict = None,  # Can include cmap
            normalize_kwargs: dict = None,  # Can include vmin, vmax
            output_path: str = None,
            mask: np.ndarray = None,
            **kwargs,
    ):
        self.load_data()
        _, scale = self.extract_pixel_scale()
        data = self.data[ext].value * 1.0
        other_args = {}
        if centre is not None and frame is not None:
            x, y = self.world_to_pixel(centre, 0)
            frame = u.check_quantity(
                number=frame,
                unit=units.pix,
                allow_mismatch=True,
                enforce_equivalency=False
            )
            other_args["x"] = x
            other_args["y"] = y
            left, right, bottom, top = u.frame_from_centre(frame.to(units.pix, scale).value, x, y, data)
        elif corners is not None:
            x_0, y_0 = self.world_to_pixel(corners[0], 0)
            x_1, y_1 = self.world_to_pixel(corners[1], 0)
            xs = x_1, x_0
            left = int(min(xs))
            right = int(max(xs))
            ys = y_1, y_0
            bottom = int(min(ys))
            top = int(max(ys))

        else:
            left = 0
            right = data.shape[1]
            bottom = 0
            top = data.shape[0]

        # print(type(data), data[bottom:top, left:right])
        if mask is not None:
            data_masked = data * np.invert(mask.astype(bool)).astype(int)
            data_masked += mask * np.nanmedian(data[bottom:top, left:right])
            data = data_masked
        u.debug_print(1, "ImagingImage.plot_subimage(): frame ==", frame)

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

        if ax is None:
            if show_coords:
                projection = self.wcs
            else:
                projection = None

            ax = fig.add_subplot(n_y, n_x, n, projection=projection)

        if not show_coords:
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.set_yticks([])
            frame1.axes.invert_yaxis()

        ax.imshow(
            data,
            norm=ImageNormalize(
                data[bottom:top, left:right],
                **normalize_kwargs
            ),
            **imshow_kwargs
        )
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        # ax.set_xlabel("Right Ascension (J2000)", size=16)
        # ax.set_ylabel("Declination (J2000)", size=16, rotation=-90)
        ax.tick_params(labelsize=14)
        ax.yaxis.set_label_position("right")

        # plt.tight_layout()

        if output_path is not None:
            fig.savefig(output_path)

        return ax, fig, other_args

    def prep_for_colour(
            self,
            output_path: str,
            frame: units.Quantity,
            centre: SkyCoord = None,
            vmax: float = None,
            vmin: float = None,
            ext: int = 0,
            scale_to_jansky: bool = False
    ):
        self.extract_pixel_scale(ext)
        frame = frame.to(units.pix, self.pixel_scale_y).value

        self.load_data()
        x, y = self.world_to_pixel(centre, 0)
        left, right, bottom, top = u.frame_from_centre(frame=frame, x=x, y=y, data=self.data[ext])
        trimmed = self.trim(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            output_path=output_path
        )
        trimmed.load_wcs(ext)

        if scale_to_jansky:
            data, vs = trimmed.scale_to_jansky(ext, vmax, vmin)
            vmax = u.dequantify(vs[0])
            vmin = u.dequantify(vs[1])
            data = data.value
        else:
            data = trimmed.data[0].value

        if vmax is not None:
            data[data > vmax] = vmax
        if vmin is not None:
            data[data < vmin] = vmin

        median = np.nanmedian(data)
        data_subbed = data - median
        data_subbed[np.isnan(data_subbed)] = median
        # data_scaled = data_subbed * 255 / np.max(data_subbed)
        return data_subbed, trimmed

    def nice_frame(
            self,
            row: Union[table.Row, dict],
            frame: units.Quantity = 10 * units.pix,
    ):
        self.extract_pixel_scale()
        u.debug_print(1, "ImagingImage.nice_frame(): row['KRON_RADIUS'], row['A_WORLD'] ==", row['KRON_RADIUS'],
                      row['A_WORLD'].to(units.arcsec))
        kron_a = row['KRON_RADIUS'] * row['A_WORLD']
        u.debug_print(1, "ImagingImage.nice_frame(): kron_a ==", kron_a)
        pix_scale = self.pixel_scale_y
        u.debug_print(1, "ImagingImage.nice_frame(): self.pixel_scale_dec ==", self.pixel_scale_y)
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

        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot()

        self.load_headers()
        kron_a = row['KRON_RADIUS'] * row['A_WORLD']
        kron_b = row['KRON_RADIUS'] * row['B_WORLD']
        # kron_theta = -kron_theta + ff.get_rotation_angle(
        #     header=self.headers[ext],
        #     astropy_units=True)
        this_frame = self.nice_frame(row=row, frame=frame)
        mid_x = row["X_IMAGE"]
        mid_y = row["Y_IMAGE"]
        self.open()
        left, right, bottom, top = u.frame_from_centre(frame=this_frame, x=mid_x, y=mid_y, data=self.hdu_list[ext].data)
        image_cut = ff.trim(hdu=self.hdu_list, left=left, right=right, bottom=bottom, top=top)
        norm = pl.nice_norm(image=image_cut[ext].data)
        ax.imshow(image_cut[ext].data, origin='lower', norm=norm)
        # theta =
        pl.plot_gal_params(
            hdu=image_cut,
            ras=[row["RA"].value],
            decs=[row["DEC"].value],
            a=[row["A_WORLD"].value],
            b=[row["B_WORLD"].value],
            theta=[row["THETA_IMAGE"].value],
            world=True,
            show_centre=True
        )
        pl.plot_gal_params(
            hdu=image_cut,
            ras=[row["RA"].value],
            decs=[row["DEC"].value],
            a=[kron_a.value],
            b=[kron_b.value],
            theta=[row["THETA_IMAGE"].value],
            world=True,
            show_centre=True
        )
        if title is None:
            title = self.name
        title = u.latex_sanitise(title)
        ax.set_title(title)
        fig.savefig(os.path.join(output))
        if show:
            fig.show()
        self.close()
        plt.close(fig)
        return

    def plot(self, fig: plt.Figure = None, ext: int = 0, **kwargs):

        if fig is None:
            fig = plt.figure(figsize=(12, 12), dpi=1000)
        ax, fig = self.wcs_axes(fig=fig)
        self.load_data()
        data = u.dequantify(self.data[ext])
        ax.imshow(
            data, **kwargs,
            norm=ImageNormalize(
                data,
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

        # TODO: Fix treatment of zeropoint here
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
                fwhm=self.fwhm_psfex.to(units.pix, self.pixel_scale_y)
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
                scale.to(units.pix, self.pixel_scale_y)
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

        plt.scatter(sources["mag_inserted"], sources["SPREAD_MODEL"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Spread Model")
        plt.savefig(os.path.join(output_dir, "spread_model.png"))
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

        plt.scatter(sources["mag_inserted"], sources["SNR_PSF"])
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
            write: str = None,
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
            back = bkg.background
            self.data_sub_bkg[ext] = (data - back)

        else:
            raise ValueError(f"Unrecognised method {method}.")

        if isinstance(write, str):
            back_file = self.copy(write)
            back_file.load_data()
            back_file.load_headers()
            back_file.data[ext] = back
            back_file.write_fits_file()

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
                # deblend_cont=True,
                clean=False,
                segmentation_map=True,
                minarea=min_area
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
            margins: tuple = (None, None, None, None),
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
            ext=ext,
            threshold=threshold,
            method=method,
            margins=margins
        )
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
            **generate_mask_kwargs
    ):
        self.load_data()
        if mask is None:
            mask = self.generate_mask(**generate_mask_kwargs)

        # if mask_type == 'zeroed-out'

        return np.ma.MaskedArray(self.data[ext].copy(), mask=mask)

    def write_mask(
            self,
            output_path: str,
            ext: int = 0,
            **mask_kwargs
    ) -> 'ImagingImage':
        """
        Generates and writes a source mask to a FITS file.
        Any argument accepted by generate_mask() can be passed as a keyword.
        :param output_path: path to write the mask file to.
        :param ext: FITS extension to modify.
        :return:
        """

        mask_file = self.copy(output_path)
        mask_file.load_data()
        mask_file.data[ext] = self.generate_mask(ext=ext, **mask_kwargs) * units.dimensionless_unscaled
        mask_file.write_fits_file()

        mask_file.add_log(
            action="Converted image to source mask.",
            method=self.source_extraction,
            output_path=output_path,
        )
        mask_file.update_output_file()
        return mask_file

    def mask_nearby(self):
        return True

    def detection_threshold(self):
        return 5.

    def do_subtract_background(self):
        return True

    def sep_aperture_photometry(
            self, x: float, y: float,
            aperture_radius: units.Quantity = 2.0 * units.arcsec,
            ext: int = 0,
            sub_background: bool = True
    ):
        self.extract_pixel_scale()
        pixel_radius = aperture_radius.to(units.pix, self.pixel_scale_y)
        self.calculate_background(ext=ext)
        if sub_background:
            data = self.data_sub_bkg[ext]
        else:
            data = u.sanitise_endianness(self.data[ext])
        flux, fluxerr, flag = sep.sum_circle(
            data,
            x, y,
            pixel_radius.value,
            err=self.sep_background[ext].rms(),
            gain=self.extract_gain().value)
        return flux, fluxerr, flag

    def sep_elliptical_photometry(
            self,
            centre: SkyCoord,
            a_world: units.Quantity,
            b_world: units.Quantity,
            theta_world: units.Quantity,
            kron_radius: float = 1.,
            ext: int = 0,
            output: str = None,
            mask_nearby=True,
            subtract_background: bool = True,
    ):

        if isinstance(output, str):
            back_output = output + "_back.fits"
            segmap_output = output + "_segmap.fits"
        else:
            back_output = None
            segmap_output = None

        self.calculate_background(ext=ext, write=back_output)
        self.load_wcs(ext=ext)
        self.extract_pixel_scale()
        if not self.wcs.footprint_contains(centre):
            return None, None, None, None
        x, y = self.wcs.all_world2pix(centre.ra.value, centre.dec.value, 0)
        x = u.check_iterable(x)
        y = u.check_iterable(y)
        a = u.check_iterable((a_world.to(units.pix, self.pixel_scale_y)).value)
        b = u.check_iterable((b_world.to(units.pix, self.pixel_scale_y)).value)
        kron_radius = u.check_iterable(kron_radius)
        rotation_angle = self.extract_rotation_angle(ext=ext)
        theta_deg = -theta_world + rotation_angle  # + 90 * units.deg
        theta = u.theta_range(theta_deg.to(units.rad)).value

        u.debug_print(2, f"sep_elliptical_photometry: mask_nearby == {mask_nearby}")

        if isinstance(mask_nearby, ImagingImage):
            mask = mask_nearby.data[0].value
        elif mask_nearby:
            mask = self.write_mask(
                unmasked=centre,
                ext=ext,
                method="sep",
                output_path=segmap_output
            ).data[0].value
        else:
            mask = np.zeros_like(self.data[ext].data)

        if subtract_background:
            data = self.data_sub_bkg[ext]
            back, _, _ = sep.sum_ellipse(
                data=self.sep_background[ext].back(),
                x=x, y=y,
                a=a, b=b,
                r=kron_radius,
                theta=theta,
            )
        else:
            data = u.sanitise_endianness(self.data[ext])
            back = [0.]

        flux, flux_err, flag = sep.sum_ellipse(
            data=data,
            x=x, y=y,
            a=a, b=b,
            r=kron_radius,
            theta=theta,
            err=self.sep_background[ext].rms(),
            gain=self.extract_gain().value,
            mask=mask.astype(bool),
        )

        if isinstance(output, str):
            # objects = sep.extract(self.data_sub_bkg[ext], 1.5, err=self.sep_background[ext].rms())
            this_frame = self.nice_frame({
                'A_WORLD': a_world,
                'B_WORLD': b_world,
                'KRON_RADIUS': kron_radius
            })

            plt.close()
            with quantity_support():
                ax, fig, _ = self.plot_subimage(
                    centre=centre,
                    frame=this_frame,
                    ext=ext,
                    mask=mask
                )

                # for i in range(len(objects)):
                #     e = Ellipse(
                #         xy=(objects["x"][i], objects["y"][i]),
                #         width=4*objects["a"][i],
                #         height=4*objects["b"][i],
                #         angle=objects["theta"][i] * 180. / np.pi)
                #     e.set_facecolor('none')
                #     e.set_edgecolor('red')
                #     ax.add_artist(e)
                #     ax.text(objects["x"][i], objects["y"][i], objects["theta"][i] * 180. / np.pi)

                theta_plot = (theta[0] * units.rad).to(units.deg).value

                e = Ellipse(
                    xy=(x[0], y[0]),
                    width=2 * kron_radius[0] * a[0],
                    height=2 * kron_radius[0] * b[0],
                    angle=theta_plot
                )
                e.set_facecolor('none')
                e.set_edgecolor('white')
                ax.add_artist(e)

                e = Ellipse(
                    xy=(x[0], y[0]),
                    width=2 * a[0],
                    height=2 * b[0],
                    angle=theta_plot
                )
                e.set_facecolor('none')
                e.set_edgecolor('white')
                ax.add_artist(e)

                ax.set_title(f"{a[0], b[0], kron_radius[0], theta_plot}")

                fig.savefig(output + ".png")

        return flux, flux_err, flag, back

    def sep_elliptical_magnitude(
            self,
            centre: SkyCoord,
            a_world: units.Quantity,
            b_world: units.Quantity,
            theta_world: units.Quantity,
            kron_radius: float = 1.,
            ext: int = 0,
            output: str = None,
            mask_nearby=True,
            detection_threshold: float = None
    ):

        if detection_threshold is None:
            detection_threshold = self.detection_threshold()

        u.debug_print(2, f"sep_elliptical_magnitude(): mask_nearby == {mask_nearby}")

        flux, flux_err, flags, back = self.sep_elliptical_photometry(
            centre=centre,
            a_world=a_world,
            b_world=b_world,
            theta_world=theta_world,
            kron_radius=kron_radius,
            ext=ext,
            output=output,
            mask_nearby=mask_nearby,
            subtract_background=self.do_subtract_background()
        )

        if flux is None:
            return None

        snr = flux / flux_err
        mag, mag_err, _, _ = self.magnitude(
            flux, flux_err
        )
        for i, m in enumerate(mag):
            if snr[i] < detection_threshold or np.isnan(m):
                mag_lim, _, _, _ = self.magnitude(
                    detection_threshold * flux_err[i]
                )

                if m > mag_lim or np.isnan(m):
                    m = mag_lim
                if np.isnan(m):
                    m = -999. * units.mag
                mag_err[i] = -999. * units.mag
                mag[i] = m

        return {
            "mag": mag,
            "mag_err": mag_err,
            "snr": snr,
            "back": back,
            "flux": flux,
            "flux_err": flux_err,
            "threshold": detection_threshold
        }

    def make_galfit_version(
            self,
            output_path: str = None,
            ext: int = 0
    ):
        """
        Generate a version of this file for use with GALFIT.
        Modifies header item GAIN to conform to GALFIT's expectations (outlined in the GALFIT User Manual,
        http://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html)
        :param output_path: path to write modified file to.
        :param ext: FITS extension to modify header of.
        :return:
        """
        if output_path is None:
            output_path = self.path.replace(".fits", "_galfit.fits")
        new = self.copy(output_path)
        new.load_headers()
        new.set_header_items(
            {
                "GAIN": self.extract_header_item(key="OLD_EXPTIME", ext=ext) *
                        self.extract_header_item(key="OLD_GAIN", ext=ext)
            }
        )
        new.write_fits_file()
        return new

    def make_galfit_psf(
            self,
            output_dir: str,
            x: float,
            y: float
    ):
        # We obtain an oversampled PSF, because GALFIT works best with one.
        psfex_path = os.path.join(output_dir, f"{self.name}_galfit_psfex.psf")
        if not os.path.isfile(psfex_path):
            self.psfex(
                output_dir=output_dir,
                PSF_SAMPLING=0.5,  # Equivalent to GALFIT fine-sampling factor = 2
                # PSF_SIZE=50,
                force=True,
                set_attributes=True
            )
        else:
            self.psfex_path = psfex_path
            self.load_psfex_output()
        # Load oversampled PSF image
        psf_img = self.psf_image(x=x, y=y, match_pixel_scale=False)[0]
        psf_img /= np.max(psf_img)
        # Write our PSF image to disk for GALFIT to find
        psf_hdu = fits.hdu.PrimaryHDU(psf_img)
        psf_hdu_list = fits.hdu.HDUList(psf_hdu)
        psf_path = os.path.join(output_dir, f"{self.name}_psf.fits")
        psf_hdu_list.writeto(
            psf_path,
            overwrite=True
        )
        return psf_path

    def make_galfit_feedme(
            self,
            feedme_path: str,
            img_block_path: str,
            psf_file: str = None,
            psf_fine_sampling: int = 2,
            mask_file: str = None,
            fitting_region_margins: tuple = None,
            convolution_size: tuple = None,
            models: List[dict] = None
    ):
        if fitting_region_margins is None:
            self.load_data()
            max_x, max_y = self.data[0].shape
            fitting_region_margins = 0, max_x - 1, 0, max_y - 1
        if convolution_size is None:
            left, right, bottom, top = fitting_region_margins
            convolution_size = int(right - left), int(top - bottom)

        self.extract_pixel_scale()
        dx = (1 * units.pixel).to(units.arcsec, self.pixel_scale_x).value
        dy = (1 * units.pixel).to(units.arcsec, self.pixel_scale_y).value

        galfit.galfit_feedme(
            feedme_path=feedme_path,
            input_file=self.filename,
            output_file=img_block_path,
            zeropoint=self.zeropoint_best["zeropoint_img"].value,
            psf_file=psf_file,
            psf_fine_sampling=psf_fine_sampling,
            mask_file=mask_file,
            fitting_region_margins=fitting_region_margins,
            convolution_size=convolution_size,
            plate_scale=(dx, dy),
            models=models
        )

    def galfit(
            self,
            output_dir: str = None,
            output_prefix=None,
            frame_lower: int = 30,
            frame_upper: int = 100,
            ext: int = 0,
            model_guesses: Union[dict, List[dict]] = None,
            psf_path: str = None,
            use_frb_galfit: bool = False
    ):
        """

        :param coords:
        :param output_dir:
        :param frame_lower:
        :param frame_upper:
        :param ext:
        :param model_guesses: dict, with:
            object_type: str
            position: Either "position" can be provided as a SkyCoord object, or x & y as pixel coordinates.

        :param use_frb_galfit: Use the FRB repo frb.galaxies.galfit module. Single-sersic only; if multiple models are provided only one will be used.
        :return:
        """
        if output_prefix is None:
            output_prefix = self.name
        if model_guesses is None:
            model_guesses = [{
                "object_type": "sersic",
                "int_mag": 20.0,
                "position": self.epoch.field.objects[0].position
            }]

        if isinstance(model_guesses, dict):
            model_guesses = [model_guesses]
        gf_tbls = {}
        for i, model in enumerate(model_guesses):
            if "position" in model:
                x, y = self.world_to_pixel(
                    coord=model["position"],
                    origin=1
                )
                model_guesses[i]["x"] = x
                model_guesses[i]["y"] = y
            elif "x" in model and "y" in model:
                model_guesses["position"] = self.pixel_to_world(
                    x=model["x"],
                    y=model["y"],
                    origin=1
                )
            else:
                raise ValueError("All model dicts must have either 'position' or 'x' & 'y' keys.")
            gf_tbls[f"COMP_{i + 1}"] = []
        gf_tbls[f"COMP_{i + 2}"] = []

        if output_dir is None:
            output_dir = self.data_path
        self.load_output_file()
        new = self.make_galfit_version(
            output_path=os.path.join(output_dir, f"{output_prefix}_galfit.fits")
        )
        new.zeropoint_best = self.zeropoint_best
        new.open()

        x = model_guesses[0]["x"]
        y = model_guesses[0]["y"]
        if psf_path is None:
            psf_path = new.make_galfit_psf(
                x=x,
                y=y,
                output_dir=output_dir
            )
        # Turn the first model into something the frb repo can use, and hope it's a sersic
        if use_frb_galfit:
            model_dict = model_guesses[0].copy()
            x = int(model_dict.pop("x"))
            y = int(model_dict.pop("y"))
            model_dict["position"] = (x, y)
            model_dict.pop("object_type")

        psf_file = os.path.split(psf_path)[-1]
        psf_path_moved = os.path.join(output_dir, psf_file)
        if not os.path.isfile(psf_path_moved):
            shutil.copy(psf_path, psf_path_moved)
        psf_path = psf_path_moved

        new.load_data()
        data = new.data[ext].copy()
        new.close()

        mask_file = f"{output_prefix}_mask.fits"
        mask_path = os.path.join(output_dir, mask_file)
        margins_max = u.frame_from_centre(frame_upper + 1, x, y, data)
        mask = new.write_mask(
            output_path=mask_path,
            unmasked=list(map(lambda m: m["position"], model_guesses)),
            ext=ext,
            method="sep",
            obj_value=1,
            back_value=0,
            margins=margins_max
        )

        self.extract_pixel_scale(ext)

        for frame in range(frame_lower, frame_upper + 1):
            margins = u.frame_from_centre(frame, x, y, data)
            print("Generating mask...")
            data_trim = u.trim_image(data, margins=margins)
            mask_data = u.trim_image(mask.data[ext], margins=margins).value
            feedme_file = f"{output_prefix}_{frame}.feedme"
            feedme_path = os.path.join(output_dir, feedme_file)
            img_block_file = f"{output_prefix}_galfit-out_{frame}.fits"
            img_block_path = os.path.join(output_dir, img_block_file)
            if not use_frb_galfit:
                new.make_galfit_feedme(
                    feedme_path=feedme_path,
                    img_block_path=img_block_file,
                    psf_file=psf_file,
                    psf_fine_sampling=2,
                    mask_file=mask_file,
                    fitting_region_margins=margins,
                    convolution_size=(frame * 2, frame * 2),
                    models=model_guesses
                )
                galfit.galfit(
                    config=feedme_file,
                    output_dir=output_dir
                )
            else:
                import frb.galaxies.galfit as galfit_frb
                galfit_frb.run(
                    imgfile=new.path,
                    psffile=psf_path,
                    outdir=output_dir,
                    configfile=feedme_file,
                    outfile=img_block_path,
                    finesample=2,
                    badpix=mask_path,
                    region=margins,
                    convobox=(frame * 2, frame * 2),
                    zeropoint=self.zeropoint_best["zeropoint_img"].value,
                    skip_sky=False,
                    **model_dict
                )
            shutil.copy(os.path.join(output_dir, "fit.log"),
                        os.path.join(output_dir, f"{output_prefix}_{frame}_fit.log"))

            try:
                img_block = fits.open(img_block_path)
            except FileNotFoundError:
                return None

            results_header = img_block[2].header
            components = galfit.extract_fit_params(results_header)
            for compname in components:
                component = components[compname]
                pos = self.pixel_to_world(component["x"], component["y"])
                component["ra"] = pos.ra
                component["dec"] = pos.dec
                if "r_eff" in component:
                    component["r_eff_ang"] = component["r_eff"].to(units.arcsec, self.pixel_scale_x)
                    component["r_eff_ang_err"] = component["r_eff_err"].to(units.arcsec, self.pixel_scale_x)
                # TODO: The below assumes RA and Dec are along x & y (neglecting image rotation), which isn't great
                component["ra_err"] = component["x_err"].to(units.deg, self.pixel_scale_x)
                component["dec_err"] = component["y_err"].to(units.deg, self.pixel_scale_y)
                component["frame"] = frame
                results_table = table.QTable([component])
                gf_tbls[compname].append(results_table)

            mask_ones = np.invert(mask_data.astype(bool)).astype(int)

            # Masked data
            img_block.insert(4, img_block[1].copy())
            img_block[4].data *= mask_ones  # + #
            img_block[4].data += mask_data * np.median(img_block[1].data)

            # Masked, subtracted data
            img_block.insert(5, img_block[3].copy())
            img_block[5].data *= mask_ones  # + #
            img_block[5].data += mask_data * np.median(img_block[3].data)

            for idx in [2, 3]:
                img_block[idx].header.insert('OBJECT', ('PCOUNT', 0))
                img_block[idx].header.insert('OBJECT', ('GCOUNT', 1))

            img_block.writeto(img_block_path, overwrite=True)

        component_tables = {}
        for compname in gf_tbls:
            gf_tbl = table.vstack(gf_tbls[compname])
            component_tables[compname] = gf_tbl

        shutil.copy(p.path_to_config_galfit(), output_dir)

        return component_tables

    def galfit_object(
            self,
            obj: objects.Galaxy,
            pivot_component: int = 2,
            **kwargs
    ):

        photometry, _ = obj.select_photometry(
            fil=self.filter_name,
            instrument=self.instrument_name,
        )

        if "model_guesses" in kwargs:
            model_guesses = kwargs["model_guesses"]
        else:
            model_guesses = [{
                "object_type": "sersic"
            }]

        for model in model_guesses:
            model["position"] = obj.position
            model["int_mag"] = photometry["mag"].value

        kwargs["model_guesses"] = model_guesses

        model_tbls = self.galfit(
            **kwargs
        )

        best_params = galfit.sersic_best_row(model_tbls[f"COMP_{pivot_component}"])
        best_params["r_eff_proj"] = obj.projected_distance(best_params["r_eff_ang"]).to("kpc")
        best_params["r_eff_proj_err"] = obj.projected_distance(best_params["r_eff_ang_err"]).to("kpc")
        return best_params

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
        elif instrument == "decam":
            return DESCutout
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
            "airmass_err": "AIRMASS_ERR",
            "astrometry_err": "ASTM_RMS",
            "ra_err": "RA_RMS",
            "dec_err": "DEC_RMS"
        })
        return header_keys

    @classmethod
    def count_exposures(cls, image_paths: list):
        return len(image_paths)

    def rank_photometric_cat(self, cats: list):
        """
        Gives the ranking of photometric catalogues available for calibration, ranked by similarity to filter set.
        :return:
        """

        self.instrument.gather_filters()
        self._filter_from_name()

        differences = {}

        for cat in cats:
            if cat in cat_instruments:
                other_instrument_name = cat_instruments[cat]
                other_instrument = inst.Instrument.from_params(other_instrument_name)
                other_instrument.gather_filters()
                if self.filter.band_name in other_instrument.bands:
                    other_filter = other_instrument.bands[self.filter.band_name]
                    differences[cat] = self.filter.compare_wavelength_range(
                        other=other_filter
                    )
            elif cat == "instrument_archive":
                differences[cat] = 0 * units.angstrom
            elif cat == "calib_pipeline":
                differences[cat] = 0.1 * units.angstrom

        differences = dict(sorted(differences.items(), key=lambda x: x[1]))
        return list(differences.keys()), list(differences.values())


class CoaddedImage(ImagingImage):
    def __init__(
            self,
            path: str,
            frame_type: str = None,
            instrument_name: str = None,
    ):
        super().__init__(
            path=path,
            frame_type=frame_type,
            instrument_name=instrument_name,
            load_outputs=False
        )

        self.area_file = None

        self.load_output_file()

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
        if output_path is None:
            output_path = trimmed.path
        new_area_path = output_path.replace(".fits", "_area.fits")

        if os.path.isfile(self.area_file):
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

    def register(
            self,
            target: 'ImagingImage',
            output_path: str,
            ext: int = 0,
            trim: bool = True,
            **kwargs
    ):
        new_img = super().register(
            target=target,
            output_path=output_path,
            ext=ext,
            trim=trim,
            **kwargs
        )
        import reproject as rp
        area = new_img.copy(new_img.path.replace(".fits", "_area.fits"))
        area.load_data()
        reprojected, footprint = rp.reproject_exact(self.area_file, new_img.headers[ext], parallel=True)
        area.data[ext] = reprojected * area.data[ext].unit
        area.area_file = None
        area.write_fits_file()

        new_img.area_file = area.path
        return new_img

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
        if new_image is not None:
            new_image.area_file = self.area_file
            new_image.update_output_file()
        return new_image

    def copy(self, destination: str):
        new_image = super().copy(destination)
        new_image.area_file = self.area_file
        new_image.update_output_file()
        return new_image

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "ncombine": "NCOMBINE"
        })
        return header_keys

    def extract_ncombine(self):
        key = self.header_keys()["ncombine"]
        return self.extract_header_item(key)

    @classmethod
    def select_child_class(cls, instrument: str, **kwargs):
        if not isinstance(instrument, str):
            instrument = str(instrument)
        if instrument is None:
            return CoaddedImage
        elif instrument == "vlt-fors2":
            return FORS2CoaddedImage
        elif instrument == "vlt-hawki":
            return HAWKICoaddedImage
        elif instrument == "panstarrs1":
            return PanSTARRS1Cutout
        elif "hst" in instrument:
            return HubbleImage
        elif instrument == "decam":
            return DESCutout
        else:
            raise ValueError(f"Unrecognised instrument {instrument}")


class SurveyCutout(CoaddedImage):
    def do_subtract_background(self):
        return False


class DESCutout(SurveyCutout):
    instrument_name = "decam"

    def zeropoint(
            self,
            **kwargs
    ):
        self.add_zeropoint(
            catalogue="calib_pipeline",
            zeropoint=self.extract_header_item("MAGZERO"),  # - 2.5 * np.log10(exptime)) * units.mag,
            zeropoint_err=0.0 * units.mag,
            extinction=0.0 * units.mag,
            extinction_err=0.0 * units.mag,
            airmass=0.0,
            airmass_err=0.0
        )
        zp = super().zeropoint(
            **kwargs
        )

        return zp

    def extract_unit(self, astropy: bool = False):
        unit = "ct / s"
        if astropy:
            unit = units.ct / units.s
        return unit

    def extract_exposure_time(self):
        self.exposure_time = 1. * units.second
        return self.exposure_time

    def extract_noise_read(self):
        self.noise_read = 0. * units.electron / units.pixel
        return self.noise_read

    def extract_integration_time(self):
        return self.extract_header_item("EXPTIME") * units.second

    def extract_filter(self):
        key = self.header_keys()["filter"]
        fil_string = self.extract_header_item(key)
        self.filter_name = fil_string[:fil_string.find(" ")]
        self.filter_short = self.filter_name

        self._filter_from_name()

        return self.filter_name

    def extract_ncombine(self):
        return 1


class PanSTARRS1Cutout(SurveyCutout):
    instrument_name = "panstarrs1"

    def __init__(self, path: str, **kwargs):
        super().__init__(path=path)
        # self.instrument_name = "panstarrs1"
        self.extract_filter()
        self.exposure_time = None
        self.extract_exposure_time()

    def mask_nearby(self):
        return True

    def detection_threshold(self):
        return 10.

    def extract_filter(self):
        key = self.header_keys()["filter"]
        fil_string = self.extract_header_item(key)
        self.filter_name = fil_string[:fil_string.find(".")]
        self.filter_short = self.filter_name

        self._filter_from_name()

        return self.filter_name

    def extract_integration_time(self):
        return self.extract_exposure_time()

    def zeropoint(
            self,
            **kwargs
    ):
        """
        According to the reference below, the PS1 cutouts are scaled to zeropoint 25.
        https://outerspace.stsci.edu/display/PANSTARRS/PS1+Stack+images
        :return:
        """

        self.load_headers()

        self.add_zeropoint(
            catalogue="calib_pipeline",
            zeropoint=self.extract_header_item("FPA.ZP"),
            zeropoint_err=0.0 * units.mag,
            extinction=0.0 * units.mag,
            extinction_err=0.0 * units.mag,
            airmass=0.0,
            airmass_err=0.0
        )

        zp = super().zeropoint(
            **kwargs
        )
        return zp

    # self.select_zeropoint(True)

    # I only wrote this function below because I couldn't find the EXPTIME key in the PS1 cutouts. It is, however, there.
    # def extract_exposure_time(self):
    #     # self.load_headers()
    #     # exp_time_keys = filter(lambda k: k.startswith("EXP_"), self.headers[0])
    #     # exp_time = 0.
    #     # exp_times = []
    #     # for key in exp_time_keys:
    #     #     exp_time += self.headers[0][key]
    #     # #    exp_times.append(self.headers[0][key])
    #     #
    #     # self.exposure_time = exp_time * units.second  # np.mean(exp_times)
    #     self.exposure_time = 1.0 * units.second
    #     return self.exposure_time

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "noise_read": "HIERARCH CELL.READNOISE",
            "filter": "HIERARCH FPA.FILTERID",
            "gain": "HIERARCH CELL.GAIN",
            "ncombine": "NINPUTS"
        })
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
    num_chips = 4
    instrument_name = "vlt-hawki"

    def zeropoint(
            self,
            **kwargs
    ):
        return self.add_zeropoint(
            catalogue="2MASS",
            zeropoint=self.extract_header_item("PHOTZP"),
            zeropoint_err=self.extract_header_item("PHOTZPER"),
            extinction=0.0 * units.mag,
            extinction_err=0.0 * units.mag,
            airmass=0.0,
            airmass_err=0.0
        )
        # self.select_zeropoint(True)
        # return self.zeropoint_best

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "gain": "GAIN",
            "filter": "HIERARCH ESO INS FILT1 NAME"
        })
        return header_keys


class FORS2Image(ESOImagingImage):
    instrument_name = "vlt-fors2"
    num_chips = 2

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
    def header_keys(cls) -> dict:
        header_keys = super().header_keys()
        header_keys.update(ESOImage.header_keys())
        header_keys.update({
            "noise_read": "HIERARCH ESO DET OUT1 RON",
            "gain": "HIERARCH ESO DET OUT1 GAIN",
            "program_id": "HIERARCH ESO OBS PROG ID",
            "filter": "HIERARCH ESO INS FILT1 NAME"
        })
        return header_keys


class FORS2CoaddedImage(CoaddedImage):
    instrument_name = "vlt-fors2"

    def __init__(
            self,
            path: str,
            frame_type: str = None,
            **kwargs
    ):
        super().__init__(
            path=path,
            frame_type=frame_type,
            instrument_name=self.instrument_name,
        )

    def zeropoint(
            self,
            **kwargs
    ):
        if self.filter.calib_retrievable():
            zp = self.calibration_from_qc1()
        else:
            zp = super().zeropoint(
                **kwargs
            )
        return zp

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

            zp = self.add_zeropoint(
                zeropoint=row["zeropoint"],
                zeropoint_err=row["zeropoint_err"],
                airmass=self.extract_airmass(),
                airmass_err=airmass_err,
                extinction=row["extinction"],
                extinction_err=row["extinction_err"],
                mjd_measured=row["mjd_obs"],
                delta_t=row["mjd_obs"] - self.mjd_obs,
                n_matches=None,
                catalogue="instrument_archive"
            )

            self.extinction_atmospheric = row["extinction"]
            self.extinction_atmospheric_err = row["extinction_err"]

            self.add_log(
                action=f"Retrieved calibration values from ESO QC1 archive.",
                method=self.calibration_from_qc1,
            )
            self.update_output_file()

            return zp
        else:
            return None


class GSAOIImage(CoaddedImage):
    instrument_name = "gs-aoi"

    def extract_pixel_scale(self, ext: int = 1, force: bool = False):
        return super().extract_pixel_scale(ext=ext, force=force)

    def extract_pointing(self):
        # GSAOI images keep the WCS information in the second HDU header.
        key = self.header_keys()["ra"]
        ra = self.extract_header_item(key, 1)
        key = self.header_keys()["dec"]
        dec = self.extract_header_item(key, 1)
        self.pointing = SkyCoord(ra, dec, unit=units.deg)
        return self.pointing


class HubbleImage(CoaddedImage):
    instrument_name = "hst-dummy"

    def __init__(
            self,
            path: str,
            frame_type: str = None,
            instrument_name: str = None
    ):
        if instrument_name is None:
            instrument_name = detect_instrument(path)
        super().__init__(
            path=path,
            frame_type=frame_type,
            instrument_name=instrument_name,
        )

    def extract_exposure_time(self):
        self.exposure_time = 1.0 * units.second
        return self.exposure_time

    def extract_noise_read(self):
        self.noise_read = 0.0 * units.electron / units.pixel
        return self.noise_read

    def zeropoint(self, **kwargs):
        """
        Returns the AB magnitude zeropoint of the image, according to
        https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
        :return:
        """
        photflam = self.extract_header_item("PHOTFLAM")
        photplam = self.extract_header_item("PHOTPLAM")

        zeropoint = (-2.5 * math.log10(photflam) - 5 * math.log10(photplam) - 2.408) * units.mag

        self.add_zeropoint(
            catalogue="calib_pipeline",
            zeropoint=zeropoint,
            zeropoint_err=0.0 * units.mag,
            extinction=0.0 * units.mag,
            extinction_err=0.0 * units.mag,
            airmass=0.0,
            airmass_err=0.0,
            image_name="self"
        )
        self.select_zeropoint(True)
        self.update_output_file()
        self.add_log(
            action=f"Calculated zeropoint {zeropoint} from PHOTFLAM and PHOTPLAM header keys.",
            method=self.trim,
        )
        self.update_output_file()
        return self.zeropoint_best

    def mask_nearby(self):
        if self.instrument_name == "hst-wfc3_uvis2":
            mask = False
        else:
            mask = True
        u.debug_print(2, "mask_nearby", self.instrument_name, mask, type(self))
        return mask

    def detection_threshold(self):
        if self.instrument_name == "hst-wfc3_uvis2":
            thresh = 5.
        else:
            thresh = 5.
        return thresh

    def do_subtract_background(self):
        return False

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
            path=os.path.join(pypeit_raw_path, attributes[1]),
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
        img_1: ImagingImage,
        img_2: ImagingImage,
        sigma: int = 3,
        depth_type: str = "secure",
        snr_type: str = "SNR_PSF"
):
    if img_1.depth[depth_type][snr_type][f"{sigma}-sigma"] > \
            img_2.depth[depth_type][snr_type][f"{sigma}-sigma"]:
        return img_1
    else:
        return img_2

# def pypeit_str(self):
#     header = self.hdu[0].header
#     string = f"| {self.filename} | "
