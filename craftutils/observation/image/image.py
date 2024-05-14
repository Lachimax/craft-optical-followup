# Code by Lachlan Marnoch, 2021
import copy
import math
import os
import shutil
import string
from typing import Union, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
import astropy.table as table
import astropy.units as units
from astropy.coordinates import SkyCoord
from astropy.time import Time

from astropy.visualization import quantity_support

try:
    import photutils
except ModuleNotFoundError:
    print("photutils not installed; some photometry-related functionality will be unavailable.")

try:
    import sep
except ModuleNotFoundError:
    print("sep not installed; some photometry-related functionality will be unavailable.")

import craftutils.utils as u
import craftutils.params as p

import craftutils.observation.log as log
import craftutils.observation.instrument as inst

# __all__ = []

quantity_support()

# This contains the names as in the header as keys and the names as used in this project as values.
instrument_header = {
    "FORS2": "vlt-fors2",
    "HAWKI": "vlt-hawki",
    "PS1": "panstarrs1"
}

active_images = {}

gain_unit = units.electron / units.ct
noise_read_unit = units.electron / units.pixel


# TODO: Make this list all fits files, then write wrapper that eliminates non-science images and use that in scripts.
# @u.export
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

    from .__init__ import ImagingImage

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


# @u.export
def fits_table_all(
        input_path: str,
        output_path: str = "",
        science_only: bool = True
):
    """
    Produces and writes to disk a table of .fits files in the given path, with the vital statistics of each. Intended
    only for use with raw ESO data.
    :param input_path:
    :param output_path:
    :param science_only: If True, we are writing a list for a folder that also contains calibration files, which we want
     to ignore.
    :return:
    """

    from .imaging import ImagingImage

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
        cls = ImagingImage.select_child_class(instrument_name=instrument)
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


# @u.export
def detect_instrument(path: str, ext: int = 0, fail_quietly: bool = False):
    if not os.path.exists(path) or not os.path.isfile(path):
        raise FileNotFoundError(f"No image found at {path}")
    elif path.endswith("_outputs.yaml"):
        path = path.replace("_outputs.yaml", ".fits")

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


# @u.export
def from_path(path: str, cls: type = None, **kwargs):
    """To be used when there may already be an image instance for this path floating around in memory, and it's okay
    (or better) to access this one instead of creating a new instance.
    When the image may have overwritten a previous file, instantiating the image directly is better.

    :param path:
    :param cls:
    :param kwargs:
    :return:
    """
    path = p.join_data_dir(path)
    u.debug_print(3, "image.from_path(): path ==", path)
    if path in active_images:
        return active_images[path]
    if cls is not None:
        return cls(path, **kwargs)


# @u.export
def expunge():
    image_list = list(active_images.keys())
    for img_path in image_list:
        del active_images[img_path]


# @u.export
class Image:
    instrument_name = "dummy"
    num_chips = 1
    class_dict = {}

    def __init__(
            self,
            path: str,
            frame_type: str = None,
            instrument_name: str = None,
            logg: log.Log = None,
            **kwargs
    ):
        ignore_missing_path = False
        if "ignore_missing_path" in kwargs:
            ignore_missing_path = kwargs["ignore_missing_path"]
        if not ignore_missing_path and not os.path.isfile(path):
            raise FileNotFoundError(f"The image file {path} does not exist.")
        active_images[path] = self

        if path.endswith("_outputs.yaml"):
            self.output_file = path
            self.path = path.replace("_outputs.yaml", ".fits")
        elif path.endswith(".fits"):
            self.path = path
            self.output_file = path.replace(".fits", "_outputs.yaml")
        elif path.endswith(".fits.fz"):
            self.path = path
            self.output_file = path.replace(".fits.fz", "_outputs.yaml")
        else:
            self.path = path
            self.output_file = path.replace(
                os.path.splitext(path)[-1],
                "_outputs.yaml"
            )

        # Attempt opening the fits file to test whether it's valid; let astropy handle the error.
        test = fits.open(self.path, "readonly")
        test.close()

        self.data_path, self.filename = os.path.split(self.path)
        self.name = self.get_id()
        self.hdu_list = None
        self.frame_type = frame_type
        self.headers = None
        self.data = []
        self.date = None
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
            self.instrument_name
        )
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

        self.derived_from = None

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
            ancestors: List['Image'] = None,
            **method_kwargs,
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
            ancestor_logs=ancestors,
            method_args=method_kwargs
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

    def copy(self, destination: str, suffix: str = ""):
        """
        A note to future me: do copy, THEN make changes, or be prepared to suffer the consequences.

        :param destination:
        :return:
        """
        u.debug_print(1, "Copying", self.path, "to", destination)
        if os.path.isdir(destination):
            filename, ext = os.path.splitext(self.filename)
            if not suffix.startswith("_"):
                suffix = "_" + suffix
            filename = filename + suffix + ext
            destination = os.path.join(destination, filename)
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
        if force or not self.data:
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
        from ccdproc import CCDData
        if unit is None:
            return CCDData.read(self.path)
        else:
            return CCDData.read(self.path, unit=unit)

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
        if value is None and not accept_absent:
            for ext in range(len(self.headers)):
                value = self._extract_header_item(key=key, ext=ext)
                if value is not None:
                    return value
            # Then, if we get to the end of the loop, the item clearly doesn't exist.
            return None
        else:
            return value

    def extract_chip_number(self, ext: int = 0):
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
            self.gain = self.extract_header_item(key) * gain_unit
        if self.gain is not None:
            self.gain = u.check_quantity(self.gain, gain_unit)
        return self.gain

    def extract_date_obs(self):
        key = self.header_keys()["date-obs"]
        self.date_obs = self.extract_header_item(key)
        key = self.header_keys()["mjd-obs"]
        self.mjd_obs = self.extract_header_item(key)
        if self.date_obs is None and self.mjd_obs is not None:
            self.date = Time(self.mjd_obs, format="mjd")
            self.date_obs = self.date.strftime("%Y-%m-%dT%H:%M:%S")
        elif self.mjd_obs is None and self.date_obs is not None:
            self.date = Time(self.date_obs)
            self.mjd_obs = self.date.mjd
        else:
            self.date = Time(self.date_obs)
        return self.date_obs

    def extract_exposure_time(self):
        key = self.header_keys()["exposure_time"]
        self.exposure_time = self.extract_header_item(key) * units.second
        return self.exposure_time

    def extract_noise_read(self):
        key = self.header_keys()["noise_read"]
        noise = self.extract_header_item(key)
        if noise is not None:
            self.noise_read = self.extract_header_item(key) * noise_read_unit
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

    def extract_saturate(self, data_ext: int = 0):
        key = self.header_keys()["saturate"]
        saturate = self.extract_header_item(key)
        if saturate is None:
            saturate = 65535
        unit = self.extract_unit(astropy=True)
        self.saturate = saturate * unit
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
            if len(self.data) > i and self.data[i] is not None:
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

        self.hdu_list.writeto(self.path, overwrite=True, output_verify="fix")

        self.close()

    @classmethod
    def header_keys(cls):
        header_keys = {
            "integration_time": "INTTIME",
            "exposure_time": "EXPTIME",
            "exposure_time_old": "HIERARCH OLD_EXPTIME",
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
            "saturate_old": "HIERARCH OLD_SATURATE",
            "program_id": "PROG_ID"
        }
        return header_keys

    @classmethod
    def from_fits(cls, path: str, mode: str = "imaging"):
        from .imaging import ImagingImage
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
            elif "FPA.TELESCOPE" in header:
                instrument = header["FPA.TELESCOPE"]
            i += 1

        if instrument is None:
            print("Instrument could not be determined from header.")
            child = ImagingImage
        else:
            # Look for standard instrument name in list
            if instrument in instrument_header:
                instrument = instrument_header[instrument]
                child = cls.select_child_class(instrument_name=instrument, mode=mode)
            else:
                child = ImagingImage
        u.debug_print(2, "Image.from_fits(): instrument ==", instrument)
        img = child(path=path, instrument_name=instrument)
        img.instrument_name = instrument
        return img

    @classmethod
    def select_child_class(cls, instrument_name: str, **kwargs):
        instrument_name = instrument_name.lower()
        if 'mode' in kwargs:
            mode = kwargs['mode']
            if mode == 'imaging':
                from .imaging import ImagingImage
                return ImagingImage.select_child_class(instrument_name=instrument_name, **kwargs)
            elif mode == 'spectroscopy':
                from .__init__ import Spectrum
                return Spectrum.select_child_class(instrument_name=instrument_name, **kwargs)
            else:
                raise ValueError(f"Unrecognised mode {mode}")
        else:
            raise KeyError(f"mode must be provided for {cls}.select_child_class()")

    def split_fits(self, output_dir: str = None):
        if output_dir is None:
            output_dir = self.data_path
        self.open()
        new_files = {}

        if self.hdu_list[0].data is None:
            update_header = self.hdu_list[0].header
        else:
            update_header = {}

        for hdu in self.hdu_list:
            if hdu.data is None or isinstance(hdu, fits.BinTableHDU):
                continue
            new_hdu_list = fits.HDUList(fits.PrimaryHDU(hdu.data, hdu.header))
            new_hdu_list[0].header.update(update_header)
            new_path = os.path.join(output_dir, self.filename.replace(".fits", f"_{hdu.name}.fits"))
            new_hdu_list.writeto(
                new_path,
                overwrite=True
            )
            new_img = self.__class__(new_path)
            new_files[hdu.name] = new_img
        self.close()
        return new_files


