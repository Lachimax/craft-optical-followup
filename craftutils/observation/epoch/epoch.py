import os
import shutil

import datetime
from typing import Union, List, Dict

import matplotlib.pyplot as plt
import numpy as np

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as units
import astropy.table as table
import astropy.io.fits as fits
from astropy.modeling import models, fitting

import ccdproc

# from .spectroscopy import *

import craftutils.astrometry as astm
import craftutils.fits_files as ff
import craftutils.observation as obs
import craftutils.observation.field as fld
import craftutils.observation.filters.eso.vlt_fors2
import craftutils.observation.objects as objects
import craftutils.observation.image as image
import craftutils.observation.instrument as inst
import craftutils.observation.filters as filters
import craftutils.observation.log as log
import craftutils.observation.survey as survey
import craftutils.params as p
import craftutils.plotting as pl
import craftutils.retrieve as retrieve
import craftutils.utils as u
import craftutils.wrap.montage as montage
import craftutils.wrap.dragons as dragons

config = p.config

active_epochs = {}

if p.data_dir:
    zeropoint_yaml = os.path.join(p.data_dir, f"zeropoints.yaml")


def _epoch_directory_path():
    path = os.path.join(p.param_dir, "fields", "directory.yaml")
    u.debug_print(2, "_epoch_directory_path(): path ==", path)
    return path


def load_epoch_directory():
    """
    Loads the epoch directory yaml file from the param directory.
    :return: epoch directory as dict.
    """
    path = _epoch_directory_path()
    directory = p.load_params(path)
    if directory is None:
        directory = {}
        if not os.path.isfile(path):
            write_epoch_directory(directory=directory)
    return directory


def epoch_from_directory(epoch_name: str, quiet: bool = False):
    from .spectroscopy import SpectroscopyEpoch
    directory = load_epoch_directory()
    if not quiet:
        print(f"Looking for {epoch_name} in directory...")
    epoch = None
    if epoch_name in directory:
        epoch_dict = directory[epoch_name]
        field_name = epoch_dict["field_name"]
        instrument = epoch_dict["instrument"]
        mode = epoch_dict["mode"]
        field = fld.Field.from_params(name=field_name)
        if mode == "imaging":
            epoch = ImagingEpoch.from_params(
                epoch_name,
                instrument=instrument,
                field=field,
                quiet=quiet
            )
        elif mode == "spectroscopy":
            epoch = SpectroscopyEpoch.from_params(
                name=epoch_name,
                field=field,
                instrument=instrument,
                quiet=quiet
            )
        return epoch


def write_epoch_directory(directory: dict):
    """
    Writes the passed dict to the directory.yaml file in param directory.
    :param directory: The updated directory as a dict.
    :return:
    """
    p.save_params(_epoch_directory_path(), directory)


def add_to_epoch_directory(
        field_name: str,
        instrument: str,
        mode: str,
        epoch_name: str,
        **other
):
    """
    Adds a single epoch to the epoch directory.
    :param field_name:
    :param instrument:
    :param mode:
    :param epoch_name:
    :return:
    """
    directory = load_epoch_directory()
    directory[epoch_name] = {
        "field_name": field_name,
        "instrument": instrument,
        "mode": mode,
        "epoch_name": epoch_name,
    }
    directory[epoch_name].update(other)
    write_epoch_directory(directory=directory)


def add_many_to_epoch_directory(
        epochs,
        field_name: str = None,
        instrument: str = None,
        mode: str = None
):
    """

    :param epochs: A nested dictionary, with keys being the epoch names; the value dictionaries should then have:
        field_name or field, instrument, mode. Any of these values can be
        overridden by the other arguments.
    :return:
    """
    directory = load_epoch_directory()
    u.debug_print(2, "add_many_to_epoch_directory(): directory ==", directory)
    for epoch_name in epochs:
        epoch_dict = epochs[epoch_name]
        # if "epoch_name" in epoch_dict:
        #     epoch_name = epoch_dict["epoch_name"]
        # elif "epoch" in epoch_dict:
        #     epoch_name = epoch_dict["epoch"]
        # elif "name" in epoch_dict:
        #     epoch_name = epoch_dict["name"]
        # else:
        #     u.debug_print(2, "add_many_to_epoch_directory(): epoch_dict ==", epoch_dict)
        #     raise ValueError("No epoch_name given or found in current dict.")

        if field_name is None:
            if "field_name" in epoch_dict:
                field_name = epoch_dict["field_name"]
            elif "field" in epoch_dict:
                field_name = epoch_dict["field"]
            else:
                raise ValueError("No field_name given or found in current dict.")

        if instrument is None:
            instrument = epoch_dict["instrument"]

        u.debug_print(2, "add_many_to_epoch_directory(): mode ==", mode)
        if mode is None:
            mode = epoch_dict["mode"]

        directory[epoch_name] = {
            "field_name": field_name,
            "instrument": epoch_dict["instrument"],
            "mode": mode,
            "epoch_name": epoch_name,
        }
        write_epoch_directory(directory)


def _check_do_list(
        do: Union[list, str],
        stages
):
    if isinstance(do, str):
        try:
            do = [int(do)]
        except ValueError:
            if " " in do:
                char = " "
            elif "," in do:
                char = ","
            else:
                raise ValueError("do string is not correctly formatted.")
            do = list(map(int, do.split(char)))

    if isinstance(do, list):
        do_nu = []
        for n in do:
            if isinstance(n, int):
                do_nu.append(stages[n])
            elif isinstance(n, str):
                if n in stages:
                    do_nu.append(n)
        do = do_nu

    return do


def expunge_epochs():
    for epoch_name in active_epochs:
        del active_epochs[epoch_name]


def _output_img_list(lst: list):
    """
    Turns a list of images into a YAML-able list of paths.
    :param lst:
    :return:
    """
    out_list = []
    for img in lst:
        out_list.append(img.output_file)
    out_list.sort()
    return out_list


def _output_img_dict_single(dictionary: dict):
    """
    Turns a dict of images into a YAML-able dict of paths.
    :param dictionary:
    :return:
    """
    out_dict = {}
    for fil in dictionary:
        img = dictionary[fil]
        if isinstance(img, image.Image):
            out_dict[fil] = img.path
        elif isinstance(img, str):
            out_dict[fil] = img
    return out_dict


def _output_img_dict_list(dictionary: dict):
    """
    Turns a dict of lists of images into a YAML-able dict of lists of paths.
    :param dictionary:
    :return:
    """
    out_dict = {}
    for fil in dictionary:
        if dictionary[fil] is None:
            out_dict[fil] = None
        elif len(dictionary[fil]) > 0:
            if isinstance(dictionary[fil][0], image.Image):
                out_dict[fil] = list(set(map(lambda f: f.path, dictionary[fil])))
                out_dict[fil].sort()
            elif isinstance(dictionary[fil][0], str):
                out_dict[fil] = dictionary[fil]
        else:
            out_dict[fil] = []
    return out_dict


class Epoch:
    instrument_name = "dummy-instrument"
    mode = "dummy_mode"
    frame_class = image.Image

    def __init__(
            self,
            param_path: str = None,
            name: str = None,
            field: Union[str, 'fld.Field'] = None,
            data_path: str = None,
            instrument: str = None,
            date: Union[str, Time] = None,
            program_id: str = None,
            target: str = None,
            do_stages: Union[list, str] = None,
            **kwargs
    ):

        self.quiet = False
        if "quiet" in kwargs:
            self.quiet = kwargs["quiet"]

        # Input attributes
        self.param_path = param_path
        self.name = name
        self.field = field
        self.data_path = None
        self.data_path_relative = None
        if data_path is not None:
            self.data_path = os.path.join(p.data_dir, data_path)
            self.data_path_relative = data_path
        if data_path is not None:
            u.mkdir_check_nested(self.data_path)
        u.debug_print(2, f"__init__(): {self.name}.data_path ==", self.data_path)
        self.instrument_name = instrument
        try:
            self.instrument = inst.Instrument.from_params(instrument_name=str(instrument))
        except FileNotFoundError:
            self.instrument = None

        self.date = date
        # If we have a datetime.date, convert it to string before attempting to turn it into an astropy Time
        if isinstance(self.date, datetime.date):
            self.date = str(self.date)
        # print(self.date, type(self.date))
        if not isinstance(self.date, Time) and self.date is not None:
            self.date = Time(self.date, out_subfmt="date")
        self.program_id = program_id
        self.target = target

        self.do = do_stages

        # Written attributes
        self.output_file = None  # This will be set during the load_output_file call
        self.stages_complete = {}
        self.log = log.Log()

        self.binning = None
        self.binning_std = None

        # Data reduction paths
        self.paths = {}

        # Frames
        self.frames_raw = []
        self.frames_bias = []
        self.frames_standard = {}
        self.frames_science = []
        self.frames_dark = []
        self.frames_flat = {}

        self.frames_reduced = []

        self.coadded = {}

        u.debug_print(2, f"Epoch.__init__(): kwargs ==", kwargs)

        self.do_kwargs = {}
        u.debug_print(2, "do" in kwargs)
        if "do" in kwargs:
            self.do_kwargs = kwargs["do"]

        u.debug_print(2, f"Epoch.__init__(): {self}.do_kwargs ==", self.do_kwargs)

        dkwargs = {}
        if "filters" in self.__dict__:
            dkwargs["bands"] = self.__dict__["filters"]

        add_to_epoch_directory(
            field_name=self.field.name,
            instrument=self.instrument_name,
            mode=self.mode,
            epoch_name=self.name,
            **dkwargs
        )

        self.param_file = kwargs

        self.combined_from = []

        self.combined_epoch = False
        if "combined_epoch" in kwargs:
            self.combined_epoch = kwargs["combined_epoch"]

        # self.load_output_file()

        active_epochs[self.name] = self

    def __str__(self):
        return self.name

    def date_str(self, include_time: bool = False):
        if not isinstance(self.date, Time):
            return str(self.date)
        elif include_time:
            return str(self.date.isot)
        else:
            return self.date.strftime('%Y-%m-%d')

    def mjd(self):
        if not isinstance(self.date, Time):
            return 0.
        else:
            return self.date.mjd

    def add_log(
            self,
            action: str,
            method=None,
            method_args=None,
            path: str = None,
            packages: List[str] = None,
    ):
        self.log.add_log(
            action=action,
            method=method,
            method_args=method_args,
            output_path=path,
            packages=packages)
        # self.update_output_file()

    @classmethod
    def stages(cls):
        stages = {
            "initial_setup": {
                "method": cls.proc_initial_setup,
                "message": "Do initial setup of files?",
                "log_message": "Initial setup conducted.",
                "default": True,
                "keywords": {

                }
            }
        }

        return stages

    def pipeline(self, no_query: bool = False, **kwargs):
        """
        Performs the pipeline methods given in stages() for this Epoch.
        :param no_query: If True, skips the query stage and performs all stages (unless "do" was provided on __init__),
            in which case it will perform only those stages without query no matter what no_query is.
        :return:
        """
        self._pipeline_init()
        # u.debug_print(2, "Epoch.pipeline(): kwargs ==", kwargs)

        # Loop through stages list specified in self.stages()
        stages = self.stages()
        u.debug_print(1, f"Epoch.pipeline(): type(self) ==", type(self))
        u.debug_print(2, f"Epoch.pipeline(): stages ==", stages)
        last_complete = None
        for n, name in enumerate(stages):
            stage = stages[name]
            message = stage["message"]
            # If default is present, then it defines whether the stage should be performed by default. If True, it
            # must be switched off by the do_key to skip the step; if False, then do_key must be set to True to perform
            # the step. This should work.
            if "default" in stage:
                do_this = stage["default"]
            else:
                do_this = True

            # Check if name is in "do" dict. If it is, defer to that setting; if not, defer to default.
            if name in self.do_kwargs:
                do_this = self.do_kwargs[name]

            u.debug_print(2, f"Epoch.pipeline(): {self}.stages_complete ==", self.stages_complete)

            if name in self.param_file:
                stage_kwargs = self.param_file[name]
            else:
                stage_kwargs = {}

            # Check if we should do this stage
            if do_this and (no_query or self.query_stage(
                    message=message,
                    n=n,
                    stage_name=name,
                    stage_kwargs=stage_kwargs
            )):
                if not self.quiet:
                    print(f"Performing processing step {n}: {name}")
                # Construct path; if dir_name is None then the step is pathless.
                dir_name = f"{n}-{name}"
                output_dir = os.path.join(self.data_path, dir_name)
                output_dir_backup = output_dir + "_backup"
                u.rmtree_check(output_dir_backup)
                u.move_check(output_dir, output_dir_backup)
                u.mkdir_check_nested(output_dir, remove_last=False)
                self.set_path(name, output_dir)

                if stage["method"](self, output_dir=output_dir, **stage_kwargs) is not False:
                    self.stages_complete[name] = Time.now()

                    if "log_message" in stage and stage["log_message"] is not None:
                        log_message = stage["log_message"]
                    else:
                        log_message = f"Performed processing step {dir_name}."
                    self.add_log(log_message, method=stage["method"], path=output_dir, method_args=stage_kwargs)

                    u.rmtree_check(output_dir_backup)

                self.update_output_file()

                last_complete = dir_name

        return last_complete

    def _pipeline_init(self):
        if self.data_path is not None:
            u.debug_print(2, f"{self}._pipeline_init(): self.data_path ==", self.data_path)
            u.mkdir_check_nested(self.data_path)
        else:
            raise ValueError(f"data_path has not been set for {self}")
        self.field.retrieve_catalogues()
        self.do = _check_do_list(self.do, stages=list(self.stages().keys()))
        if not self.quiet:
            print(f"Doing stages {self.do}")
        self.paths["download"] = os.path.join(self.data_path, "0-download")

    def proc_initial_setup(self, output_dir: str, **kwargs):
        self._initial_setup(output_dir=output_dir, **kwargs)
        return True

    def _initial_setup(self, output_dir: str, **kwargs):
        pass

    @classmethod
    def _check_output_file_path(cls, key: str, dictionary: dict):
        return key in dictionary and dictionary[key] is not None and os.path.isfile(dictionary[key])

    def load_output_file(self, **kwargs) -> dict:
        """
        Loads the output .yaml file, which contains various values derived from this Epoch, using the object's
        output_file attribute (which is a path to the file).
        :param kwargs: keyword arguments to pass to the add_coadded_image() method.
        :return: output file as a dict.
        """
        outputs = p.load_output_file(self)
        if type(outputs) is dict:
            if "stages" in outputs:
                self.stages_complete.update(outputs["stages"])
            if "coadded" in outputs:
                for fil in outputs["coadded"]:
                    if outputs["coadded"][fil] is not None:
                        self.add_coadded_image(img=outputs["coadded"][fil], key=fil, **kwargs)
            if "log" in outputs:
                self.log = log.Log(outputs["log"])
        return outputs

    def _output_dict(self):

        return {
            "date": self.date,
            "stages": self.stages_complete,
            "paths": self.paths,
            "frames_science": _output_img_dict_list(self.frames_science),
            "frames_flat": _output_img_dict_list(self.frames_flat),
            "frames_std": _output_img_dict_list(self.frames_standard),
            "frames_bias": _output_img_list(self.frames_bias),
            "coadded": _output_img_dict_single(self.coadded),
            "log": self.log.to_dict(),
            "combined_from": self.combined_from
        }

    def update_output_file(self):
        p.update_output_file(self)

    def check_done(self, stage: str):
        u.debug_print(2, "Epoch.check_done(): stage ==", stage)
        u.debug_print(2, f"Epoch.check_done(): {self}.stages_complete ==", self.stages_complete)
        if stage not in self.stages():
            raise ValueError(f"{stage} is not a valid stage for this Epoch.")
        if stage in self.stages_complete:
            return self.stages_complete[stage]
        else:
            return None

    def query_stage(self, message: str, stage_name: str, n: float, stage_kwargs: dict = None):
        """
        Helper method for asking the user if we need to do this stage of processing.
        If self.do is True, skips the query and returns True.
        :param message: Message to display.
        :param stage_name: code-friendly name of stage, eg "coadd" or "initial_setup"
        :param n: Stage number
        :return:
        """
        # Check if n is an integer, and if so cast to int.
        if n == int(n):
            n = int(n)
        if self.do is not None:
            if stage_name in self.do:
                return True
        else:
            message = f"{self.name} {n}. {message}"
            done = self.check_done(stage=stage_name)
            u.debug_print(2, "Epoch.query_stage(): done ==", done)
            if done is not None:
                time_since = (Time.now() - done).sec * units.second
                time_since = u.relevant_timescale(time_since)
                message += f" (last performed at {done.isot}, {time_since.round(1)} ago)"
                if stage_kwargs:
                    message += f"\nSpecified config keywords:\n{stage_kwargs}"
            return u.select_yn_exit(message=message)

    # def set_survey(self):

    def set_program_id(self, program_id: str):
        self.program_id = program_id
        self.update_param_file("program_id")

    def set_date(self, date: Union[str, Time]):
        if isinstance(date, str):
            date = Time(date)
        self.date = date
        self.update_param_file("date")

    def set_target(self, target: str):
        self.target = target
        self.update_param_file("target")

    def get_binning(self):
        return self.binning

    def set_binning(self, binning: str):
        self.binning = binning
        return binning

    def get_binning_std(self):
        return self.binning_std

    def set_binning_std(self, binning: str):
        self.binning_std = binning
        return binning

    def get_path(self, key: str):
        if key in self.paths:
            return self.paths[key]
        else:
            raise KeyError(f"{key} has not been set.")

    def set_path(self, key: str, value: str):
        self.paths[key] = value

    def update_param_file(self, param: str):
        p_dict = {"program_id": self.program_id,
                  "date": self.date,
                  "target": self.target}
        if param not in p_dict:
            raise ValueError(f"Either {param} is not a valid parameter, or it has not been configured.")
        if self.param_path is None:
            raise ValueError("param_path has not been set.")
        else:
            params = p.load_params(self.param_path)
        params[param] = p_dict[param]
        p.save_params(file=self.param_path, dictionary=params)

    @classmethod
    def sort_by_chip(cls, images: list):
        chips = {}

        for img in images:
            chip_this = img.extract_chip_number()
            if chip_this is None:
                print(f"The chip number for {img.name} could not be determined.")
            else:
                if chip_this not in chips:
                    chips[chip_this] = []
                chips[chip_this].append(img)

        return chips

    def add_frame_raw(self, raw_frame: Union[image.ImagingImage, str]):
        u.debug_print(
            2,
            f"add_frame_raw(): Adding frame {raw_frame.name}, type {raw_frame.frame_type}, to {self}, type {type(self)}")
        self.frames_raw.append(raw_frame)
        self.sort_frame(raw_frame)

    def add_frame_reduced(self, reduced_frame: image.Image):
        if reduced_frame not in self.frames_reduced:
            self.frames_reduced.append(reduced_frame)

    def _add_coadded(self, img: Union[str, image.Image], key: str, image_dict: dict):
        if isinstance(img, str):
            u.debug_print(2, f"Epoch._add_coadded(): {self.name}.instrument_name ==", self.instrument_name)
            if os.path.isfile(img):
                cls = image.CoaddedImage.select_child_class(instrument_name=self.instrument_name)
                u.debug_print(2, f"Epoch._add_coadded(): cls ==", cls)

                img = image.from_path(
                    path=img,
                    instrument_name=self.instrument_name,
                    cls=cls
                )
            else:
                return None
        img.epoch = self
        image_dict[key] = img
        return img

    def add_coadded_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded)

    def sort_frame(self, frame: image.Image, sort_key: str = None):
        frame.extract_frame_type()
        u.debug_print(
            2,
            f"sort_frame(); Adding frame {frame.name}, type {frame.frame_type}, to {self}, type {type(self)}")

        # chip = frame.extract_chip_number()
        # print(frame.frame_type)
        u.debug_print(2, f"Epoch.sort_frame(): {type(self.frames_science)=}")
        if frame.frame_type == "bias" and frame not in self.frames_bias:
            self.frames_bias.append(frame)

        elif frame.frame_type == "science":
            if isinstance(self.frames_science, list):
                if frame not in self.frames_science:
                    self.frames_science.append(frame)
            elif isinstance(self.frames_science, dict):
                if frame not in self.frames_science[sort_key]:
                    self.frames_science[sort_key].append(frame)

        elif frame.frame_type == "standard":
            if isinstance(self.frames_standard, list):
                if frame not in self.frames_standard:
                    self.frames_standard.append(frame)
            elif isinstance(self.frames_standard, dict):
                if frame not in self.frames_standard[sort_key]:
                    self.frames_standard[sort_key].append(frame)

        elif frame.frame_type == "dark" and frame not in self.frames_dark:
            self.frames_dark.append(frame)

        elif frame.frame_type == "flat":
            if isinstance(self.frames_flat, list):
                if frame not in self.frames_flat:
                    self.frames_flat.append(frame)
            elif isinstance(self.frames_flat, dict):
                if frame not in self.frames_flat[sort_key]:
                    self.frames_flat[sort_key].append(frame)

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "field": None,
            "data_path": None,
            "instrument": None,
            "date": None,
            "target": None,
            "program_id": None,
            "do": {},
            "notes": [],
            "combined_epoch": False
        }
        # Pull the list of applicable kwargs from the stage information
        stages = cls.stages()
        print(cls)
        print(stages)
        for stage in stages:
            stage_info = stages[stage]
            if "keywords" in stage_info:
                default_params[stage] = stage_info["keywords"]
            else:
                default_params[stage] = {}
        return default_params

    @classmethod
    def new_yaml(cls, name: str, path: str = None, **kwargs):
        param_dict = cls.default_params()
        param_dict["name"] = name
        for kwarg in kwargs:
            param_dict[kwarg] = kwargs[kwarg]
        if path is not None:
            if os.path.isdir(path):
                path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict)
        return param_dict

    @classmethod
    def _from_params_setup(cls, name: str, field: Union['fld.Field', str] = None):
        field_name = None
        if isinstance(field, fld.Field):
            field_name = field.name
        elif isinstance(field, str):
            field_name = field
            field = None
        elif field is not None:
            raise TypeError(f"field must be str or Field, not {type(field)}")
        if field_name is None:
            field_name = name.split("_")[0]
        return field_name, field


class StandardEpoch(Epoch):
    instrument_name = "dummy-instrument"

    def __init__(
            self,
            centre_coords: SkyCoord,
            instrument: str,
            frames_standard: Dict[str, List[image.ImagingImage]] = {},
            frames_flat: Dict[str, List[image.ImagingImage]] = {},
            frames_bias: List[image.ImagingImage] = [],
            date: Union[str, Time] = None,
            **kwargs
    ):
        field = fld.StandardField(centre_coords=centre_coords)
        name = f"{field.name}_{date.strftime('%Y-%m-%d')}"
        param_path = os.path.join(p.param_dir, "fields", field.name, "imaging", f"{name}.yaml")

        if not os.path.isfile(param_path):
            self.new_yaml(
                name=name,
                path=param_path,
                centre=objects.skycoord_to_position_dict(centre_coords)
            )

        super().__init__(
            param_path=param_path,
            name=f"{field.name}_{date}",
            field=field,
            data_path=os.path.join(field.data_path, "imaging", str(instrument), name),
            instrument=str(instrument),
            date=date,
            **kwargs
        )

        self.frames_standard = frames_standard
        self.frames_bias = frames_bias,
        self.frames_flat = frames_flat

    @classmethod
    def select_child_class(cls, instrument: Union[str, inst.Instrument]):
        if isinstance(instrument, inst.Instrument):
            instrument = instrument.name
        if instrument == "vlt-fors2":
            return FORS2StandardEpoch
        else:
            return StandardEpoch


class ImagingEpoch(Epoch):
    instrument_name = "dummy-instrument"
    mode = "imaging"
    frame_class = image.ImagingImage
    coadded_class = image.CoaddedImage
    frames_for_combined = "astrometry"
    skip_for_combined = [
        "download",
        "initial_setup",
        "sort_reduced",
        "trim_reduced",
        "convert_to_cs",
        "correct_astrometry_frames"
    ]

    def __init__(
            self,
            name: str = None,
            field: Union[str, 'fld.Field'] = None,
            param_path: str = None,
            data_path: str = None,
            instrument: str = None,
            date: Union[str, Time] = None,
            program_id: str = None,
            target: str = None,
            source_extractor_config: dict = None,
            standard_epochs: list = None,
            **kwargs
    ):
        super().__init__(
            name=name,
            field=field,
            param_path=param_path,
            data_path=data_path,
            instrument=instrument,
            date=date,
            program_id=program_id,
            target=target,
            **kwargs
        )
        self.guess_data_path()
        self.source_extractor_config = source_extractor_config
        if self.source_extractor_config is None:
            self.source_extractor_config = {
                "dual_mode": False,
                "threshold": 1.5,
                "kron_factor": 3.5,
                "kron_radius_min": 1.0
            }

        self.filters = []
        self.deepest = None
        self.deepest_filter = None

        self.exp_time_mean = {}
        self.exp_time_err = {}
        self.airmass_mean = {}
        self.airmass_err = {}

        self.frames_science = {}
        self.frames_reduced = {}
        self.frames_trimmed = {}
        self.frames_subtracted = {}
        self.frames_normalised = {}
        self.frames_registered = {}
        self.frames_astrometry = {}
        self.astrometry_successful = {}
        self.frames_diagnosed = {}
        self.frames_final = None

        self.std_pointings = []
        self.std_objects = {}
        self.std_epochs = {}

        self.coadded_trimmed = {}
        self.coadded_unprojected = {}
        self.coadded_astrometry = {}
        self.coadded_subtracted = {}
        self.coadded_final = None

        self.gaia_catalogue = None
        self.astrometry_indices = []

        self.frame_stats = {}
        self.astrometry_stats = {}
        self.psf_stats = {}

        # self.load_output_file(mode="imaging")

    def _pipeline_init(self):
        super()._pipeline_init()
        for fil in self.filters:
            self.check_filter(fil)

    @classmethod
    def stages(cls):

        stages = super().stages()
        stages.update({
            "download": {
                "method": cls.proc_download,
                "message": "Pretend to download files? (download not actualy implemented for this class)",
                "default": False,
                "keywords": {
                    "alternate_dir": None
                }
            },
            "register_frames": {
                "method": cls.proc_register,
                "message": "Register frames using astroalign?",
                "default": False,
                "keywords": {
                    "template": 0,
                    "include_chips": "all"
                }
            },
            "correct_astrometry_frames": {
                "method": cls.proc_correct_astrometry_frames,
                "message": "Correct astrometry of individual frames?",
                "default": True,
                "keywords": {
                    "tweak": True,
                    "upper_only": False,
                    "method": "individual",
                    "back_subbed": False,
                }
            },
            "frame_diagnostics": {
                "method": cls.proc_frame_diagnostics,
                "message": "Run diagnostics on individual frames?",
                "default": False,
            },
            "subtract_background_frames": {
                "method": cls.proc_subtract_background_frames,
                "message": "Subtract local background from frames?",
                "default": False,
                "keywords": {}
            },
            "coadd": {
                "method": cls.proc_coadd,
                "message": "Coadd frames with Montage?",
                "default": True,
                "keywords": {
                    "frames": "astrometry",  # normalised, trimmed
                    "sigma_clip": 1.0
                }
            },
            "correct_astrometry_coadded": {
                "method": cls.proc_correct_astrometry_coadded,
                "message": "Correct astrometry of coadded images?",
                "default": False,
                "keywords": {
                    "tweak": True,
                    "astroalign_template": None,
                }
            },
            "trim_coadded": {
                "method": cls.proc_trim_coadded,
                "message": "Trim / reproject coadded images to same footprint?",
                "default": True,
                "keywords": {
                    "reproject": False  # Reproject to same footprint?
                }
            },
            "source_extraction": {
                "method": cls.proc_source_extraction,
                "message": "Do source extraction and diagnostics?",
                "default": True,
                "keywords": {
                    "do_astrometry_diagnostics": True
                }
            },
            "photometric_calibration": {
                "method": cls.proc_photometric_calibration,
                "message": "Do photometric calibration?",
                "default": True,
                "keywords": {
                    "distance_tolerance": None,
                    "snr_min": 3.,
                    "class_star_tolerance": 0.95,
                    "image_type": "final",
                    "preferred_zeropoint": {},
                    "suppress_select": True
                }
            },
            "dual_mode_source_extraction": {
                "method": cls.proc_dual_mode_source_extraction,
                "message": "Do source extraction in dual-mode, using deepest image as footprint?",
                "default": False,
            },
            "get_photometry": {
                "method": cls.proc_get_photometry,
                "message": "Get photometry?",
                "default": True,
            },
            # "get_photometry_all": {
            #     "method": cls.proc_get_photometry_all,
            #     "message": "Get all photometry?",
            #     "default": True
            # }
        }
        )
        return stages

    def n_frames(self, fil: str):
        return len(self.frames_reduced[fil])

    def proc_download(self, output_dir: str, **kwargs):
        pass

    def proc_subtract_background_frames(self, output_dir: str, **kwargs):
        self.frames_subtracted = {}
        if "frames" not in kwargs:
            if "correct_astrometry_frames" in self.do_kwargs and self.do_kwargs["correct_astrometry_frames"]:
                kwargs["frames"] = "astrometry"
            else:
                kwargs["frames"] = "normalised"
        self.subtract_background_frames(
            output_dir=output_dir,
            **kwargs
        )

    def subtract_background_frames(
            self,
            output_dir: str,
            frames: Union[dict, str] = None,
            method: str = "local",
            **kwargs
    ):
        if isinstance(frames, str):
            frames = self._get_frames(frames)

        print(self.field.objects_dict)

        if "do_not_mask" in kwargs:
            do_not_mask = kwargs.pop("do_not_mask")
            for i, obj in enumerate(do_not_mask):
                if isinstance(obj, str):
                    if obj in self.field.objects_dict:
                        obj = self.field.objects_dict[obj].position
                    elif not isinstance(obj, SkyCoord):
                        obj = astm.attempt_skycoord(obj)
                do_not_mask[i] = obj
            if "mask_kwargs" not in kwargs:
                kwargs["mask_kwargs"] = {}
            kwargs["mask_kwargs"]["do_not_mask"] = do_not_mask

        for fil in frames:
            frame_list = frames[fil]
            for frame in frame_list:
                subbed_path = os.path.join(output_dir, fil, frame.name + "_backsub.fits")
                back_path = os.path.join(output_dir, fil, frame.name + "_background.fits")
                if method in ("sep", "photutils"):
                    frame.model_background_photometry(
                        write_subbed=subbed_path,
                        write=back_path,
                        do_mask=True,
                        method=method,
                        **kwargs
                    )
                elif method == "local":
                    if "centre" not in kwargs:
                        if isinstance(self.field, fld.FRBField):
                            kwargs["centre"] = self.field.frb.position
                        else:
                            kwargs["centre"] = frame.extract_pointing()
                    else:
                        kwargs["centre"] = astm.attempt_skycoord(kwargs["centre"])
                    if "frame" not in kwargs:
                        kwargs["frame"] = 15 * units.arcsec
                    if isinstance(self.field, fld.FRBField):
                        a, b = self.field.frb.position_err.uncertainty_quadrature_equ()
                        mask_ellipses = [{
                            "a": a, "b": b,
                            "theta": self.field.frb.position_err.theta,
                            "centre": self.field.frb.position
                        }]
                    else:
                        mask_ellipses = None
                    frame.model_background_local(
                        write_subbed=subbed_path,
                        write=back_path,
                        generate_mask=True,
                        mask_ellipses=mask_ellipses,
                        **kwargs
                    )

                new_frame = type(frame)(subbed_path)
                self.add_frame_subtracted(new_frame)

    def proc_register(self, output_dir: str, **kwargs):
        self.frames_registered = {}
        self.register(
            output_dir=output_dir,
            **kwargs,
        )

    def register(
            self,
            output_dir: str,
            frames: dict = None,
            template: Union[int, dict, image.ImagingImage, str] = 0,
            **kwargs
    ):
        """

        :param output_dir:
        :param frames:
        :param template: There are three options for this parameter:
            int: An integer specifying the position of the image in the list to use as the template for
            alignment (ie, each filter will use the same list position)
            dict: a dictionary with keys reflecting the filter names, with values specifying the list position as above
            ImagingImage: an image from outside this epoch to use as template. You can also pass the path to the image
                as a string.
        :param kwargs:
        :return:
        """

        u.mkdir_check(output_dir)
        u.debug_print(1, f"{self}.register(): template ==", template)

        if frames is None:
            frames = self.frames_normalised

        for fil in frames:
            if not self.quiet:
                print(f"Registering frames for {fil}")
            if isinstance(template, int):
                tmp = frames[fil][template]
                n_template = template
            elif isinstance(template, image.ImagingImage):
                # When
                tmp = template
                n_template = -1
            elif isinstance(template, str):
                tmp = image.from_path(
                    path=template,
                    cls=image.ImagingImage
                )
                n_template = -1
            else:
                tmp = frames[fil][template[fil]]
                n_template = template[fil]
            u.debug_print(1, f"{self}.register(): tmp", tmp)

            output_dir_fil = os.path.join(output_dir, fil)
            u.mkdir_check(output_dir_fil)

            self._register(frames=frames, fil=fil, tmp=tmp, output_dir=output_dir_fil, n_template=n_template, **kwargs)

    def _register(self, frames: dict, fil: str, tmp: image.ImagingImage, n_template: int, output_dir: str, **kwargs):

        include_chips = list(range(1, self.frame_class.num_chips + 1))
        if "include_chips" in kwargs and isinstance(kwargs["include_chips"], list):
            include_chips = kwargs["include_chips"]

        frames_by_chip = self.sort_by_chip(frames[fil])

        for chip in include_chips:
            for i, frame in enumerate(frames_by_chip[chip]):
                if i != n_template:
                    registered = frame.register(
                        target=tmp,
                        output_path=os.path.join(
                            output_dir,
                            frame.filename.replace(".fits", "_registered.fits"))
                    )
                    self.add_frame_registered(registered)
                else:
                    registered = frame.copy(
                        os.path.join(
                            output_dir,
                            tmp.filename.replace(".fits", "_registered.fits")))
                    self.add_frame_registered(registered)

    def proc_correct_astrometry_frames(
            self,
            output_dir: str,
            **kwargs
    ):

        if "correct_to_epoch" in kwargs:
            if not self.quiet:
                print(f"correct_to_epoch 1: {kwargs['correct_to_epoch']}")
            correct_to_epoch = kwargs.pop("correct_to_epoch")
            if not self.quiet:
                print(f"correct_to_epoch 2: {correct_to_epoch}")
        else:
            correct_to_epoch = True

        u.debug_print(2, kwargs)

        self.generate_astrometry_indices(correct_to_epoch=correct_to_epoch)

        self.frames_astrometry = {}

        if "frames" in kwargs:
            frames = self._get_frames(frame_type=kwargs.pop("frames"))
        elif "register_frames" in self.do_kwargs and self.do_kwargs["register_frames"]:
            frames = self._get_frames(frame_type="registered")
        else:
            frames = self._get_frames(frame_type="normalised")

        self.correct_astrometry_frames(
            output_dir=output_dir,
            frames=frames,
            **kwargs
        )

    def correct_astrometry_frames(
            self,
            output_dir: str,
            frames: dict = None,
            am_params: dict = {},
            background_kwargs: dict = {},
            **kwargs
    ):
        self.frames_astrometry = {}
        self.astrometry_successful = {}

        if "back_subbed" in kwargs:
            back_subbed = kwargs.pop("back_subbed")
        else:
            back_subbed = False

        if frames is None:
            frames = self.frames_reduced

        for fil in frames:
            frames_by_chip = self.sort_by_chip(frames[fil])
            for chip in frames_by_chip:
                if not self.quiet:
                    print()
                    print(f"Processing frames for chip {chip} in astrometry.net:")
                    print()
                first_success = None
                astrometry_fil_path = os.path.join(output_dir, fil)
                for frame in frames_by_chip[chip]:
                    frame_alt = None
                    if back_subbed:
                        # For some fields, we want to subtract the background before attempting to solve, because of
                        # bright stars or the like.
                        frame_alt = frame
                        # Store the original frame for later.
                        subbed_path = os.path.join(output_dir, fil, frame.name + "_backsub.fits")
                        back_path = os.path.join(output_dir, fil, frame.name + "_background.fits")
                        # Use sep to subtract a background model.
                        frame.model_background_photometry(
                            write_subbed=subbed_path,
                            write=back_path,
                            do_mask=True,
                            method="sep",
                            **background_kwargs
                        )
                        # Assign frame to the subtracted file
                        frame = type(frame)(subbed_path)

                    new_frame = frame.correct_astrometry(
                        output_dir=astrometry_fil_path,
                        am_params=am_params,
                        **kwargs
                    )

                    if new_frame is not None:
                        if not self.quiet:
                            print(f"{frame} astrometry successful.")
                        if back_subbed:
                            new_frame = frame_alt.correct_astrometry_from_other(
                                new_frame,
                                output_dir=astrometry_fil_path
                            )
                            frame = frame_alt
                        self.add_frame_astrometry(new_frame)
                        self.astrometry_successful[fil][frame.name] = "astrometry.net"
                        if first_success is None:
                            first_success = new_frame
                    else:
                        if not self.quiet:
                            print(f"{frame} Astrometry.net unsuccessful; adding frame to astroalign queue.")
                        self.astrometry_successful[fil][frame.name] = False

                    u.debug_print(1, f"ImagingEpoch.correct_astrometry_frames(): {self}.astrometry_successful ==\n",
                                  self.astrometry_successful)
                    self.update_output_file()

                if 'registration_template' in kwargs and kwargs['registration_template'] is not None:
                    first_success = image.from_path(
                        kwargs['registration_template'],
                        cls=image.ImagingImage
                    )
                elif first_success is None:
                    tmp = frames_by_chip[chip][0]
                    if not self.quiet:
                        print(
                            f"There were no successful frames for chip {chip} using astrometry.net; performing coarse correction on {tmp}.")
                    first_success = tmp.correct_astrometry_coarse(
                        output_dir=astrometry_fil_path,
                        cat=self.gaia_catalogue,
                        cat_name="gaia"
                    )
                    self.add_frame_astrometry(first_success)
                    self.astrometry_successful[fil][tmp.name] = "coarse"
                    self.update_output_file()

                u.debug_print(2, "first_success", first_success)

                if not self.quiet:
                    print()
                    print(
                        f"Re-processing failed frames for chip {chip} with astroalign, with template {first_success}:")
                    print()
                for frame in frames_by_chip[chip]:
                    if not self.astrometry_successful[fil][frame.name]:
                        if not self.quiet:
                            print(f"Running astroalign on {frame}...")
                        new_frame = frame.register(
                            target=first_success,
                            output_path=os.path.join(
                                astrometry_fil_path,
                                frame.filename.replace(".fits", "_astrometry.fits")),
                        )
                        self.add_frame_astrometry(new_frame)
                        self.astrometry_successful[fil][frame.name] = "astroalign"
                    self.update_output_file()

    def proc_frame_diagnostics(self, output_dir: str, **kwargs):
        if "frames" in kwargs:
            frames = kwargs["frames"]
        else:
            frames = self.frames_final

        frame_dict = self._get_frames(frames)

        self.frame_psf_diagnostics(output_dir, frame_dict=frame_dict)
        self.frames_final = "diagnosed"

    def frame_psf_diagnostics(self, output_dir: str, frame_dict: dict, chip: int = 1, sigma: float = 1.):
        for fil in frame_dict:
            frame_list = frame_dict[fil]
            # Grab one chip only, to save time
            frame_lists = self.sort_by_chip(images=frame_list)
            frame_list_chip = frame_lists[chip]
            match_cat = None

            names = []

            fwhms_mean_psfex = []
            fwhms_mean_gauss = []
            fwhms_mean_moffat = []
            fwhms_mean_se = []

            fwhms_median_psfex = []
            fwhms_median_gauss = []
            fwhms_median_moffat = []
            fwhms_median_se = []

            sigma_gauss = []
            sigma_moffat = []
            sigma_se = []

            for frame in frame_list_chip:
                configs = self.source_extractor_config
                frame.psfex_path = None
                frame.source_extraction_psf(
                    output_dir=output_dir,
                    phot_autoparams=f"{configs['kron_factor']},{configs['kron_radius_min']}"
                )
                if match_cat is None:
                    match_cat = frame.source_cat
                offset_tolerance = 0.5 * units.arcsec
                # If the frames haven't been astrometrically corrected, give some extra leeway
                if "correct_astrometry_frames" in self.do_kwargs and not self.do_kwargs["correct_astrometry_frames"]:
                    offset_tolerance = 1.0 * units.arcsec
                frame_stats, stars_moffat, stars_gauss, stars_sex = frame.psf_diagnostics(
                    match_to=match_cat
                )

                names.append(frame.name)

                fwhms_mean_psfex.append(frame_stats["fwhm_psfex"].value)
                fwhms_mean_gauss.append(frame_stats["gauss"]["fwhm_mean"].value)
                fwhms_mean_moffat.append(frame_stats["moffat"]["fwhm_mean"].value)
                fwhms_mean_se.append(frame_stats["sextractor"]["fwhm_mean"].value)

                fwhms_median_psfex.append(frame_stats["fwhm_psfex"].value)
                fwhms_median_gauss.append(frame_stats["gauss"]["fwhm_median"].value)
                fwhms_median_moffat.append(frame_stats["moffat"]["fwhm_median"].value)
                fwhms_median_se.append(frame_stats["sextractor"]["fwhm_median"].value)

                sigma_gauss.append(frame_stats["gauss"]["fwhm_sigma"].value)
                sigma_moffat.append(frame_stats["moffat"]["fwhm_sigma"].value)
                sigma_se.append(frame_stats["sextractor"]["fwhm_sigma"].value)

                self.frame_stats[fil][frame.name] = frame_stats

            median_all = np.median(fwhms_mean_gauss)
            sigma_all = np.std(fwhms_median_gauss)
            upper_limit = median_all + (sigma * sigma_all)

            plt.close()

            plt.title(f"PSF FWHM Mean")
            plt.ylabel("FWHM (\")")
            plt.errorbar(names, fwhms_mean_gauss, yerr=sigma_gauss, fmt="o", label="Gaussian")
            plt.errorbar(names, fwhms_mean_moffat, yerr=sigma_gauss, fmt="o", label="Moffat")
            plt.errorbar(names, fwhms_mean_se, yerr=sigma_gauss, fmt="o", label="Source Extractor")
            plt.plot([0, len(names)], [upper_limit, upper_limit], c="black", label="Clip threshold")
            plt.legend()
            plt.xticks(rotation=-90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{fil}_psf_diagnostics_mean.png"))
            plt.close()

            plt.title(f"PSF FWHM Median")
            plt.ylabel("FWHM (\")")
            plt.errorbar(names, fwhms_median_gauss, yerr=sigma_gauss, fmt="o", label="Gaussian")
            plt.errorbar(names, fwhms_median_moffat, yerr=sigma_gauss, fmt="o", label="Moffat")
            plt.errorbar(names, fwhms_median_se, yerr=sigma_gauss, fmt="o", label="Source Extractor")
            plt.plot([0, len(names)], [upper_limit, upper_limit], c="black", label="Clip threshold")
            plt.legend()
            plt.xticks(rotation=-90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{fil}_psf_diagnostics_median.png"))
            plt.close()

            self.frames_diagnosed[fil] = []
            for i, fwhm_median in enumerate(fwhms_median_gauss):
                if fwhm_median < upper_limit:
                    print(f"Median PSF FWHM {fwhm_median} < upper limit {upper_limit}")
                    for chip in frame_lists:
                        if not self.quiet:
                            print(f"\tAdding {frame_lists[chip][i]}")
                        self.add_frame_diagnosed(frame_lists[chip][i])
                elif not self.quiet:
                    print(f"Median PSF FWHM {fwhm_median} > upper limit {upper_limit}")

    def proc_coadd(self, output_dir: str, **kwargs):
        if "frames" not in kwargs:
            kwargs["frames"] = self.frames_final
        self.coadd(output_dir, **kwargs)
        if np.any(list(map(lambda k: len(self.frames_subtracted[k]) > 0, self.frames_subtracted))):
            kwargs["frames"] = "subtracted"
            self.coadd(
                output_dir + "_background_subtracted",
                out_dict="subtracted",
                **kwargs
            )

    def coadd(
            self,
            output_dir: str,
            frames: str = "astrometry",
            out_dict: Union[dict, str] = "coadded",
            sigma_clip: float = 1.5
    ):
        """
        Use Montage and ccdproc to coadd individual frames.
        :param output_dir: Directory in which to write data products.
        :param frames: Name of frames list to coadd.
        :param sigma_clip: Multiple of pixel stack standard deviation to clip when doing sigma-clipped stack.
        :return:
        """
        if isinstance(out_dict, str):
            if not self.quiet:
                print("out_dict:", out_dict)
            out_dict = self._get_images(image_type=out_dict)

        u.mkdir_check(output_dir)
        frame_dict = self._get_frames(frame_type=frames)
        if not self.quiet:
            print(f"Coadding {frames} frames.")
        for fil in self.filters:
            frame_list = frame_dict[fil]
            output_directory_fil = os.path.join(output_dir, fil)
            u.rmtree_check(output_directory_fil)
            u.mkdir_check(output_directory_fil)
            input_directory_fil = os.path.join(output_directory_fil, "inputdir")
            u.mkdir_check(input_directory_fil)
            for frame in frame_list:
                frame.copy_with_outputs(input_directory_fil)

            coadded_path = montage.standard_script(
                input_directory=input_directory_fil,
                output_directory=output_directory_fil,
                output_file_name=f"{self.name}_{self.date_str()}_{fil}_coadded.fits",
                coadd_types=["median", "mean"],
                add_with_ccdproc=False,
                sigma_clip=True,
                # unit="electron / second"
                # sigma_clip_low_threshold=5,
            )[0]

            sigclip_path = coadded_path.replace("median", "mean-sigmaclip")
            area_final = sigclip_path.replace(".fits", "_area.fits")
            shutil.copy(coadded_path.replace(".fits", "_area.fits"), area_final)

            corr_dir = os.path.join(output_directory_fil, "corrdir")
            coadded_median = image.FORS2CoaddedImage(coadded_path)
            coadded_median.add_log(
                "Co-added image using Montage; see ancestor_logs for images.",
                input_path=input_directory_fil,
                output_path=coadded_path,
                ancestors=frame_list
            )
            if self.combined_epoch:
                coadded_median.set_header_item("M_EPOCH", True, write=True)
            else:
                coadded_median.set_header_item("M_EPOCH", False, write=True)
            ccds = []
            # Here we gather the projected images in preparation for custom reprojection / coaddition
            for proj_img_path in list(map(
                    lambda m: os.path.join(corr_dir, m),
                    filter(
                        lambda f: f.endswith(".fits") and not f.endswith("area.fits"),
                        os.listdir(corr_dir)))):
                proj_img = image.FORS2Image(proj_img_path)
                reproj_img = proj_img.reproject(coadded_median, include_footprint=True)
                reproj_img_ccd = reproj_img.to_ccddata(unit="electron / second")
                ccds.append(reproj_img_ccd)

            combined_ccd = ccdproc.combine(
                img_list=ccds,
                method="average",
                sigma_clip=True,
                sigma_clip_func=np.nanmean,
                sigma_clip_dev_func=np.nanstd,
                sigma_clip_high_thresh=sigma_clip,
                sigma_clip_low_thresh=sigma_clip
            )
            combined_img = coadded_median.copy(sigclip_path)
            combined_img.area_file = area_final
            coadded_median.load_headers()
            combined_img.load_data()
            combined_img.data[0] = combined_ccd.data * coadded_median.extract_unit(astropy=True)
            u.debug_print(3, f"ImagingEpoch.coadd(): {combined_img}.headers ==", combined_img.headers)
            combined_img.add_log(
                "Co-added image using Montage for reprojection & ccdproc for coaddition; see ancestor_logs for input images.",
                input_path=input_directory_fil,
                output_path=coadded_path,
                ancestors=frame_list
            )
            combined_img.write_fits_file()
            combined_img.update_output_file()

            self._add_coadded(img=sigclip_path, key=fil, image_dict=out_dict)

    def proc_correct_astrometry_coadded(self, output_dir: str, **kwargs):
        self.correct_astrometry_coadded(
            output_dir=output_dir,
            **kwargs
        )

    def correct_astrometry_coadded(
            self,
            output_dir: str,
            image_type: str = None,
            **kwargs
    ):
        self.generate_astrometry_indices()

        self.coadded_astrometry = {}

        if image_type is None:
            image_type = "coadded"

        images = self._get_images(image_type)

        if "tweak" in kwargs:
            tweak = kwargs["tweak"]
        else:
            tweak = True

        if "astroalign_template" in kwargs:
            aa_template = kwargs["astroalign_template"]
        else:
            aa_template = None

        first_success = None
        unsuccessful = []
        for fil in images:
            img = images[fil]
            new_img = img.correct_astrometry(
                output_dir=output_dir,
                tweak=tweak
            )
            if new_img is None:
                if not self.quiet:
                    print(f"{img} Astrometry.net unsuccessful; adding image to astroalign queue.")
                unsuccessful.append(fil)
            else:
                if first_success is None:
                    first_success = new_img
                self.add_coadded_astrometry_image(new_img, key=fil)

        if first_success is None and aa_template is not None:
            cls = image.detect_instrument(path=aa_template)
            first_success = image.from_path(path=aa_template, cls=cls)

        if first_success is not None:
            for fil in unsuccessful:
                img = images[fil]
                new_img = img.register(
                    target=first_success,
                    output_path=os.path.join(
                        output_dir,
                        img.filename.replace(".fits", "_astrometry.fits")
                    )
                )
                self.add_coadded_astrometry_image(new_img, key=fil)

        for fil in images:
            if fil in self.coadded_subtracted and self.coadded_subtracted[fil] is not None:
                self.coadded_subtracted[fil] = self.coadded_subtracted[fil].correct_astrometry_from_other(
                    other_image=self.coadded_astrometry[fil],
                    output_dir=output_dir + "_background_subtracted"
                )
                self.coadded_subtracted[fil].area_file = self.coadded_astrometry[fil].area_file

    def proc_trim_coadded(self, output_dir: str, **kwargs):
        if "correct_astrometry_coadded" in self.do_kwargs and self.do_kwargs["correct_astrometry_coadded"]:
            images = self.coadded_astrometry
        else:
            images = self.coadded

        if "reproject" in kwargs:
            reproject = kwargs["reproject"]
        else:
            reproject = False
        self.trim_coadded(output_dir, images=images, reproject=reproject)

    def trim_coadded(
            self,
            output_dir: str,
            images: dict = None,
            reproject: bool = False
    ):
        if images is None:
            images = self.coadded

        u.mkdir_check(output_dir)
        template = None
        for fil in images:
            img = images[fil]
            if not self.quiet:
                print()
                print("Coadded Image Path:")
                print(img.output_file)
            output_path = os.path.join(output_dir, img.filename.replace(".fits", "_trimmed.fits"))
            u.debug_print(2, "trim_coadded img.path:", img.output_file)
            u.debug_print(2, "trim_coadded img.area_file:", img.area_file)
            trimmed = img.trim_from_area(output_path=output_path)
            # trimmed.write_fits_file()
            self.add_coadded_unprojected_image(trimmed, key=fil)
            if reproject:
                if template is None:
                    template = trimmed
                else:
                    # Using the first image as a template, reproject this one into the pixel space (for alignment)
                    trimmed = trimmed.reproject(
                        other_image=template,
                        output_path=output_path.replace(".fits", "_reprojected.fits")
                    )
            self.add_coadded_trimmed_image(trimmed, key=fil)
            if fil in self.coadded_subtracted and self.coadded_subtracted[fil] is not None:
                img_sub = self.coadded_subtracted[fil]
                output_path_sub = os.path.join(output_dir, img_sub.filename.replace(".fits", "_bgsub_trimmed.fits"))
                trimmed_sub = img_sub.trim_from_area(output_path=output_path_sub)
                self.coadded_subtracted[fil] = trimmed_sub

    def proc_source_extraction(self, output_dir: str, **kwargs):
        if "do_astrometry_diagnostics" not in kwargs:
            kwargs["do_astrometry_diagnostics"] = True
        if "do_psf_diagnostics" not in kwargs:
            kwargs["do_psf_diagnostics"] = True
        if "image_type" not in kwargs:
            kwargs["image_type"] = "final"
        self.source_extraction(
            output_dir=output_dir,
            **kwargs
        )

    def source_extraction(
            self,
            output_dir: str,
            do_astrometry_diagnostics: bool = True,
            do_psf_diagnostics: bool = True,
            image_type: str = "final",
            **kwargs
    ):
        images = self._get_images(image_type)
        if not self.quiet:
            print("\nExtracting sources for", image_type, "with", len(list(images.keys())), "\n")
        for fil, img in images.items():
            if not self.quiet:
                print(f"Extracting sources from {fil} image: {img}")
            configs = self.source_extractor_config

            img.psfex_path = None
            img.source_extraction_psf(
                output_dir=output_dir,
                phot_autoparams=f"{configs['kron_factor']},{configs['kron_radius_min']}"
            )
            # if fil in self.coadded_subtracted and self.coadded_subtracted[fil] is not None:
            #     img_subbed = self.coadded_subtracted[fil]
            #     img_subbed.source_cat = img.source_cat
            #     img_subbed.update_output_file()
        if do_astrometry_diagnostics:
            if "offset_tolerance" in kwargs:
                offset_tolerance = kwargs["offset_tolerance"]
            else:
                offset_tolerance = 0.5 * units.arcsec
            self.astrometry_diagnostics(
                images=images,
                offset_tolerance=offset_tolerance
            )

        if do_psf_diagnostics:
            self.psf_diagnostics(images=images)

    def proc_photometric_calibration(self, output_dir: str, **kwargs):
        if "image_type" in kwargs and kwargs["image_type"] is not None:
            image_type = kwargs["image_type"]
        else:
            image_type = "final"

        if "distance_tolerance" in kwargs and kwargs["distance_tolerance"] is not None:
            kwargs["distance_tolerance"] = u.check_quantity(kwargs["distance_tolerance"], units.arcsec, convert=True)
        if "snr_min" not in kwargs or kwargs["snr_min"] is None:
            kwargs["snr_min"] = 3.
        if "suppress_select" not in kwargs:
            kwargs["suppress_select"] = True

        image_dict = self._get_images(image_type=image_type)
        for fil in image_dict:
            img = image_dict[fil]
            img.zeropoints = {}
            img.zeropoint_best = None

        self.photometric_calibration(
            output_path=output_dir,
            image_dict=image_dict,
            **kwargs
        )

    def photometric_calibration(
            self,
            output_path: str,
            image_dict: dict,
            **kwargs
    ):
        u.mkdir_check(output_path)

        deepest = self.zeropoint(
            image_dict=image_dict,
            output_path=output_path,
            **kwargs
        )

        for fil in image_dict:
            if self.coadded_unprojected[fil] is not None and self.coadded_unprojected[fil] is not image_dict[fil]:
                img = self.coadded_unprojected[fil]
                img.zeropoints = image_dict[fil].zeropoints
                img.zeropoint_best = image_dict[fil].zeropoint_best
                img.update_output_file()
            if self.coadded_subtracted[fil] is not None:
                img = self.coadded_subtracted[fil]
                img.zeropoints = image_dict[fil].zeropoints
                img.zeropoint_best = image_dict[fil].zeropoint_best
                img.update_output_file()

        self.deepest_filter = deepest.filter_name
        self.deepest = deepest
        if not self.quiet:
            print("DEEPEST FILTER:", self.deepest_filter, self.deepest.depth["secure"]["SNR_PSF"]["5-sigma"])

    def zeropoint(
            self,
            image_dict: dict,
            output_path: str,
            distance_tolerance: units.Quantity = None,
            snr_min: float = 3.,
            star_class_tolerance: int = 0.95,
            suppress_select: bool = True,
            **kwargs
    ):

        deepest = image_dict[self.filters[0]]
        for fil in self.filters:
            img = image_dict[fil]
            for cat_name in retrieve.photometry_catalogues:
                if cat_name == "gaia":
                    continue
                if cat_name in retrieve.cat_systems and retrieve.cat_systems[cat_name] == "vega":
                    vega = True
                else:
                    vega = False
                fil_path = os.path.join(output_path, fil)
                u.mkdir_check(fil_path)
                if f"in_{cat_name}" in self.field.cats and self.field.cats[f"in_{cat_name}"]:
                    img.zeropoint(
                        cat_path=self.field.get_path(f"cat_csv_{cat_name}"),
                        output_path=os.path.join(fil_path, cat_name),
                        cat_name=cat_name,
                        dist_tol=distance_tolerance,
                        show=False,
                        snr_cut=snr_min,
                        star_class_tol=star_class_tolerance,
                        vega=vega,
                        **kwargs
                    )

            if "preferred_zeropoint" in kwargs and fil in kwargs["preferred_zeropoint"]:
                preferred = kwargs["preferred_zeropoint"][fil]
            else:
                preferred = None

            zeropoint, cat = img.select_zeropoint(suppress_select, preferred=preferred)

            img.estimate_depth(zeropoint_name="best")

            deepest = image.deepest(deepest, img)

        return deepest

    def proc_dual_mode_source_extraction(self, output_dir: str, **kwargs):
        if "image_type" in kwargs and isinstance(kwargs["image_type"], str):
            image_type = kwargs["image_type"]
        else:
            image_type = "final"
        self.dual_mode_source_extraction(output_dir, image_type)

    def dual_mode_source_extraction(self, path: str, image_type: str = "coadded_trimmed"):
        image_dict = self._get_images(image_type=image_type)
        u.mkdir_check(path)
        if self.deepest is None:
            if self.deepest_filter is not None:
                self.deepest = image_dict[self.deepest_filter]
            else:
                raise ValueError(f"deepest for {self.name} is None; make sure you have run photometric_calibration.")
        for fil in image_dict:
            img = image_dict[fil]
            configs = self.source_extractor_config
            img.source_extraction_psf(
                output_dir=path,
                phot_autoparams=f"{configs['kron_factor']},{configs['kron_radius_min']}",
                template=self.deepest
            )

    def proc_get_photometry(self, output_dir: str, **kwargs):
        if "image_type" in kwargs and isinstance(kwargs["image_type"], str):
            image_type = kwargs["image_type"]
        else:
            image_type = "final"
        u.debug_print(2, f"{self}.proc_get_photometry(): image_type ==:", image_type)
        # Run PATH on imaging if we're doing FRB stuff
        if isinstance(self.field, fld.FRBField):
            if 'path_kwargs' in kwargs:
                path_kwargs = kwargs["path_kwargs"]
            else:
                path_kwargs = {
                    "priors": {"U": 0.1},
                    "config": {"radius": 10}
                }
            self.probabilistic_association(image_type=image_type, **path_kwargs)
        self.get_photometry(output_dir, image_type=image_type)

    def probabilistic_association(
            self,
            image_type: str = "final",
            **path_kwargs
    ):
        image_dict = self._get_images(image_type=image_type)
        self.field.frb.load_output_file()
        for fil in image_dict:
            img = image_dict[fil]
            self.field.frb.probabilistic_association(
                img=img,
                **path_kwargs
            )
        self.field.frb.consolidate_candidate_tables()
        for obj in self.field.frb.host_candidates:
            if obj.P_Ox is not None and obj.P_Ox > 0.1:
                self.field.objects.append(obj)

    def get_photometry(
            self,
            path: str,
            image_type: str = "final",
            dual: bool = False,
            match_tolerance: units.Quantity = 1 * units.arcsec,
    ):
        """
        Retrieve photometric properties of key objects and write to disk.

        :param path: Path to which to write the data products.
        :return:
        """
        if not self.quiet:
            print(f"Getting finalised photometry for key objects, in {image_type}.")

        match_tolerance = u.check_quantity(match_tolerance, unit=units.arcsec)

        obs.load_master_objects_table()

        staging_dir = os.path.join(
            p.data_dir,
            "Finalised"
        )

        image_dict = self._get_images(image_type=image_type)
        u.mkdir_check(path)
        # Loop through filters
        for fil in image_dict:
            fil_output_path = os.path.join(path, fil)
            u.mkdir_check(fil_output_path)
            img = image_dict[fil]

            if "secure" not in img.depth:
                img.estimate_depth()
            if not self.quiet:
                print("Getting photometry for", img)

            img.calibrate_magnitudes(zeropoint_name="best", dual=dual, force=True)
            rows = []
            names = []
            separations = []
            ra_target = []
            dec_target = []

            if "SNR_PSF" in img.depth["secure"]:
                depth = img.depth["secure"]["SNR_PSF"][f"5-sigma"]
            else:
                depth = img.depth["secure"]["SNR_AUTO"][f"5-sigma"]

            img.load_data()

            for obj in self.field.objects:
                # obj.load_output_file()
                plt.close()
                # Get nearest Source-Extractor object:
                nearest, separation = img.find_object(obj.position, dual=dual)
                names.append(obj.name)
                rows.append(nearest)
                separations.append(separation.to(units.arcsec))
                ra_target.append(obj.position.ra)
                dec_target.append(obj.position.dec)

                print()
                print(obj.name)
                print("FILTER:", fil)

                if "subtract_background_frames" in self.param_file and self.param_file["subtract_background_frames"]:
                    good_image_path = self.coadded_subtracted[fil].output_file
                else:
                    good_image_path = self.coadded_unprojected[fil].output_file

                if separation > match_tolerance:
                    obj.add_photometry(
                        instrument=self.instrument_name,
                        fil=fil,
                        epoch_name=self.name,
                        mag=-999 * units.mag,
                        mag_err=-999 * units.mag,
                        snr=-999,
                        ellipse_a=-999 * units.arcsec,
                        ellipse_a_err=-999 * units.arcsec,
                        ellipse_b=-999 * units.arcsec,
                        ellipse_b_err=-999 * units.arcsec,
                        ellipse_theta=-999 * units.arcsec,
                        ellipse_theta_err=-999 * units.arcsec,
                        ra=-999 * units.deg,
                        ra_err=-999 * units.deg,
                        dec=-999 * units.deg,
                        dec_err=-999 * units.deg,
                        kron_radius=-999.,
                        separation_from_given=separation,
                        epoch_date=self.date_str(),
                        class_star=-999.,
                        spread_model=-999.,
                        spread_model_err=-999.,
                        class_flag=-999,
                        mag_psf=-999. * units.mag,
                        mag_psf_err=-999. * units.mag,
                        snr_psf=-999.,
                        image_depth=depth,
                        image_path=img.path,
                        good_image_path=good_image_path,
                        do_mask=img.mask_nearby()
                    )
                    print(f"No object detected at position.")
                    print()
                else:
                    u.debug_print(2, "ImagingImage.get_photometry(): nearest.colnames ==", nearest.colnames)
                    err = nearest[f'MAGERR_AUTO_ZP_best']
                    if not self.quiet:
                        print(f"MAG_AUTO = {nearest['MAG_AUTO_ZP_best']} +/- {err}")
                        print(f"A = {nearest['A_WORLD'].to(units.arcsec)}; B = {nearest['B_WORLD'].to(units.arcsec)}")
                    img.plot_source_extractor_object(
                        nearest,
                        output=os.path.join(fil_output_path, f"{obj.name_filesys}.png"),
                        show=False,
                        title=f"{obj.name}, {fil}-band, {nearest['MAG_AUTO_ZP_best'].round(3).value} ± {err.round(3)}",
                        find=obj.position
                    )
                    obj.cat_row = nearest

                    if "MAG_PSF_ZP_best" in nearest.colnames:
                        mag_psf = nearest["MAG_PSF_ZP_best"]
                        mag_psf_err = nearest["MAGERR_PSF_ZP_best"]
                        snr_psf = nearest["FLUX_PSF"] / nearest["FLUXERR_PSF"]
                        spread_model = nearest["SPREAD_MODEL"]
                        spread_model_err = nearest["SPREADERR_MODEL"]
                        class_flag = nearest["CLASS_FLAG"]
                    else:
                        mag_psf = -999.0 * units.mag
                        mag_psf_err = -999.0 * units.mag
                        snr_psf = -999.0
                        spread_model = -999.0
                        spread_model_err = -999.0
                        class_flag = -999

                    obj.add_photometry(
                        instrument=self.instrument_name,
                        fil=fil,
                        epoch_name=self.name,
                        mag=nearest['MAG_AUTO_ZP_best'],
                        mag_err=err,
                        snr=nearest['SNR_AUTO'],
                        ellipse_a=nearest['A_WORLD'],
                        ellipse_a_err=nearest["ERRA_WORLD"],
                        ellipse_b=nearest['B_WORLD'],
                        ellipse_b_err=nearest["ERRB_WORLD"],
                        ellipse_theta=nearest['THETA_WORLD'],
                        ellipse_theta_err=nearest['ERRTHETA_WORLD'],
                        ra=nearest['RA'],
                        ra_err=np.sqrt(nearest["ERRX2_WORLD"]),
                        dec=nearest['DEC'],
                        dec_err=np.sqrt(nearest["ERRY2_WORLD"]),
                        kron_radius=nearest["KRON_RADIUS"],
                        separation_from_given=separation,
                        epoch_date=self.date_str(),
                        class_star=nearest["CLASS_STAR"],
                        spread_model=spread_model,
                        spread_model_err=spread_model_err,
                        class_flag=class_flag,
                        mag_psf=mag_psf,
                        mag_psf_err=mag_psf_err,
                        snr_psf=snr_psf,
                        image_depth=depth,
                        image_path=img.path,
                        good_image_path=good_image_path,
                        do_mask=img.mask_nearby()
                    )

                    if isinstance(self.field, fld.FRBField):
                        frames = [
                            img.nice_frame(row=obj.cat_row),
                            10 * units.arcsec,
                            20 * units.arcsec,
                            40 * units.arcsec,
                        ]
                        if "frame" in obj.plotting_params and obj.plotting_params["frame"] is not None:
                            frames.append(obj.plotting_params["frame"])

                        normalize_kwargs = {}
                        if fil in obj.plotting_params:
                            if "normalize" in obj.plotting_params[fil]:
                                normalize_kwargs = obj.plotting_params[fil]["normalize"]

                        for frame in frames:
                            for stretch in ["log", "sqrt"]:
                                print(f"\nPlotting {frame=}, {stretch=}")
                                normalize_kwargs["stretch"] = stretch
                                centre = obj.position_from_cat_row()

                                fig = plt.figure(figsize=(6, 5))
                                ax, fig, _ = self.field.plot_host(
                                    img=img,
                                    fig=fig,
                                    centre=centre,
                                    show_frb=True,
                                    frame=frame,
                                    imshow_kwargs={
                                        "cmap": "plasma"
                                    },
                                    frb_kwargs={
                                        "edgecolor": "black"
                                    },
                                    normalize_kwargs=normalize_kwargs
                                )
                                output_path = os.path.join(
                                    fil_output_path,
                                    f"{obj.name_filesys}_{fil}_{str(frame).replace(' ', '-')}_{stretch}")
                                name = obj.name
                                img.extract_filter()
                                if img.filter is None:
                                    f_name = fil
                                else:
                                    f_name = img.filter.nice_name()
                                ax.set_title(f"{name}, {f_name}")
                                fig.savefig(output_path + ".pdf")
                                fig.savefig(output_path + ".png")
                                ax.clear()
                                fig.clf()
                                plt.close("all")
                                pl.latex_off()

            tbl = table.vstack(rows)
            tbl.add_column(names, name="NAME")
            tbl.add_column(separations, name="OFFSET_FROM_TARGET")
            tbl.add_column(ra_target, name="RA_TARGET")
            tbl.add_column(dec_target, name="DEC_TARGET")

            tbl.write(
                os.path.join(fil_output_path, f"{self.field.name}_{self.name}_{fil}.ecsv"),
                format="ascii.ecsv",
                overwrite=True
            )
            tbl.write(
                os.path.join(fil_output_path, f"{self.field.name}_{self.name}_{fil}.csv"),
                format="ascii.csv",
                overwrite=True
            )

        for fil in self.coadded_unprojected:

            img = self.coadded_unprojected[fil]
            img_projected = image_dict[fil]

            if img is None:
                continue

            if isinstance(self.instrument, inst.Instrument):
                inst_name = self.instrument.nice_name().replace('/', '-')
            else:
                inst_name = self.instrument_name

            if self.combined_epoch:
                date = "combined"
            else:
                date = self.date_str()

            nice_name = f"{self.field.name}_{inst_name}_{fil.replace('_', '-')}_{date}.fits"

            astm_rms = img_projected.extract_astrometry_err().value
            psf_fwhm = img_projected.extract_header_item(key="PSF_FWHM")
            psf_fwhm_err = img_projected.extract_header_item(key="PSF_FWHM_ERR")

            if img != img_projected:
                img.set_header_items(
                    items={
                        # 'ASTM_RMS': astm_rms,
                        # 'RA_RMS': img_projected.extract_header_item(key="RA_RMS"),
                        # 'DEC_RMS': img_projected.extract_header_item(key="DEC_RMS"),
                        # 'PSF_FWHM': psf_fwhm,
                        # 'PSF_FWHM_ERR': psf_fwhm_err,
                        'ZP': img_projected.extract_header_item(key="ZP"),
                        'ZP_ERR': img_projected.extract_header_item(key="ZP_ERR"),
                        'ZPCAT': str(img_projected.extract_header_item(key="ZPCAT"))
                    },
                    write=True,
                )
            if self.coadded_subtracted[fil] is not None:
                img.set_header_items(
                    items={
                        'ASTM_RMS': astm_rms,
                        'RA_RMS': img_projected.extract_header_item(key="RA_RMS"),
                        'DEC_RMS': img_projected.extract_header_item(key="DEC_RMS"),
                        'PSF_FWHM': psf_fwhm,
                        'PSF_FWHM_ERR': psf_fwhm_err,
                        'ZP': img_projected.extract_header_item(key="ZP"),
                        'ZP_ERR': img_projected.extract_header_item(key="ZP_ERR"),
                        'ZPCAT': str(img_projected.extract_header_item(key="ZPCAT"))
                    },
                    write=True,
                )

            img.copy_with_outputs(os.path.join(
                self.data_path,
                nice_name)
            )

            img.copy_with_outputs(
                os.path.join(
                    staging_dir,
                    nice_name
                )
            )

            if isinstance(self.field.survey, survey.Survey):
                refined_path = self.field.survey.refined_stage_path

                if refined_path is not None:
                    img.copy_with_outputs(
                        os.path.join(
                            refined_path,
                            nice_name
                        )
                    )

        self.push_to_table()

        for obj in self.field.objects:
            obj.update_output_file()
            # obj.push_to_table(select=True)
            # obj.write_plot_photometry()

    def proc_get_photometry_all(self, output_dir: str, **kwargs):
        if "image_type" in kwargs and isinstance(kwargs["image_type"], str):
            image_type = kwargs["image_type"]
        else:
            image_type = "final"
        self.get_photometry_all(output_dir, image_type=image_type)

    def get_photometry_all(
            self, path: str,
            image_type: str = "coadded_trimmed",
            dual: bool = False
    ):
        obs.load_master_all_objects_table()
        image_dict = self._get_images(image_type=image_type)
        u.mkdir_check(path)
        # Loop through filters
        for fil in image_dict:
            fil_output_path = os.path.join(path, fil)
            u.mkdir_check(fil_output_path)
            img = image_dict[fil]
            img.push_source_cat(dual=dual)

    def astrometry_diagnostics(
            self,
            images: dict = None,
            reference_cat: table.QTable = None,
            offset_tolerance: units.Quantity = 0.5 * units.arcsec
    ):

        if images is None:
            images = self._get_images("final")
        elif isinstance(images, str):
            images = self._get_images(images)

        if reference_cat is None:
            reference_cat = self.epoch_gaia_catalogue()

        for fil, img in images.items():
            print("Attempting astrometry diagnostics for", img.name)
            img.source_cat.load_table()
            stats = -99.
            while not isinstance(stats, dict):
                stats = img.astrometry_diagnostics(
                    reference_cat=reference_cat,
                    local_coord=self.field.centre_coords,
                    offset_tolerance=offset_tolerance
                )
                offset_tolerance += 0.5 * units.arcsec
            stats["file_path"] = img.path
            self.astrometry_stats[fil] = stats

            # if fil in self.coadded_subtracted and self.coadded_subtracted[fil] is not None:
            #     self.coadded_subtracted[fil].astrometry_stats

        self.add_log(
            "Ran astrometry diagnostics.",
            method=self.astrometry_diagnostics,
        )

        self.update_output_file()
        return self.astrometry_stats

    def psf_diagnostics(
            self,
            images: dict = None
    ):
        if images is None:
            images = self._get_images("final")

        for fil in images:
            img = images[fil]
            if not self.quiet:
                print(f"Performing PSF measurements on {img}...")
            self.psf_stats[fil], _ = img.psf_diagnostics()
            self.psf_stats[fil]["file_path"] = img.path

        self.update_output_file()
        return self.psf_stats

    def _get_images(self, image_type: str) -> Dict[str, image.CoaddedImage]:
        """
        A helper method for finding the desired coadded image dictionary.
        :param image_type: "trimmed", "coadded", "unprojected" or "astrometry"
        :return: dict with filter names as keys and CoaddedImage objects as values.
        """

        if image_type in ["final", "coadded_final"]:
            if self.coadded_final is not None:
                image_type = self.coadded_final
            else:
                raise ValueError("coadded_final has not been set.")

        if image_type in ["coadded_trimmed", "trimmed"]:
            image_dict = self.coadded_trimmed
        elif image_type == "coadded":
            image_dict = self.coadded
        elif image_type in ["coadded_unprojected", "unprojected"]:
            image_dict = self.coadded_unprojected
        elif image_type in ["coadded_subtracted", "subtracted"]:
            image_dict = self.coadded_subtracted
        elif image_type in ["coadded_astrometry", "astrometry"]:
            image_dict = self.coadded_astrometry
        else:
            raise ValueError(f"Images type '{image_type}' not recognised.")
        return image_dict

    def _get_frames(self, frame_type: str) -> Dict[str, List[image.ImagingImage]]:
        """
        A helper method for finding the desired frame dictionary
        :param frame_type: "science", "reduced", "trimmed", "normalised", "registered", "astrometry" or "diagnosed"
        :return: dictionary, with filter names as keys, and lists of frame Image objects as keys.
        """
        if frame_type == "final":
            if self.frames_final is not None:
                frame_type = self.frames_final
            else:
                raise ValueError("frames_final has not been set.")

        if frame_type in ("science", "frames_science"):
            image_dict = self.frames_science
        elif frame_type in ("reduced", "frames_reduced"):
            image_dict = self.frames_reduced
        elif frame_type in ("trimmed", "frames_trimmed"):
            image_dict = self.frames_trimmed
        elif frame_type in ("normalised", "frames_normalised"):
            image_dict = self.frames_normalised
        elif frame_type in ("subtracted", "frames_substracted"):
            image_dict = self.frames_subtracted
        elif frame_type in ("registered", "frames_registered"):
            image_dict = self.frames_registered
        elif frame_type in ("astrometry", "frames_astrometry"):
            image_dict = self.frames_astrometry
        elif frame_type == ("diagnosed", "frames_diagnosed"):
            image_dict = self.frames_diagnosed
        else:
            raise ValueError(f"Frame type '{frame_type}' not recognised.")

        return image_dict

    def guess_data_path(self):
        if self.data_path is None and self.field is not None and self.field.data_path is not None and \
                self.instrument_name is not None and self.date is not None:
            self.data_path = self.build_data_path_absolute(
                field=self.field,
                instrument_name=self.instrument_name,
                date=self.date,
                name=self.name
            )
        return self.data_path

    def _output_dict(self):
        output_dict = super()._output_dict()
        if self.deepest is not None:
            deepest = self.deepest.path
        else:
            deepest = None

        output_dict.update({
            "filters": self.filters,
            "deepest": deepest,
            "deepest_filter": self.deepest_filter,
            "coadded": _output_img_dict_single(self.coadded),
            "coadded_final": self.coadded_final,
            "coadded_trimmed": _output_img_dict_single(self.coadded_trimmed),
            "coadded_unprojected": _output_img_dict_single(self.coadded_unprojected),
            "coadded_astrometry": _output_img_dict_single(self.coadded_astrometry),
            "coadded_subtracted": _output_img_dict_single(self.coadded_subtracted),
            "std_pointings": self.std_pointings,
            "frames_final": self.frames_final,
            "frames_raw": _output_img_list(self.frames_raw),
            "frames_reduced": _output_img_dict_list(self.frames_reduced),
            "frames_normalised": _output_img_dict_list(self.frames_normalised),
            "frames_subtracted": _output_img_dict_list(self.frames_subtracted),
            "frames_registered": _output_img_dict_list(self.frames_registered),
            "frames_astrometry": _output_img_dict_list(self.frames_astrometry),
            "frames_diagnosed": _output_img_dict_list(self.frames_diagnosed),
            "exp_time_mean": self.exp_time_mean,
            "exp_time_err": self.exp_time_err,
            "airmass_mean": self.airmass_mean,
            "airmass_err": self.airmass_err,
            "astrometry_indices": self.astrometry_indices,
            "astrometry_successful": self.astrometry_successful,
            "astrometry_stats": self.astrometry_stats,
            "psf_stats": self.psf_stats
        })
        return output_dict

    def load_output_file(self, **kwargs):
        outputs = super().load_output_file(**kwargs)
        if isinstance(outputs, dict):
            frame_cls = image.ImagingImage.select_child_class(instrument_name=self.instrument_name, mode='imaging')
            coadd_class = image.CoaddedImage.select_child_class(instrument_name=self.instrument_name, mode='imaging')
            if self.date is None:
                if "date" in outputs:
                    self.date = outputs["date"]
            if "filters" in outputs:
                self.filters = outputs["filters"]
            if self._check_output_file_path("deepest", outputs):
                self.deepest = image.from_path(
                    path=outputs["deepest"],
                    cls=coadd_class
                )
            if "deepest_filter" in outputs:
                self.deepest_filter = outputs["deepest_filter"]
            if "exp_time_mean" in outputs:
                self.exp_time_mean = outputs["exp_time_mean"]
            if "exp_time_err" in outputs:
                self.exp_time_err = outputs["exp_time_err"]
            if "airmass_mean" in outputs:
                self.airmass_mean = outputs["airmass_mean"]
            if "airmass_err" in outputs:
                self.airmass_err = outputs["airmass_err"]
            if "psf_stats" in outputs:
                self.psf_stats = outputs["psf_stats"]
            if "astrometry_stats" in outputs:
                self.astrometry_stats = outputs["astrometry_stats"]
            if "astrometry_successful" in outputs:
                self.astrometry_successful = outputs["astrometry_successful"]
            if "astrometry_indices" in outputs:
                self.astrometry_indices = outputs["astrometry_indices"]
            if "frames_raw" in outputs:
                for frame in set(outputs["frames_raw"]):
                    if os.path.isfile(frame):
                        self.add_frame_raw(frame=frame)
            if "frames_reduced" in outputs:
                for fil in outputs["frames_reduced"]:
                    if outputs["frames_reduced"][fil] is not None:
                        for frame in set(outputs["frames_reduced"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_reduced(frame=frame)
            if "frames_normalised" in outputs:
                for fil in outputs["frames_normalised"]:
                    if outputs["frames_normalised"][fil] is not None:
                        for frame in set(outputs["frames_normalised"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_normalised(frame=frame)
            if "frames_subtracted" in outputs:
                for fil in outputs["frames_subtracted"]:
                    if outputs["frames_subtracted"][fil] is not None:
                        for frame in set(outputs["frames_subtracted"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_subtracted(frame=frame)

            if "frames_registered" in outputs:
                for fil in outputs["frames_registered"]:
                    if outputs["frames_registered"][fil] is not None:
                        for frame in set(outputs["frames_registered"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_registered(frame=frame)
            if "frames_astrometry" in outputs:
                for fil in outputs["frames_astrometry"]:
                    if outputs["frames_astrometry"][fil] is not None:
                        for frame in set(outputs["frames_astrometry"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_astrometry(frame=frame)
            if "frames_diagnosed" in outputs:
                for fil in outputs["frames_diagnosed"]:
                    if outputs["frames_diagnosed"][fil] is not None:
                        for frame in set(outputs["frames_diagnosed"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_diagnosed(frame=frame)
            if "coadded" in outputs:
                for fil in outputs["coadded"]:
                    if outputs["coadded"][fil] is not None:
                        self.add_coadded_image(img=outputs["coadded"][fil], key=fil, **kwargs)
            if "coadded_subtracted" in outputs:
                for fil in outputs["coadded_subtracted"]:
                    if outputs["coadded_subtracted"][fil] is not None:
                        self.add_coadded_subtracted_image(img=outputs["coadded_subtracted"][fil], key=fil, **kwargs)
            if "coadded_trimmed" in outputs:
                for fil in outputs["coadded_trimmed"]:
                    if outputs["coadded_trimmed"][fil] is not None:
                        u.debug_print(1, f"Attempting to load coadded_trimmed[{fil}]")
                        self.add_coadded_trimmed_image(img=outputs["coadded_trimmed"][fil], key=fil, **kwargs)
            if "coadded_unprojected" in outputs:
                for fil in outputs["coadded_unprojected"]:
                    if outputs["coadded_unprojected"][fil] is not None:
                        u.debug_print(1, f"Attempting to load coadded_unprojected[{fil}]")
                        self.add_coadded_unprojected_image(img=outputs["coadded_unprojected"][fil], key=fil, **kwargs)
            if "coadded_astrometry" in outputs:
                for fil in outputs["coadded_astrometry"]:
                    if outputs["coadded_astrometry"][fil] is not None:
                        u.debug_print(1, f"Attempting to load coadded_astrometry[{fil}]")
                        self.add_coadded_astrometry_image(img=outputs["coadded_astrometry"][fil], key=fil, **kwargs)
            if "std_pointings" in outputs:
                self.std_pointings = outputs["std_pointings"]

        return outputs

    def generate_astrometry_indices(
            self,
            cat_name="gaia",
            correct_to_epoch: bool = True,
            force: bool = False
    ):
        """
        Generates astrometry indices using astrometry.net and the specified catalogue, unless they have been generated
        before; in which case it simply copies them to the main index directory (overwriting those of other epochs there).
        :param cat_name:
        :param correct_to_epoch:
        :param force:
        :return:
        """
        if not isinstance(self.field, fld.Field):
            raise ValueError("field has not been set for this observation.")

        if force or not self.astrometry_indices:
            epoch_index_path = os.path.join(self.data_path, "astrometry_indices")
            self.field.retrieve_catalogue(cat_name=cat_name)

            csv_path = self.field.get_path(f"cat_csv_{cat_name}")

            if cat_name == "gaia":
                cat = self.epoch_gaia_catalogue(correct_to_epoch=correct_to_epoch)
            else:
                cat = retrieve.load_catalogue(
                    cat_name=cat_name,
                    cat=csv_path
                )

            unique_id_prefix = int(
                f"{abs(int(self.field.centre_coords.ra.value))}{abs(int(self.field.centre_coords.dec.value))}")

            self.astrometry_indices = astm.generate_astrometry_indices(
                cat_name=cat_name,
                cat=cat,
                output_file_prefix=f"{cat_name}_index_{self.field.name}",
                index_output_dir=epoch_index_path,
                fits_cat_output=csv_path.replace(".csv", ".fits"),
                p_lower=-1,
                p_upper=2,
                unique_id_prefix=unique_id_prefix,
                add_path=False
            )
        index_path = os.path.join(config["top_data_dir"], "astrometry_index_files")
        u.mkdir_check(index_path)
        cat_index_path = os.path.join(index_path, cat_name)
        astm.astrometry_net.add_index_directory(cat_index_path)
        for index_path in self.astrometry_indices:
            shutil.copy(index_path, cat_index_path)
        self.update_output_file()
        return self.astrometry_indices

    def epoch_gaia_catalogue(
            self,
            correct_to_epoch: bool = True
    ):
        if correct_to_epoch:
            if self.date is None:
                raise ValueError(f"{self}.date not set; needed to correct Gaia cat to epoch.")
            self.gaia_catalogue = astm.correct_gaia_to_epoch(
                self.field.get_path(f"cat_csv_gaia"),
                new_epoch=self.date
            )
        else:
            self.gaia_catalogue = astm.load_catalogue(cat_name="gaia", cat=self.field.get_path(f"cat_csv_gaia"))
        return self.gaia_catalogue

    def _check_frame(self, frame: Union[image.ImagingImage, str], frame_type: str):
        if isinstance(frame, str):
            if os.path.isfile(frame):
                cls = image.ImagingImage.select_child_class(instrument_name=self.instrument_name)
                u.debug_print(2, f"{cls} {self.instrument_name}")
                frame = image.from_path(
                    path=frame,
                    frame_type=frame_type,
                    cls=cls
                )
            else:
                u.debug_print(2, f"File {frame} not found.")
                return None, None
        fil = frame.extract_filter()
        frame.epoch = self

        return frame, fil

    def _add_frame(self, frame: Union[image.ImagingImage, str], frames_dict: dict, frame_type: str):
        frame, fil = self._check_frame(frame=frame, frame_type=frame_type)
        if frame is None:
            return None
        if self.check_filter(fil=fil) and frame not in frames_dict[fil]:
            frames_dict[fil].append(frame)
        return frame

    def add_frame_raw(self, frame: Union[image.ImagingImage, str]):
        frame, fil = self._check_frame(frame=frame, frame_type="raw")
        self.check_filter(fil)
        if frame is None:
            return None
        if frame not in self.frames_raw:
            self.frames_raw.append(frame)
        self.sort_frame(frame, sort_key=fil)
        return frame

    def add_frame_reduced(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_reduced, frame_type="reduced")

    def add_frame_trimmed(self, frame: image.ImagingImage):
        self._add_frame(frame=frame, frames_dict=self.frames_trimmed, frame_type="reduced")

    def add_frame_subtracted(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_subtracted, frame_type="subtracted")

    def add_frame_registered(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_registered, frame_type="registered")

    def add_frame_astrometry(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_astrometry, frame_type="astrometry")

    def add_frame_diagnosed(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_diagnosed, frame_type="diagnosed")

    def add_frame_normalised(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_normalised, frame_type="reduced")

    def add_coadded_trimmed_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_trimmed)

    def add_coadded_unprojected_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_unprojected)

    def add_coadded_subtracted_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_subtracted)

    def add_coadded_astrometry_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_astrometry)

    def check_filter(self, fil: str):
        """
        If a filter name is not present in the various lists and dictionaries that use it, adds it.
        :param fil:
        :return: False if None, True if not.
        """
        if fil not in (None, "", " "):
            if fil not in self.filters:
                if not self.quiet:
                    print(f"Adding {fil} to filter list")
                self.filters.append(fil)
            if fil not in self.astrometry_successful:
                self.astrometry_successful[fil] = {}
            if fil not in self.frames_standard:
                if isinstance(self.frames_standard, dict):
                    self.frames_standard[fil] = []
            if fil not in self.frames_flat:
                if isinstance(self.frames_flat, dict):
                    self.frames_flat[fil] = []
            if fil not in self.frames_science:
                if isinstance(self.frames_science, dict):
                    self.frames_science[fil] = []
            if fil not in self.frames_reduced:
                if isinstance(self.frames_reduced, dict):
                    self.frames_reduced[fil] = []
            if fil not in self.frames_normalised:
                if isinstance(self.frames_normalised, dict):
                    self.frames_normalised[fil] = []
            if fil not in self.frames_subtracted:
                if isinstance(self.frames_subtracted, dict):
                    self.frames_subtracted[fil] = []
            if fil not in self.frames_registered:
                if isinstance(self.frames_registered, dict):
                    self.frames_registered[fil] = []
            if fil not in self.frames_diagnosed:
                if isinstance(self.frames_diagnosed, dict):
                    self.frames_diagnosed[fil] = []
            if fil not in self.frames_astrometry:
                self.frames_astrometry[fil] = []
            if fil not in self.coadded:
                self.coadded[fil] = None
            if fil not in self.coadded_trimmed:
                self.coadded_trimmed[fil] = None
            if fil not in self.coadded_unprojected:
                self.coadded_unprojected[fil] = None
            if fil not in self.coadded_subtracted:
                self.coadded_subtracted[fil] = None
            if fil not in self.coadded_astrometry:
                self.coadded_astrometry[fil] = None
            if fil not in self.exp_time_mean:
                self.exp_time_mean[fil] = None
            if fil not in self.exp_time_err:
                self.exp_time_err[fil] = None
            if fil not in self.airmass_mean:
                self.airmass_mean[fil] = None
            if fil not in self.airmass_err:
                self.airmass_err[fil] = None
            if fil not in self.astrometry_stats:
                self.astrometry_stats[fil] = {}
            if fil not in self.frame_stats:
                self.frame_stats[fil] = {}
            return True
        else:
            return False

    def plot_object(
            self, img: str,
            fil: str,
            fig: plt.Figure,
            centre: SkyCoord,
            frame: units.Quantity = 30 * units.pix,
            n: int = 1, n_x: int = 1, n_y: int = 1,
            cmap: str = 'viridis', show_cbar: bool = False,
            stretch: str = 'sqrt',
            vmin: float = None,
            vmax: float = None,
            show_grid: bool = False,
            ticks: int = None, interval: str = 'minmax',
            show_coords: bool = True,
            font_size: int = 12,
            reverse_y=False,
            **kwargs):
        if img == "coadded":
            u.debug_print(1, self.name, type(self))
            u.debug_print(1, self.coadded)
            to_plot = self.coadded[fil]
        else:
            raise ValueError(f"img type {img} not recognised.")

        u.debug_print(1, f"PIXEL SCALE: {to_plot.extract_pixel_scale()}")

        subplot, hdu_cut = to_plot.plot_subimage(
            fig=fig, frame=frame,
            centre=centre,
            n=n, n_x=n_x, n_y=n_y,
            cmap=cmap, show_cbar=show_cbar, stretch=stretch,
            vmin=vmin, vmax=vmax,
            show_grid=show_grid,
            ticks=ticks, interval=interval,
            show_coords=show_coords,
            font_size=font_size,
            reverse_y=reverse_y,
            **kwargs
        )
        return subplot, hdu_cut

    def push_to_table(self):

        obs.load_master_imaging_table()

        # frames = self._get_frames("final")
        coadded = self._get_images("final")

        for fil in self.filters:
            img = coadded[fil]

            inttime = coadded[fil].extract_header_item("INTTIME") * units.second
            n_frames = self.n_frames(fil)
            if self.exp_time_mean[fil] is None:
                final_frames = self._get_frames("final")
                exp_times = list(map(lambda frame: frame.extract_exposure_time().value, final_frames[fil]))
                self.exp_time_mean[fil] = np.mean(exp_times) * units.s
            frame_exp_time = self.exp_time_mean[fil].round()

            if "SNR_PSF" in img.depth["secure"]:
                depth = img.depth["secure"]["SNR_PSF"][f"5-sigma"]
            else:
                depth = img.depth["secure"]["SNR_AUTO"][f"5-sigma"]

            entry = {
                "field_name": self.field.name,
                "epoch_name": self.name,
                "date_utc": self.date_str(),
                "mjd": self.mjd() * units.day,
                "instrument": self.instrument_name,
                "filter_name": fil,
                "filter_lambda_eff": self.instrument.filters[fil].lambda_eff.to(units.Angstrom).round(3),
                "n_frames": n_frames,
                "n_frames_included": coadded[fil].extract_ncombine(),
                "frame_exp_time": frame_exp_time,
                "total_exp_time": n_frames * frame_exp_time,
                "total_exp_time_included": inttime,
                "psf_fwhm": self.psf_stats[fil]["gauss"]["fwhm_median"],
                "program_id": str(self.program_id),
                "zeropoint": coadded[fil].zeropoint_best["zeropoint_img"],
                "zeropoint_err": coadded[fil].zeropoint_best["zeropoint_img_err"],
                "zeropoint_source": coadded[fil].zeropoint_best["catalogue"],
                "last_processed": Time.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "depth": depth
            }

            if isinstance(self.field, fld.FRBField) and self.field.frb.tns_name is not None:
                entry["transient_tns_name"] = self.field.frb.tns_name

            obs.add_epoch(
                epoch_name=self.name,
                fil=fil,
                entry=entry
            )

        obs.write_master_imaging_table()

    @classmethod
    def from_params(
            cls,
            name: str,
            instrument: str,
            field: Union['fld.Field', str] = None,
            old_format: bool = False,
            quiet: bool = False
    ):
        if name in active_epochs:
            return active_epochs[name]
        instrument = instrument.lower()
        field_name, field = cls._from_params_setup(name=name, field=field)
        if old_format:
            instrument = instrument.split("-")[-1]
            path = os.path.join(p.param_dir, f"epochs_{instrument}", name)
        else:
            path = cls.build_param_path(
                instrument_name=instrument,
                field_name=field_name,
                epoch_name=name)
        return cls.from_file(param_file=path, field=field, quiet=quiet)

    @classmethod
    def build_param_path(cls, instrument_name: str, field_name: str, epoch_name: str):
        path = u.mkdir_check_args(p.param_dir, "fields", field_name, "imaging", instrument_name)
        return os.path.join(path, f"{epoch_name}.yaml")

    @classmethod
    def build_data_path_absolute(
            cls,
            field: 'fld.Field',
            instrument_name: str,
            name: str,
            date: Time = None
    ):
        if date is not None:
            name_str = f"{date.isot}-{name}"
        else:
            name_str = name

        return u.mkdir_check_args(field.data_path, "imaging", instrument_name, name_str)

    @classmethod
    def from_file(
            cls,
            param_file: Union[str, dict],
            old_format: bool = False,
            field: 'fld.Field' = None,
            quiet: bool = False
    ):
        if not quiet:
            print("Initializing epoch...")

        name, param_file, param_dict = p.params_init(param_file)

        if param_dict is None:
            raise FileNotFoundError(f"There is no param file at {param_file}")

        pdict_backup = param_dict.copy()

        if old_format:
            instrument = "vlt-fors2"
        else:
            instrument = param_dict.pop("instrument").lower()

        fld_from_dict = param_dict.pop("field")
        if field is None:
            field = fld_from_dict
        # else:
        #     param_dict.pop("field")

        sub_cls = cls.select_child_class(instrument=instrument)
        u.debug_print(1, sub_cls)
        if sub_cls is ImagingEpoch:
            return cls(
                name=name,
                field=field,
                param_path=param_file,
                data_path=os.path.join(config["top_data_dir"], param_dict.pop('data_path')),
                instrument=instrument,
                date=param_dict.pop('date'),
                program_id=param_dict.pop("program_id"),
                target=param_dict.pop("target"),
                source_extractor_config=param_dict.pop('sextractor'),
                quiet=quiet,
                **param_dict
            )
        elif sub_cls is FORS2ImagingEpoch:
            return sub_cls.from_file(param_dict, name=name, old_format=old_format, field=field)
        else:
            return sub_cls.from_file(pdict_backup, name=name, field=field)

    @classmethod
    def default_params(cls):
        default_params = super().default_params()

        default_params.update({
            "astrometry":
                {"tweak": True
                 },
            "coadd":
                {"frames": "astrometry"},
            "sextractor":
                {"dual_mode": False,
                 "threshold": 1.5,
                 "kron_factor": 2.5,
                 "kron_radius_min": 3.5
                 },
            # "background_subtraction":
            #     {"renormalise_centre": objects.position_dictionary.copy(),
            #      "test_synths":
            #          [{"position": objects.position_dictionary.copy(),
            #            "mags": {}
            #            }]
            #
            #      },
        })

        return default_params

    @classmethod
    def select_child_class(cls, instrument: str):
        instrument = instrument.lower()
        if instrument == "vlt-fors2":
            child_class = FORS2ImagingEpoch
        elif instrument == "vlt-hawki":
            child_class = HAWKIImagingEpoch
        elif instrument == "panstarrs1":
            child_class = PanSTARRS1ImagingEpoch
        elif instrument == "gs-aoi":
            child_class = GSAOIImagingEpoch
        elif instrument in ["hst-wfc3_ir", "hst-wfc3_uvis2"]:
            child_class = HubbleImagingEpoch
        elif instrument == "decam":
            child_class = DESEpoch
        elif instrument in p.instruments_imaging:
            child_class = ImagingEpoch
        else:
            raise ValueError(f"Unrecognised instrument {instrument}")
        u.debug_print(2, f"field.select_child_class(): instrument ==", instrument, "child_class ==", child_class)
        return child_class


class FORS2StandardEpoch(StandardEpoch, ImagingEpoch):
    frame_class = image.FORS2Image
    coadded_class = image.FORS2CoaddedImage
    instrument_name = "vlt-fors2"

    def source_extraction(self, output_dir: str, do_diagnostics: bool = True, **kwargs):
        for fil in self.frames_reduced:
            for img in self.frames_reduced[fil]:
                img.remove_extra_extensions()
                configs = self.source_extractor_config

                img.psfex_path = None
                img.source_extraction_psf(
                    output_dir=output_dir,
                    phot_autoparams=f"3.5,1.0"
                )

    def photometric_calibration(
            self,
            output_path: str = None,
            **kwargs
    ):
        zeropoints = {}

        if output_path is None:
            output_path = os.path.join(self.data_path, "photometric_calibration")

        u.mkdir_check_nested(output_path)

        self.source_extraction(
            output_dir=output_path,
            do_diagnostics=False
        )

        self.zeropoint(
            image_dict=self.frames_reduced,
            output_path=output_path,
            suppress_select=True,
            zp_dict=zeropoints,
            **kwargs
        )

    def zeropoint(
            self,
            image_dict: dict,
            output_path: str,
            distance_tolerance: units.Quantity = None,
            snr_min: float = 3.,
            star_class_tolerance: float = 0.9,
            suppress_select: bool = True,
            **kwargs
    ):

        if "zp_dict" in kwargs:
            zp_dict = kwargs["zp_dict"]
        else:
            zp_dict = {}

        zp_dict[1] = {}
        zp_dict[2] = {}
        for fil in self.filters:
            for img in image_dict[fil]:
                cats = retrieve.photometry_catalogues
                # cats.append("eso_calib_cats")
                for cat_name in cats:
                    if cat_name == "gaia":
                        continue
                    if cat_name in retrieve.cat_systems and retrieve.cat_systems[cat_name] == "vega":
                        vega = True
                    else:
                        vega = False
                    fil_path = os.path.join(output_path, fil)
                    u.mkdir_check_nested(fil_path, remove_last=False)
                    if f"in_{cat_name}" in self.field.cats and self.field.cats[f"in_{cat_name}"]:
                        zp = img.zeropoint(
                            cat_path=self.field.get_path(f"cat_csv_{cat_name}"),
                            output_path=os.path.join(fil_path, cat_name),
                            cat_name=cat_name,
                            dist_tol=distance_tolerance,
                            show=False,
                            snr_cut=snr_min,
                            star_class_tol=star_class_tolerance,
                            iterate_uncertainty=True,
                            vega=vega
                        )

                        chip = img.extract_chip_number()

                if "preferred_zeropoint" in kwargs and fil in kwargs["preferred_zeropoint"]:
                    preferred = kwargs["preferred_zeropoint"][fil]
                else:
                    preferred = None

                img.select_zeropoint(suppress_select, preferred=preferred)


class GSAOIImagingEpoch(ImagingEpoch):
    """
    This class works a little differently to the other epochs; instead of keeping track of the files internally, we let
    DRAGONS do that for us. Thus, many of the dictionaries and lists of files used in other Epoch classes
    will be empty even if the files are actually being tracked correctly. See eg science_table instead.
    """
    instrument_name = "gs-aoi"
    frame_class = image.GSAOIImage
    coadded_class = image.GSAOIImage

    def __init__(
            self,
            name: str = None,
            field: Union[str, 'fld.Field'] = None,
            param_path: str = None,
            data_path: str = None,
            instrument: str = None,
            date: Union[str, Time] = None,
            program_id: str = None,
            target: str = None,
            source_extractor_config: dict = None,
            **kwargs
    ):
        super().__init__(
            name=name,
            field=field,
            param_path=param_path,
            data_path=data_path,
            instrument=instrument,
            date=date,
            program_id=program_id,
            target=target,
            source_extractor_config=source_extractor_config)
        self.science_table = None
        self.flats_lists = {}
        self.std_lists = {}

        self.load_output_file(mode="imaging")

    @classmethod
    def stages(cls):
        stages_super = super().stages()
        stages = {
            "download": {
                "method": cls.proc_download,
                "message": "Download raw data from Gemini archive?",
                "default": True,
                "keywords": {
                    "overwrite_download": True,
                }
            },
            "initial_setup": stages_super["initial_setup"],
            "reduce_flats": {
                "method": cls.proc_reduce_flats,
                "message": "Reduce flat-field images?",
                "default": True,
            },
            "reduce_science": {
                "method": cls.proc_reduce_science,
                "message": "Reduce science images?",
                "default": True,
            },
            "stack_science": {
                "method": cls.proc_stack_science,
                "message": "Stack science images with DISCO-STU?",
                "default": True,
            }
        }
        return stages

    def proc_download(self, output_dir: str, **kwargs):
        if 'overwrite_download' in kwargs:
            overwrite = kwargs['overwrite_download']
        else:
            overwrite = False
        self.retrieve(output_dir=output_dir, overwrite=overwrite)

    def retrieve(self, output_dir: str, overwrite: bool = False):
        # Get the science files
        science_files = retrieve.save_gemini_epoch(
            output=output_dir,
            program_id=self.program_id,
            coord=self.field.centre_coords,
            overwrite=overwrite
        )

        # Get the observation date from image headers if we don't have one specified
        if self.date is None:
            self.set_date(science_files["ut_datetime"][0])

        # Set up filters from retrieved science files.
        for img in science_files:
            fil = str(img["filter_name"])
            self.check_filter(fil)

        # Get the calibration files for the retrieved filters
        for fil in self.filters:
            print()
            print(f"Retrieving calibration files for {fil} band...")
            print()
            retrieve.save_gemini_calibs(
                output=output_dir,
                obs_date=self.date,
                fil=fil,
                overwrite=overwrite
            )

    def _initial_setup(self, output_dir: str, **kwargs):
        data_dir = self.data_path
        raw_dir = self.paths["download"]
        self.paths["redux_dir"] = redux_dir = os.path.join(data_dir, "redux")
        u.mkdir_check(redux_dir)
        # DO the initial database setup for DRAGONS.
        dragons.caldb_init(redux_dir=redux_dir)

        # Get a list of science files from the raw directory, using DRAGONS.
        science_list_name = "science.list"
        science_list = dragons.data_select(
            redux_dir=redux_dir,
            directory=raw_dir,
            expression="observation_class==\"science\"",
            output=science_list_name
        ).splitlines(False)[3:]
        self.paths["science_list"] = os.path.join(redux_dir, science_list_name)

        science_tbl_name = "science.csv"
        science_tbl = dragons.showd(
            input_filenames=science_list,
            descriptors="filter_name,exposure_time,object",
            output=science_tbl_name,
            csv=True,
            working_dir=redux_dir
        )
        # # Set up filters.
        # for img in science_tbl:
        #     fil = img["filter_name"]
        #     fil = fil[:fil.find("_")]
        #     self.check_filter(fil)
        u.debug_print(1, f"GSAOIImagingEpoch._inital_setup(): {self}.filters ==", self.filters)
        self.science_table = science_tbl

        # Get lists of flats for each filter.
        for fil in self.filters:
            flats_list_name = f"flats_{fil}.list"
            flats_list = dragons.data_select(
                redux_dir=redux_dir,
                directory=raw_dir,
                tags=["FLAT"],
                expression=f"filter_name==\"{fil}\"",
                output=flats_list_name
            ).splitlines(False)[3:]

            self.flats_lists[fil] = os.path.join(redux_dir, flats_list_name)
            self.frames_flat[fil] = flats_list

        # Get list of standard observations:
        std_tbl_name = "std_objects.csv"
        self.paths["std_tbl"] = os.path.join(redux_dir, std_tbl_name)
        std_list_name = "std_objects.list"
        self.paths["std_list"] = os.path.join(redux_dir, std_list_name)
        std_list = dragons.data_select(
            redux_dir=redux_dir,
            directory=raw_dir,
            expression=f"observation_class==\"partnerCal\"",
            output=std_list_name
        ).splitlines(False)[3:]

        std_tbl = dragons.showd(
            input_filenames=std_list,
            descriptors="object",
            output=std_tbl_name,
            csv=True,
            working_dir=redux_dir
        )

        # Set up dictionary of standard objects
        # TODO: ACCOUNT FOR MULTIPLE FILTERS.
        for std in std_tbl:
            if std["object"] not in self.std_objects:
                self.std_objects[std["object"]] = None

        for obj in self.std_objects:
            # And get the individual objects imaged like so:
            std_list_obj_name = f"std_{obj}.list"
            std_list_obj = dragons.data_select(
                redux_dir=redux_dir,
                directory=raw_dir,
                expression=f"object==\"{obj}\"",
                output=std_list_obj_name
            ).splitlines(False)[3:]
            self.std_objects[obj] = std_list_obj
            self.std_lists[obj] = os.path.join(redux_dir, std_list_obj_name)

    def proc_reduce_flats(self, output_dir: str, **kwargs):
        for fil in self.flats_lists:
            dragons.reduce(self.flats_lists[fil], redux_dir=self.paths["redux_dir"])
        flat_dir = os.path.join(self.paths["redux_dir"], "calibrations", "processed_flat")
        for flat in os.listdir(flat_dir):
            flat = os.path.join(flat_dir, flat)
            if not self.quiet:
                print(f"Adding {flat} to database.")
            sys_str = f"caldb add {flat}"
            if not self.quiet:
                print(sys_str)
            os.system(sys_str)

    def proc_reduce_science(self, output_dir: str, **kwargs):
        dragons.reduce(self.paths["science_list"], redux_dir=self.paths["redux_dir"])

    def proc_stack_science(self, output_dir: str, **kwargs):
        for fil in self.filters:
            dragons.disco(
                redux_dir=self.paths["redux_dir"],
                expression=f"filter_name==\"{fil}\" and observation_class==\"science\"",
                output=f"{self.name}_{fil}_stacked.fits",
                file_glob="*_sky*ed.fits",
                # refcat=self.field.paths["cat_csv_gaia"],
                # refcat_format="ascii.csv",
                # refcat_ra="ra",
                # refcat_dec="dec",
                # ignore_objcat=False
            )

    def check_filter(self, fil: str):
        not_none = super().check_filter(fil)
        if not_none:
            self.flats_lists[fil] = None
        return not_none

    def _output_dict(self):
        output_dict = super()._output_dict()
        output_dict.update({
            "flats_lists": self.flats_lists,
            "std_lists": self.std_lists
        })
        return output_dict

    def load_output_file(self, **kwargs):
        outputs = super().load_output_file(**kwargs)
        if type(outputs) is dict:
            if "flats_list" in outputs:
                self.flats_lists = outputs["flats_lists"]
            if "std" in outputs:
                self.std_lists = outputs["std"]
            if "flats" in outputs:
                self.frames_flat = outputs["flats"]
        return outputs

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        # default_params.update({})
        return default_params

    @classmethod
    def from_file(
            cls,
            param_file: Union[str, dict],
            name: str = None,
            field: 'fld.Field' = None
    ):

        name, param_file, param_dict = p.params_init(param_file)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")

        if field is None:
            field = param_dict.pop("field")

        if field is None:
            field = param_dict.pop("field")
        if 'target' in param_dict:
            target = param_dict.pop('target')
        else:
            target = None

        if "field" in param_dict:
            param_dict.pop("field")
        if "instrument" in param_dict:
            param_dict.pop("instrument")
        if "name" in param_dict:
            param_dict.pop("name")
        if "param_path" in param_dict:
            param_dict.pop("param_path")

        print(f"Loading field {field}...")
        u.debug_print(2, f"GSAOIImagingEpoch.from_file(): {param_dict=}")

        return cls(
            name=name,
            field=field,
            param_path=param_file,
            data_path=os.path.join(config["top_data_dir"], param_dict.pop('data_path')),
            instrument='gs-aoi',
            program_id=param_dict.pop('program_id'),
            date=param_dict.pop('date'),
            target=target,
            source_extractor_config=param_dict.pop('sextractor'),
            **param_dict
        )

    @classmethod
    def sort_files(cls, input_dir: str, output_dir: str = None, tolerance: units.Quantity = 3 * units.arcmin):
        """
        A routine to sort through a directory containing an arbitrary number of GSAOI observations and assign epochs to
        them.

        :param input_dir:
        :param tolerance: Images will be grouped if they are < tolerance from each other (or, specifically, from the
        first encountered in that cluster).
        :return:
        """
        pointings = {}
        if output_dir is None:
            output_dir = input_dir
        u.mkdir_check(output_dir)
        files = os.listdir(input_dir)
        files.sort()
        for file in filter(lambda f: f.endswith(".fits"), files):
            # Since GSAOI science files cannot be relied upon to include the object/target in the header, we group
            # images by RA and Dec.
            path = os.path.join(input_dir, file)
            img = image.from_path(
                path,
                cls=image.GSAOIImage
            )
            pointing = img.extract_pointing()
            associated = False
            for pointing_str in pointings:
                pointings_list = pointings[pointing_str]
                other_pointing = pointings_list[0]
                if pointing.separation(other_pointing) <= tolerance:
                    pointings_list.append(pointing)
                    associated = True
                    shutil.move(path, pointing_str)
                    break

            if not associated:
                pointing_str = os.path.join(output_dir, pointing.to_string())
                u.mkdir_check(pointing_str)
                pointings[pointing_str] = [pointing]
                shutil.move(path, pointing_str)

        return pointings


class HubbleImagingEpoch(ImagingEpoch):
    instrument_name = "hst-dummy"
    coadded_class = image.HubbleImage

    def __init__(
            self,
            name: str = None,
            field: 'fld.Field' = None,
            param_path: str = None,
            data_path: str = None,
            instrument: str = None,
            program_id: str = None,
            date: Union[str, Time] = None,
            target: str = None,
            standard_epochs: list = None,
            source_extractor_config: dict = None,
            **kwargs
    ):
        super().__init__(
            name=name,
            field=field,
            param_path=param_path,
            data_path=data_path,
            instrument=instrument,
            date=date,
            program_id=program_id,
            target=target,
            standard_epochs=standard_epochs, source_extractor_config=source_extractor_config)

        self.load_output_file(mode="imaging")

    @classmethod
    def stages(cls):
        super_stages = super().stages()
        stages = {
            "download": super_stages["download"],
            "initial_setup": super_stages["initial_setup"],
            "source_extraction": super_stages["source_extraction"],
            "photometric_calibration": super_stages["photometric_calibration"],
            "get_photometry": super_stages["get_photometry"]
        }
        return stages

    def _pipeline_init(self):
        super()._pipeline_init()
        self.coadded_final = "coadded"
        self.paths["download"] = os.path.join(self.data_path, "0-download")

    def _initial_setup(self, output_dir: str, **kwargs):
        download_dir = self.paths["download"]
        # for file in filter(lambda f: f.endswith(".fits"), os.listdir(self.data_path)):
        #     shutil.move(os.path.join(self.data_path, file), output_dir)
        for file in filter(lambda f: f.endswith(".fits"), os.listdir(download_dir)):
            path = os.path.join(download_dir, file)
            img = image.from_path(
                path,
                cls=image.HubbleImage
            )
            if self.instrument_name in [None, "hst-dummy"]:
                self.instrument_name = img.instrument_name
            fil = img.extract_filter()
            img.extract_date_obs()
            self.date = img.date
            self.exp_time_mean[fil] = img.extract_header_item('TEXPTIME') * units.second / img.extract_ncombine()
            img.set_header_item('INTTIME', img.extract_header_item('TEXPTIME'))
            self.add_coadded_image(img, key=fil)
            self.add_coadded_unprojected_image(img, key=fil)
            self.check_filter(img.filter_name)

    def photometric_calibration(
            self,
            image_dict: dict,
            output_path: str,
            **kwargs):

        for fil in image_dict:
            image_dict[fil].zeropoint()
            image_dict[fil].estimate_depth()
            self.deepest = image_dict[fil]

    def proc_get_photometry(self, output_dir: str, **kwargs):
        self.get_photometry(output_dir, image_type="coadded", dual=False)

    def psf_diagnostics(
            self,
            images: dict = None
    ):
        if images is None:
            images = self._get_images("final")

        for fil in images:
            img = images[fil]
            if fil == "F300X":
                self.psf_stats[fil] = {
                    "n_stars": 0,
                    "fwhm_psfex": -999 * units.arcsec,
                    "gauss": {
                        "fwhm_median": -999 * units.arcsec,
                        "fwhm_mean": -999 * units.arcsec,
                        "fwhm_max": -999 * units.arcsec,
                        "fwhm_min": -999 * units.arcsec,
                        "fwhm_sigma": -999 * units.arcsec,
                        "fwhm_rms": -999 * units.arcsec
                    },
                    "moffat": {
                        "fwhm_median": -999 * units.arcsec,
                        "fwhm_mean": -999 * units.arcsec,
                        "fwhm_max": -999 * units.arcsec,
                        "fwhm_min": -999 * units.arcsec,
                        "fwhm_sigma": -999 * units.arcsec,
                        "fwhm_rms": -999 * units.arcsec
                    },
                    "sextractor": {
                        "fwhm_median": -999 * units.arcsec,
                        "fwhm_mean": -999 * units.arcsec,
                        "fwhm_max": -999 * units.arcsec,
                        "fwhm_min": -999 * units.arcsec,
                        "fwhm_sigma": -999 * units.arcsec,
                        "fwhm_rms": -999 * units.arcsec
                    }
                }
                img.set_header_items(
                    {
                        "PSF_FWHM": -999,
                        "PSF_FWHM_ERR": -999,
                    },
                    write=True
                )
            else:
                if not self.quiet:
                    print(f"Performing PSF measurements on {img}...")
                self.psf_stats[fil], _ = img.psf_diagnostics()

        self.update_output_file()
        return self.psf_stats

    def add_coadded_image(self, img: Union[str, image.Image], key: str, **kwargs):
        try:
            if isinstance(img, str):
                img = image.from_path(
                    img,
                    cls=image.HubbleImage
                )
            img.epoch = self
            self.coadded[key] = img
            return img
        except FileNotFoundError:
            return None

    def n_frames(self, fil: str):
        return self.coadded[fil].extract_ncombine()

    @classmethod
    def from_file(
            cls,
            param_file: Union[str, dict],
            name: str = None,
            field: 'fld.Field' = None
    ):

        name, param_file, param_dict = p.params_init(param_file)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")

        if field is None:
            field = param_dict.pop("field")
        if 'target' in param_dict:
            target = param_dict.pop('target')
        else:
            target = None

        if "field" in param_dict:
            param_dict.pop("field")
        if "name" in param_dict:
            param_dict.pop("name")
        if "param_path" in param_dict:
            param_dict.pop("param_path")

        return cls(
            name=name,
            field=field,
            param_path=param_file,
            data_path=os.path.join(config["top_data_dir"], param_dict.pop('data_path')),
            instrument=param_dict.pop("instrument"),
            program_id=param_dict.pop('program_id'),
            date=param_dict.pop('date'),
            target=target,
            source_extractor_config=param_dict.pop('sextractor'),
            **param_dict
        )


class SurveyImagingEpoch(ImagingEpoch):
    mode = "imaging"
    catalogue = None
    coadded_class = image.SurveyCutout
    preferred_zeropoint = "calib_pipeline"

    def __init__(
            self,
            name: str = None,
            field: Union[str, 'fld.Field'] = None,
            param_path: str = None,
            data_path: str = None,
            source_extractor_config: dict = None,
            **kwargs
    ):
        super().__init__(
            name=name,
            field=field,
            param_path=param_path,
            data_path=data_path,
            source_extractor_config=source_extractor_config,
            instrument=self.instrument_name
        )
        self.load_output_file(mode="imaging")
        # if isinstance(field, Field):
        # self.field.retrieve_catalogue(cat_name=self.catalogue)

        u.debug_print(1, f"SurveyImagingEpoch.__init__(): {self}.filters ==", self.filters)

    @classmethod
    def stages(cls):
        super_stages = super().stages()
        super_stages["source_extraction"]["do_astrometry_diagnostics"] = False
        stages = {
            "download": super_stages["download"],
            "initial_setup": super_stages["initial_setup"],
            "source_extraction": super_stages["source_extraction"],
            "photometric_calibration": super_stages["photometric_calibration"],
            # "dual_mode_source_extraction": super_stages["dual_mode_source_extraction"],
            "get_photometry": super_stages["get_photometry"]
        }
        return stages

    def _pipeline_init(self):
        super()._pipeline_init()
        self.coadded_final = "coadded"
        self.paths["download"] = os.path.join(self.data_path, "0-download")
        # self.frames_final = "coadded"

    # TODO: Automatic cutout download; don't worry for now.
    def proc_download(self, output_dir: str, **kwargs):
        """
        Automatically download survey cutout.
        :param output_dir:
        :param kwargs:
        :return:
        """
        pass

    def proc_source_extraction(self, output_dir: str, **kwargs):
        self.source_extraction(
            output_dir=output_dir,
            do_astrometry_diagnostics=False,
            do_psf_diagnostics=True,
            **kwargs
        )

    def proc_get_photometry(self, output_dir: str, **kwargs):
        self.load_output_file()
        self.get_photometry(output_dir, image_type="coadded")

    def _initial_setup(self, output_dir: str, **kwargs):
        download_dir = self.paths["download"]
        # for file in filter(lambda f: f.endswith(".fits"), os.listdir("download")):
        #     shutil.move(os.path.join(self.data_path, file), output_dir)
        self.set_path("imaging_dir", download_dir)
        # Write a table of fits files from the 0-imaging directory.
        table_path_all = os.path.join(self.data_path, f"{self.name}_fits_table_all.csv")
        self.set_path("fits_table", table_path_all)
        image.fits_table_all(input_path=download_dir, output_path=table_path_all, science_only=False)
        for file in filter(lambda f: f.endswith(".fits"), os.listdir(download_dir)):
            path = os.path.join(download_dir, file)
            img = self.coadded_class(path=path)
            fil = img.extract_filter()
            u.debug_print(2, f"PanSTARRS1ImagingEpoch._initial_setup(): {fil=}")
            self.exp_time_mean[fil] = img.extract_exposure_time() / img.extract_ncombine()
            img.set_header_item('INTTIME', img.extract_integration_time())
            self.add_coadded_image(img, key=fil)
            self.check_filter(img.filter_name)
            img.write_fits_file()

    def guess_data_path(self):
        if self.data_path is None and self.field is not None and self.field.data_path is not None:
            self.data_path = os.path.join(self.field.data_path, "imaging", self.catalogue)
        return self.data_path

    def zeropoint(
            self,
            output_path: str,
            distance_tolerance: units.Quantity = 1 * units.arcsec,
            snr_min: float = 3.,
            star_class_tolerance: float = 0.95,
            **kwargs
    ):
        u.debug_print(2, f"", self.filters)
        deepest = None
        for fil in self.coadded:
            img = self.coadded[fil]
            zp = img.zeropoint(
                cat_path=self.field.get_path(f"cat_csv_{self.catalogue}"),
                output_path=os.path.join(output_path, img.name),
                cat_name=self.catalogue,
                dist_tol=distance_tolerance,
                show=False,
                snr_cut=snr_min,
                star_class_tol=star_class_tolerance,
                image_name=f"{self.catalogue}",
            )
            img.select_zeropoint(True, preferred=self.preferred_zeropoint)
            img.estimate_depth(zeropoint_name=self.catalogue)  # , do_magnitude_calibration=False)
            if deepest is not None:
                deepest = image.deepest(deepest, img)
            else:
                deepest = img

        return deepest

    def n_frames(self, fil: str):
        img = self.coadded[fil]
        return img.extract_ncombine()

    def add_coadded_image(self, img: Union[str, image.Image], key: str, **kwargs):
        if isinstance(img, str):
            if os.path.isfile(img):
                cls = self.coadded_class
                img = image.from_path(path=img, cls=cls)
            else:
                return None
        img.epoch = self
        self.coadded[key] = img
        self.coadded_unprojected[key] = img
        return img

    @classmethod
    def from_file(
            cls,
            param_file: Union[str, dict],
            name: str = None,
            field: 'fld.Field' = None
    ):
        name, param_file, param_dict = p.params_init(param_file)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")

        if field is None:
            field = param_dict.pop("field")

        if "field" in param_dict:
            param_dict.pop("field")
        if "instrument" in param_dict:
            param_dict.pop("instrument")
        if "name" in param_dict:
            param_dict.pop("name")
        if "param_path" in param_dict:
            param_dict.pop("param_path")

        epoch = cls(
            name=name,
            field=field,
            param_path=param_file,
            data_path=os.path.join(config["top_data_dir"], param_dict.pop('data_path')),
            source_extractor_config=param_dict.pop('sextractor'),
            **param_dict
        )
        # epoch.instrument = cls.instrument_name
        return epoch


class DESEpoch(SurveyImagingEpoch):
    instrument_name = "decam"
    catalogue = "des"
    coadded_class = image.DESCutout

    def n_frames(self, fil: str):
        return 1

    def proc_split(self, output_dir: str, **kwargs):
        if "image_type" not in kwargs:
            kwargs["image_type"] = "coadded"
        self.split(output_dir=output_dir, **kwargs)

    def split(self, output_dir: str, image_type):
        image_dict = self._get_images(image_type=image_type)
        for fil in image_dict:
            img = image_dict[fil]
            split_imgs = img.split_fits(output_dir=output_dir)
            self.add_coadded_image(split_imgs["SCI"], key=fil)

    @classmethod
    def stages(cls):
        super_stages = super().stages()
        stages = {
            "download": super_stages["download"],
            "initial_setup": super_stages["initial_setup"],
            "split": {
                "method": cls.proc_split,
                "message": "Split fits files into components?",
                "default": True,
                "keywords": {}
            },
            "source_extraction": super_stages["source_extraction"],
            "photometric_calibration": super_stages["photometric_calibration"],
            # "dual_mode_source_extraction": super_stages["dual_mode_source_extraction"],
            "get_photometry": super_stages["get_photometry"]
        }
        return stages


class PanSTARRS1ImagingEpoch(SurveyImagingEpoch):
    instrument_name = "panstarrs1"
    catalogue = "panstarrs1"
    coadded_class = image.PanSTARRS1Cutout
    preferred_zeropoint = "panstarrs1"

    # TODO: Automatic cutout download; don't worry for now.
    def proc_download(self, output_dir: str, **kwargs):
        """
        Automatically download PanSTARRS1 cutout.
        :param output_dir:
        :param kwargs:
        :return:
        """
        pass


def _retrieve_eso_epoch(
        epoch: Union['ESOImagingEpoch', 'ESOSpectroscopyEpoch'],
        path: str
):
    from .spectroscopy import ESOSpectroscopyEpoch
    epoch_date = None
    if epoch.date is not None:
        epoch_date = epoch.date
    program_id = None
    if epoch.program_id is not None:
        program_id = epoch.program_id
    if isinstance(epoch, ESOImagingEpoch):
        mode = "imaging"
    elif isinstance(epoch, ESOSpectroscopyEpoch):
        mode = "spectroscopy"
    else:
        raise TypeError("epoch must be either an ESOImagingEpoch or an ESOSpectroscopyEpoch.")
    if epoch.target is None:
        obj = epoch.field.centre_coords
    else:
        obj = epoch.target

    u.mkdir_check(path)
    instrument = epoch.instrument_name.split('-')[-1]

    query = retrieve.query_eso_raw(
        select="dp_id,date_obs",
        date_obs=epoch_date,
        program_id=program_id,
        instrument=instrument,
        mode=mode,
        obj=obj,
        coord_tol=3.0 * units.arcmin
    )

    frame_list = retrieve.get_eso_raw_frame_list(query=query)
    epoch_dates = retrieve.count_epochs(frame_list["date_obs"])
    if len(epoch_dates) > 1:
        _, epoch_date = u.select_option(
            message="Multiple observation dates found matching epoch criteria. Please select one:",
            options=epoch_dates,
            sort=True
        )
    epoch.set_date(epoch_date)

    r = retrieve.save_eso_raw_data_and_calibs(
        output=path,
        date_obs=epoch_date,
        program_id=program_id,
        instrument=instrument,
        mode=mode,
        obj=obj,
        coord_tol=3.0 * units.arcmin
    )

    if r:
        os.system(f"uncompress {path}/*.Z -f")

    for file in os.listdir(path):
        shutil.move(
            os.path.join(path, file),
            os.path.join(path, file.replace(":", "_"))
        )

    return r


class ESOImagingEpoch(ImagingEpoch):
    instrument_name = "dummy-instrument"
    mode = "imaging"
    eso_name = None

    def __init__(
            self,
            name: str = None,
            field: 'fld.Field' = None,
            param_path: str = None,
            data_path: str = None,
            instrument: str = None,
            program_id: str = None,
            date: Union[str, Time] = None,
            target: str = None,
            standard_epochs: list = None,
            source_extractor_config: dict = None,
            **kwargs
    ):
        u.debug_print(2, f"ESOImagingEpoch.__init__(): kwargs ==", kwargs)
        super().__init__(
            name=name,
            field=field,
            param_path=param_path,
            data_path=data_path,
            instrument=instrument,
            date=date,
            program_id=program_id,
            target=target,
            standard_epochs=standard_epochs,
            source_extractor_config=source_extractor_config,
            **kwargs)

        self.frames_esoreflex_backgrounds = {}

        self.load_output_file(mode="imaging")

    @classmethod
    def stages(cls):
        super_stages = super().stages()

        super_stages["initial_setup"].update(
            {
                "keywords": {"skip_esoreflex_copy": False}
            }
        )

        stages = {
            "download": {
                "method": cls.proc_download,
                "message": "Download raw data from ESO archive?",
                "default": True,
                "keywords": {
                    "alternate_dir": None
                }
            },
            "initial_setup": super_stages["initial_setup"],
            "sort_reduced": {
                "method": cls.proc_sort_reduced,
                "message": "Sort ESOReflex products? Requires reducing data with ESOReflex first.",
                "default": True,
                "keywords": {
                    "alternate_dir": None,  # alternate directory to pull reduced files from.
                    "delete_eso_output": False
                }
            },
            "trim_reduced": {
                "method": cls.proc_trim_reduced,
                "message": "Trim reduced images?",
                "default": True,
            },
            "convert_to_cs": {
                "method": cls.proc_convert_to_cs,
                "message": "Convert image values to counts/second?",
                "default": True,
                "keywords": {
                    "upper_only": False
                }
            },
        }
        return stages

    def proc_download(self, output_dir: str, **kwargs):

        # Check for alternate directory.
        alt_dir = None
        if "alternate_dir" in kwargs and isinstance(kwargs["alternate_dir"], str):
            alt_dir = kwargs["alternate_dir"]

        if alt_dir is None:
            r = self.retrieve(output_dir)
            if r:
                return True
        else:
            u.rmtree_check(output_dir)
            shutil.copytree(alt_dir, output_dir)
            return True

    def retrieve(self, output_dir: str):
        """
        Check ESO archive for the epoch raw frames, and download those frames and associated files.
        :return:
        """
        r = []
        r = _retrieve_eso_epoch(self, path=output_dir)
        return r

    def _initial_setup(self, output_dir: str, **kwargs):
        u.debug_print(2, f"ESOImagingEpoch._initial_setup(): {self.paths=}")
        raw_dir = self.get_path("download")
        data_dir = self.data_path
        data_title = self.name

        p.set_eso_user()

        self.frames_science = {}
        self.frames_flat = {}
        self.frames_bias = []
        self.frames_raw = []
        self.filters = []

        # Write tables of fits files to main directory; firstly, science images only:
        tbl = image.fits_table(
            input_path=raw_dir,
            output_path=os.path.join(data_dir, data_title + "_fits_table_science.csv"),
            science_only=True
        )
        # Then including all calibration files
        tbl_full = image.fits_table(
            input_path=raw_dir,
            output_path=os.path.join(data_dir, data_title + "_fits_table_all.csv"),
            science_only=False
        )
        image.fits_table_all(
            input_path=raw_dir,
            output_path=os.path.join(data_dir, data_title + "_fits_table_detailed.csv"),
            science_only=False
        )

        not_science = []
        # We do this in two pieces so that we don't add calibration frames that aren't for relevant filters
        # (which the ESO archive often associates anyway, especially with HAWK-I)
        for i, row in enumerate(tbl_full):
            path = os.path.join(raw_dir, row["identifier"])
            cls = image.ImagingImage.select_child_class(instrument_name=self.instrument_name, mode="imaging")
            img = image.from_path(path, cls=cls)
            img.extract_frame_type()
            img.extract_filter()
            u.debug_print(1, self.instrument_name, cls, img.name, img.frame_type)
            # The below will also update the filter list.
            u.debug_print(
                2,
                f"_initial_setup(): Adding frame {img.name}, type {img.frame_type}/{type(img)}, to {self}, type {type(self)}")
            if img.frame_type == "science":
                self.add_frame_raw(img)
            else:
                not_science.append(img)

        for img in not_science:
            if img.filter_name in self.filters:
                self.add_frame_raw(img)

        u.debug_print(2, f"ESOImagingEpoch._initial_setup(): {self.frames_science=}")
        # Collect and save some stats on those filters:
        for i, fil in enumerate(self.filters):
            if len(self.frames_science[fil]) == 0:
                self.filters.remove(fil)
                self.frames_science.pop(fil)
                continue
            exp_times = list(map(lambda frame: frame.extract_exposure_time().value, self.frames_science[fil]))
            u.debug_print(1, "exposure times:")
            u.debug_print(1, exp_times)
            self.exp_time_mean[fil] = np.nanmean(exp_times) * units.second
            self.exp_time_err[fil] = np.nanstd(exp_times) * units.second

            airmasses = list(map(lambda frame: frame.extract_airmass(), self.frames_science[fil]))

            self.airmass_mean[fil] = np.nanmean(airmasses)
            self.airmass_err[fil] = max(
                np.nanmax(airmasses) - self.airmass_mean[fil],
                self.airmass_mean[fil] - np.nanmin(airmasses)
            )

        inst_reflex_dir = {
            "vlt-fors2": "fors",
            "vlt-hawki": "hawki"
        }[self.instrument_name]

        inst_reflex_dir = os.path.join(config["esoreflex_input_dir"], inst_reflex_dir)
        u.mkdir_check_nested(inst_reflex_dir, remove_last=False)

        survey_raw_path = None
        if isinstance(self.field.survey, survey.Survey) and self.field.survey.raw_stage_path is not None:
            survey_raw_path = os.path.join(self.field.survey.raw_stage_path, self.field.name, self.instrument_name)
            u.mkdir_check_nested(survey_raw_path, remove_last=False)

        if not ("skip_esoreflex_copy" in kwargs and kwargs["skip_esoreflex_copy"]):
            for file in os.listdir(raw_dir):
                if not self.quiet:
                    print(f"Copying {file} to ESOReflex input directory...")
                origin = os.path.join(raw_dir, file)
                shutil.copy(origin, os.path.join(config["esoreflex_input_dir"], inst_reflex_dir))
                if not self.quiet:
                    print("Done.")

                if survey_raw_path is not None:
                    survey_raw_path_file = os.path.join(
                        survey_raw_path,
                        file
                    )
                    if not self.quiet:
                        print(f"Copying {file} to {survey_raw_path_file}...")
                    shutil.copy(
                        origin,
                        survey_raw_path_file
                    )
                    if not self.quiet:
                        print("Done.")

        # This line looks for a non-empty frames_science list
        i = 0
        while not self.frames_science[self.filters[i]]:
            i += 1
        tmp = self.frames_science[self.filters[i]][0]
        if self.date is None:
            self.set_date(tmp.extract_date_obs())
        if self.target is None:
            self.set_target(tmp.extract_object())
        if self.program_id is None:
            self.set_program_id(tmp.extract_program_id())

        self.update_output_file()

        # if str(self.field.survey) == "FURBY":
        #     u.system_command_verbose(
        #         f"furby_vlt_ob {self.field.name} {tmp.filter.band_name} --observed {self.date_str()}"
        #     )
        # u.system_command_verbose(f"furby_vlt_ob {self.field.name} {tmp.filter.band_name} --completed")

        try:
            u.system_command_verbose("esoreflex")
        except SystemError:
            print("Could not open ESO Reflex; may not be installed, or installed to other environment.")

    def proc_sort_reduced(self, output_dir: str, **kwargs):
        self.sort_after_esoreflex(output_dir=output_dir, **kwargs)

    def sort_after_esoreflex(self, output_dir: str, **kwargs):
        """
        Scans through the ESO Reflex directory for the files matching this epoch, and puts them where we want them.
        :param output_dir:
        :param kwargs:
        :return:
        """

        self.frames_reduced = {}
        self.frames_esoreflex_backgrounds = {}

        # Check for alternate directory.
        if "alternate_dir" in kwargs and isinstance(kwargs["alternate_dir"], str):
            eso_dir = kwargs["alternate_dir"]
            expect_sorted = True
            if "expect_sorted" in kwargs and isinstance(kwargs["expect_sorted"], bool):
                expect_sorted = kwargs["expect_sorted"]
        else:
            eso_dir = os.path.join(p.config['esoreflex_output_dir'], "reflex_end_products")
            expect_sorted = False

        if "delete_eso_output" in kwargs:
            delete_output = kwargs["delete_eso_output"]
        else:
            delete_output = False

        if not self.quiet:
            print(f"Copying files from {eso_dir} to {output_dir}")
            print(self.date_str())

        if os.path.isdir(eso_dir):
            if expect_sorted:
                shutil.rmtree(output_dir)
                shutil.copytree(
                    eso_dir,
                    output_dir,
                )

                science = os.path.join(output_dir, "science")
                for fil in filter(lambda d: os.path.isdir(os.path.join(science, d)), os.listdir(science)):
                    output_subdir = os.path.join(science, fil)
                    if not self.quiet:
                        print(f"Adding reduced science images from {output_subdir}")
                    for file in filter(lambda f: f.endswith(".fits"), os.listdir(output_subdir)):
                        path = os.path.join(output_subdir, file)
                        # TODO: This (and other FORS2Image instances in this method) WILL NOT WORK WITH HAWKI. Must make more flexible.
                        img = image.from_path(
                            path,
                            cls=image.FORS2Image
                        )
                        self.add_frame_reduced(img)
                backgrounds = os.path.join(output_dir, "backgrounds")
                for fil in filter(lambda d: os.path.isdir(os.path.join(backgrounds, d)), os.listdir(backgrounds)):
                    output_subdir = os.path.join(backgrounds, fil)
                    if not self.quiet:
                        print(f"Adding background images from {output_subdir}")
                    for file in filter(lambda f: f.endswith(".fits"), os.listdir(output_subdir)):
                        path = os.path.join(output_subdir, file)
                        img = image.from_path(
                            path,
                            cls=image.FORS2Image
                        )
                        self.add_frame_background(img)

            else:

                # The ESOReflex output directory is structured in a very specific way, which we now traverse.
                mjd = int(self.mjd())
                obj = self.target.lower()
                if not self.quiet:
                    print(f"Looking for data with object '{obj}' and MJD of observation {mjd} inside {eso_dir}")
                # Look for files with the appropriate object and MJD, as recorded in output_values

                # List directories in eso_output_dir; these are dates on which data was reduced using ESOReflex.
                date_dirs = filter(
                    lambda d: os.path.isdir(os.path.join(eso_dir, d)),
                    os.listdir(eso_dir)
                )
                date_dirs = map(lambda d: os.path.join(eso_dir, d), date_dirs)
                for date_dir in date_dirs:
                    if not self.quiet:
                        print(f"Searching {date_dir}")
                    eso_subdirs = filter(
                        lambda d: os.path.isdir(os.path.join(date_dir, d)) and self.eso_name in d,
                        os.listdir(date_dir)
                    )
                    eso_subdirs = list(map(
                        lambda d: os.path.join(os.path.join(date_dir, d)),
                        eso_subdirs
                    ))
                    for subpath in eso_subdirs:
                        if not self.quiet:
                            print(f"\tSearching {subpath}")
                        self._sort_after_esoreflex(
                            output_dir=output_dir,
                            date_dir=date_dir,
                            obj=obj,
                            mjd=mjd,
                            delete_output=delete_output,
                            subpath=subpath,
                            **kwargs
                        )

        else:
            raise IOError(f"ESO output directory '{eso_dir}' not found.")

        if not self.frames_reduced:
            u.debug_print(2, "ESOImagingEpoch._sort_after_esoreflex(): kwargs ==", kwargs)

            print(f"WARNING: No reduced frames were found in the target directory {eso_dir}.")

    def _sort_after_esoreflex(
            self,
            output_dir: str,
            date_dir: str,
            obj: str,
            mjd: int,
            delete_output: bool,
            subpath: str,
            **kwargs
    ):
        """

        :param output_dir:
        :param date_dir:
        :param obj:
        :param mjd:
        :param kwargs:
        :return:
        """

    def proc_trim_reduced(self, output_dir: str, **kwargs):
        self.trim_reduced(
            output_dir=output_dir,
            **kwargs
        )

    def trim_reduced(
            self,
            output_dir: str,
            **kwargs
    ):

        u.mkdir_check(os.path.join(output_dir, "backgrounds"))
        u.mkdir_check(os.path.join(output_dir, "science"))

        u.debug_print(
            2, f"ESOImagingEpoch.trim_reduced(): {self}.frames_esoreflex_backgrounds ==",
            self.frames_esoreflex_backgrounds)

        self.frames_trimmed = {}
        for fil in self.filters:
            self.check_filter(fil)

        edged = False

        up_left = 0
        up_right = 0
        up_bottom = 0
        up_top = 0

        dn_left = 0
        dn_right = 0
        dn_bottom = 0
        dn_top = 0

        for fil in self.filters:
            fil_path_back = os.path.join(output_dir, "backgrounds", fil)
            fil_path_science = os.path.join(output_dir, "science", fil)
            u.mkdir_check(fil_path_back)
            u.mkdir_check(fil_path_science)

            if not edged:
                # Find borders of noise frame using backgrounds.
                # First, make sure that the background we're using is for the top chip.
                i = 0
                img = self.frames_esoreflex_backgrounds[fil][i]
                while img.extract_chip_number() != 1:
                    u.debug_print(1, i, img.extract_chip_number())
                    i += 1
                    img = self.frames_esoreflex_backgrounds[fil][i]
                up_left, up_right, up_bottom, up_top = ff.detect_edges(img.path)
                # Ditto for the bottom chip.
                i = 0
                img = self.frames_esoreflex_backgrounds[fil][i]
                while img.extract_chip_number() != 2:
                    i += 1
                    img = self.frames_esoreflex_backgrounds[fil][i]
                dn_left, dn_right, dn_bottom, dn_top = ff.detect_edges(img.path)
                up_left = up_left + 5
                up_right = up_right - 5
                up_top = up_top - 5
                dn_left = dn_left + 5
                dn_right = dn_right - 5
                dn_bottom = dn_bottom + 5

                edged = True

            for i, frame in enumerate(self.frames_esoreflex_backgrounds[fil]):
                new_path = os.path.join(
                    fil_path_back,
                    frame.filename.replace(".fits", "_trim.fits")
                )
                if not self.quiet:
                    print(f'Trimming {i} {frame}')

                # Split the files into upper CCD and lower CCD
                if frame.extract_chip_number() == 1:
                    frame.trim(left=up_left, right=up_right, top=up_top, bottom=up_bottom, output_path=new_path)
                elif frame.extract_chip_number() == 2:
                    frame.trim(left=dn_left, right=dn_right, top=dn_top, bottom=dn_bottom, output_path=new_path)
                else:
                    raise ValueError('Invalid chip ID; could not trim based on upper or lower chip.')

            # Repeat for science images

            for i, frame in enumerate(self.frames_reduced[fil]):
                # Split the files into upper CCD and lower CCD
                new_file = frame.filename.replace(".fits", "_trim.fits")
                new_path = os.path.join(fil_path_science, new_file)
                frame.set_header_item(
                    key='GAIN',
                    value=frame.extract_gain())
                frame.set_header_item(
                    key='SATURATE',
                    value=65535.)
                frame.set_header_item(
                    key='BUNIT',
                    value="ct"
                )

                frame.write_fits_file()

                if frame.extract_chip_number() == 1:
                    trimmed = frame.trim(
                        left=up_left,
                        right=up_right,
                        top=up_top,
                        bottom=up_bottom,
                        output_path=new_path)
                    self.add_frame_trimmed(trimmed)

                elif frame.extract_chip_number() == 2:
                    trimmed = frame.trim(
                        left=dn_left,
                        right=dn_right,
                        top=dn_top,
                        bottom=dn_bottom,
                        output_path=new_path)
                    self.add_frame_trimmed(trimmed)

    def proc_convert_to_cs(self, output_dir: str, **kwargs):
        self.convert_to_cs(
            output_dir=output_dir,
            **kwargs
        )

    def convert_to_cs(self, output_dir: str, **kwargs):

        self.frames_normalised = {}

        if "upper_only" in kwargs:
            upper_only = kwargs["upper_only"]
        else:
            upper_only = False

        u.mkdir_check(output_dir)
        u.mkdir_check(os.path.join(output_dir, "science"))
        u.mkdir_check(os.path.join(output_dir, "backgrounds"))

        for fil in self.filters:
            fil_path_science = os.path.join(output_dir, "science", fil)
            fil_path_back = os.path.join(output_dir, "backgrounds", fil)
            u.mkdir_check(fil_path_science)
            u.mkdir_check(fil_path_back)
            for frame in self.frames_trimmed[fil]:
                do = True
                if upper_only:
                    if frame.extract_chip_number() != 1:
                        do = False

                if do:
                    science_destination = os.path.join(
                        output_dir,
                        "science",
                        fil,
                        frame.filename.replace("trim", "norm"))

                    # Divide by exposure time to get an image in counts/second.
                    normed = frame.convert_to_cs(output_path=science_destination)
                    self.add_frame_normalised(normed)

    def add_frame_background(self, background_frame: Union[image.ImagingImage, str]):
        self._add_frame(
            frame=background_frame,
            frames_dict=self.frames_esoreflex_backgrounds,
            frame_type="reduced"
        )

    def check_filter(self, fil: str):
        not_none = super().check_filter(fil)
        if not_none:
            if fil not in self.frames_esoreflex_backgrounds:
                self.frames_esoreflex_backgrounds[fil] = []
            if fil not in self.frames_trimmed:
                self.frames_trimmed[fil] = []
        return not_none

    def _output_dict(self):
        output_dict = super()._output_dict()

        output_dict.update({
            "frames_trimmed": _output_img_dict_list(self.frames_trimmed),
            "frames_esoreflex_backgrounds": _output_img_dict_list(self.frames_esoreflex_backgrounds)
        })
        return output_dict

    def load_output_file(self, **kwargs):
        outputs = super().load_output_file(**kwargs)
        if type(outputs) is dict:
            # cls = image.Image.select_child_class(instrument=self.instrument_name, mode='imaging')
            if "frames_trimmed" in outputs:
                for fil in outputs["frames_trimmed"]:
                    if outputs["frames_trimmed"][fil] is not None:
                        for frame in outputs["frames_trimmed"][fil]:
                            self.add_frame_trimmed(frame=frame)
            if "frames_esoreflex_backgrounds" in outputs:
                for fil in outputs["frames_esoreflex_backgrounds"]:
                    if outputs["frames_esoreflex_backgrounds"][fil] is not None:
                        for frame in outputs["frames_esoreflex_backgrounds"][fil]:
                            self.add_frame_background(background_frame=frame)

        return outputs

    @classmethod
    def from_file(
            cls,
            param_file: Union[str, dict],
            name: str = None,
            field: 'fld.Field' = None
    ):

        name, param_file, param_dict = p.params_init(param_file)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")

        if field is None:
            field = param_dict.pop("field")
        if 'target' in param_dict:
            target = param_dict.pop('target')
        else:
            target = None

        if "field" in param_dict:
            param_dict.pop("field")
        if "instrument" in param_dict:
            param_dict.pop("instrument")
        if "name" in param_dict:
            param_dict.pop("name")
        if "param_path" in param_dict:
            param_dict.pop("param_path")

        u.debug_print(2, f"ESOImagingEpoch.from_file(), cls ==", cls)
        u.debug_print(2, 'ESOImagingEpoch.from_file(), config["top_data_dir"] == ', config["top_data_dir"])
        u.debug_print(2, 'ESOImagingEpoch.from_file(), param_dict["data_path"] == ', param_dict["data_path"])

        u.debug_print(2, "ESOImagingEpoch.from_file(): param_dict ==", param_dict)

        if "sextractor" in param_dict:
            se = param_dict.pop("sextractor")
        else:
            se = None

        return cls(
            name=name,
            field=field,
            param_path=param_file,
            data_path=param_dict.pop('data_path'),
            instrument=cls.instrument_name,
            program_id=param_dict.pop('program_id'),
            date=param_dict.pop('date'),
            target=target,
            source_extractor_config=se,
            **param_dict
        )


class HAWKIImagingEpoch(ESOImagingEpoch):
    instrument_name = "vlt-hawki"
    frame_class = image.HAWKIImage
    coadded_class = image.HAWKICoaddedImage
    eso_name = "HAWKI"

    def __init__(
            self,
            **kwargs
    ):
        self.coadded_esoreflex = {}
        self.frames_split = {}
        super().__init__(**kwargs)

    def n_frames(self, fil: str):
        return self.coadded_astrometry[fil].extract_ncombine()

    @classmethod
    def stages(cls):
        eso_stages = super().stages()
        ie_stages = ImagingEpoch.stages()
        stages = {
            "download": eso_stages["download"],
            "initial_setup": eso_stages["initial_setup"],
            "sort_reduced": eso_stages["sort_reduced"],
            "split_frames": {
                "method": cls.proc_split_frames,
                "message": "Split ESO Reflex frames into separate files?",
                "log_message": "Split ESO Reflex frames into separate .fits files",
                "default": True,
            },
            "coadd": ie_stages["coadd"],
            "correct_astrometry_coadded": ie_stages["correct_astrometry_coadded"],
            "source_extraction": ie_stages["source_extraction"],
            "photometric_calibration": ie_stages["photometric_calibration"],
            "get_photometry": ie_stages["get_photometry"]
        }
        stages["coadd"]["default"] = False
        stages["coadd"]["frames"] = "split"
        stages["correct_astrometry_coadded"]["default"] = True
        return stages

    def add_coadded_esoreflex_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_esoreflex)

    def add_frame_split(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_split, frame_type="reduced")

    def check_filter(self, fil: str):
        not_none = super().check_filter(fil)
        if not_none:
            if fil not in self.frames_split:
                if isinstance(self.frames_split, dict):
                    self.frames_split[fil] = []
            if fil not in self.coadded_esoreflex:
                self.coadded_esoreflex[fil] = None
        return not_none

    def _output_dict(self):
        output_dict = super()._output_dict()
        output_dict.update({
            "coadded_esoreflex": _output_img_dict_single(self.coadded_esoreflex),
            "frames_split": _output_img_dict_list(self.frames_split)
        })

        return output_dict

    def load_output_file(self, **kwargs):
        outputs = super().load_output_file(**kwargs)
        if isinstance(outputs, dict):
            if "coadded_esoreflex" in outputs:
                for fil in outputs["coadded_esoreflex"]:
                    if outputs["coadded_esoreflex"][fil] is not None:
                        u.debug_print(1, f"Attempting to load coadded_esoreflex[{fil}]")
                        self.add_coadded_esoreflex_image(img=outputs["coadded_esoreflex"][fil], key=fil, **kwargs)
            if "frames_split" in outputs:
                for fil in outputs["frames_split"]:
                    if outputs["frames_split"][fil] is not None:
                        for frame in outputs["frames_split"][fil]:
                            self.add_frame_split(frame=frame)

    def _pipeline_init(self):
        super()._pipeline_init()
        self.coadded_final = "coadded_astrometry"
        self.frames_final = "frames_split"

    def sort_after_esoreflex(self, output_dir: str, **kwargs):
        """
        Scans through the ESO Reflex directory for the files matching this epoch, and puts them where we want them.
        :param output_dir:
        :param kwargs:
        :return:
        """
        self.frames_reduced = {}
        self.coadded_esoreflex = {}

        super().sort_after_esoreflex(
            output_dir=output_dir,
            **kwargs
        )

        esodir_root = p.config['esoreflex_output_dir']

        eso_tmp_dir = os.path.join(
            esodir_root,
            "reflex_tmp_products",
            "hawki",
            "hawki_science_process_1"
        )

        tmp_subdirs = os.listdir(eso_tmp_dir)
        mjd = int(self.mjd())
        obj = self.target.lower()

        # Also grab the intermediate, individual chip frames from the reflex temp products directory
        for subdir in tmp_subdirs:
            subpath = os.path.join(eso_tmp_dir, subdir)
            if os.path.isfile(os.path.join(subpath, "exp_1.fits")):
                with fits.open(os.path.join(subpath, "exp_1.fits")) as file:
                    if "OBJECT" in file[0].header:
                        file_obj = file[0].header["OBJECT"].lower()
                    else:
                        continue
                    if "MJD-OBS" in file[0].header:
                        file_mjd = int(file[0].header["MJD-OBS"])
                    else:
                        continue
                    if "FILTER" in file[0].header:
                        fil = file[0].header["FILTER"]
                if file_obj == obj and file_mjd == mjd:
                    fil_destination = os.path.join(
                        output_dir,
                        fil,
                        "frames",
                    )
                    u.mkdir_check(fil_destination)
                    i = 1
                    while os.path.isfile(os.path.join(subpath, f"exp_{i}.fits")):
                        file_path = os.path.join(subpath, f"exp_{i}.fits")
                        new_file_name = f"{self.name}_{self.date_str()}_{fil}_exp_{i}.fits"
                        file_destination = os.path.join(
                            fil_destination,
                            new_file_name
                        )
                        if not self.quiet:
                            print(f"Copying: {file_path} \n\tto \n\t {file_destination}")
                        shutil.copy(file_path, file_destination)
                        img = image.HAWKIImage(path=file_path, frame_type="science")
                        self.add_frame_reduced(img)
                        i += 1

    def _sort_after_esoreflex(
            self,
            output_dir: str,
            date_dir: str,
            obj: str,
            mjd: int,
            delete_output: bool,
            subpath: str,
            **kwargs
    ):
        files = filter(
            lambda f: os.path.isfile(os.path.join(subpath, f)) and f.endswith(".fits"),
            os.listdir(subpath)
        )
        good_dir = False
        for file_name in files:
            file_path = os.path.join(subpath, file_name)
            with fits.open(file_path) as file:
                if "OBJECT" in file[0].header:
                    file_obj = file[0].header["OBJECT"].lower()
                else:
                    continue
                if "MJD-OBS" in file[0].header:
                    file_mjd = int(file[0].header["MJD-OBS"])
                else:
                    continue
                if "FILTER" in file[0].header:
                    fil = file[0].header["FILTER"]
            if file_obj == obj and file_mjd == mjd:
                suffix = file_name[file_name.find("_") + 1:-5]
                new_file_name = f"{self.name}_{self.date_str()}_{fil}_{suffix}.fits"
                fil_destination = os.path.join(
                    output_dir,
                    fil
                )
                u.mkdir_check(fil_destination)
                file_destination = os.path.join(
                    fil_destination,
                    new_file_name
                )
                if not self.quiet:
                    print(f"Copying: {file_path} \n\tto \n\t {file_destination}")
                shutil.copy(file_path, file_destination)
                if file_name.endswith("TILED_IMAGE.fits"):
                    img = self.add_coadded_esoreflex_image(
                        img=file_destination,
                        key=fil
                    )
                    img.set_header_items({
                        "EXPTIME": 1.0,
                        "INTIME": img.extract_header_item("TEXPTIME"),
                    })
                if delete_output and os.path.isfile(file_destination):
                    os.remove(file_path)

    def proc_split_frames(self, output_dir: str, **kwargs):
        self.split_frames(output_dir=output_dir, **kwargs)

    def split_frames(
            self,
            output_dir: str,
            **kwargs
    ):
        for fil in self.frames_reduced:
            for frame in self.frames_reduced[fil]:

                results = frame.split_fits(
                    output_dir=output_dir
                )
                if not self.quiet:
                    print(f"Split {frame} into:")
                for name in results:
                    if not self.quiet:
                        print(f"\t{name}")
                    self.add_frame_split(frame=results[name])

    def coadd(self, output_dir: str, frames: str = "split", sigma_clip: float = 1.5):
        return super().coadd(
            output_dir=output_dir,
            frames=frames,
            sigma_clip=sigma_clip
        )

    def correct_astrometry_coadded(
            self,
            output_dir: str,
            image_type: str = None,
            **kwargs
    ):
        if image_type is None:
            image_type = "coadded_esoreflex"
        super().correct_astrometry_coadded(
            output_dir=output_dir,
            image_type=image_type,
            **kwargs
        )
        if not self.coadded_astrometry:
            self.coadded_final = "coadded_esoreflex"
            self.coadded_unprojected = self.coadded_esoreflex.copy()
        else:
            self.coadded_unprojected = self.coadded_astrometry.copy()

    def _get_images(self, image_type: str) -> Dict[str, image.CoaddedImage]:
        if image_type in ("final", "coadded_final"):
            if self.coadded_final is not None:
                image_type = self.coadded_final
            else:
                raise ValueError("coadded_final has not been set.")

        if image_type in ("coadded_esoreflex", "esoreflex"):
            return self.coadded_esoreflex
        else:
            return super()._get_images(image_type=image_type)

    def _get_frames(self, frame_type: str) -> Dict[str, List[image.ImagingImage]]:
        if frame_type == "final":
            if self.frames_final is not None:
                frame_type = self.frames_final
            else:
                raise ValueError("frames_final has not been set.")

        if frame_type in ("split", "frames_split"):
            image_dict = self.frames_split
        else:
            image_dict = super()._get_frames(
                frame_type=frame_type
            )

        return image_dict


class FORS2ImagingEpoch(ESOImagingEpoch):
    instrument_name = "vlt-fors2"
    frame_class = image.FORS2Image
    coadded_class = image.FORS2CoaddedImage
    eso_name = "FORS2"

    def n_frames(self, fil: str):
        frame_pairs = self.pair_files(self.frames_reduced[fil])
        return len(frame_pairs)

    @classmethod
    def stages(cls):

        eso_stages = super().stages()
        ie_stages = ImagingEpoch.stages()

        stages = {
            "download": eso_stages["download"],
            "initial_setup": eso_stages["initial_setup"],
            "sort_reduced": eso_stages["sort_reduced"],
            "trim_reduced": eso_stages["trim_reduced"],
            "convert_to_cs": eso_stages["convert_to_cs"],
            "register_frames": ie_stages["register_frames"],
            "correct_astrometry_frames": ie_stages["correct_astrometry_frames"],
            "frame_diagnostics": ie_stages["frame_diagnostics"],
            "subtract_background_frames": ie_stages["subtract_background_frames"],
            "coadd": ie_stages["coadd"],
            "correct_astrometry_coadded": ie_stages["correct_astrometry_coadded"],
            "trim_coadded": ie_stages["trim_coadded"],
            "source_extraction": ie_stages["source_extraction"],
            "photometric_calibration": ie_stages["photometric_calibration"],
            "dual_mode_source_extraction": ie_stages["dual_mode_source_extraction"],
            "get_photometry": ie_stages["get_photometry"],
            # "get_photometry_all": ie_stages["get_photometry_all"]
        }

        stages["photometric_calibration"]["skip_retrievable"] = True

        u.debug_print(2, f"FORS2ImagingEpoch.stages(): stages ==", stages)
        return stages

    def _pipeline_init(self):
        super()._pipeline_init()
        self.frames_final = "astrometry"
        # If told not to correct astrometry on frames:
        if "correct_astrometry_frames" in self.do_kwargs and not self.do_kwargs["correct_astrometry_frames"]:
            self.frames_final = "normalised"
            # If told to register frames
            if "register_frames" in self.do_kwargs and self.do_kwargs["register_frames"]:
                self.frames_final = "registered"
            if "frame_diagnostics" in self.do_kwargs and self.do_kwargs["frame_diagnostics"]:
                self.frames_final = "diagnosed"

        self.coadded_final = "coadded_trimmed"

    # def _register(self, frames: dict, fil: str, tmp: image.ImagingImage, n_template: int, output_dir: str, **kwargs):
    #     pairs = self.pair_files(images=frames[fil])
    #     if n_template >= 0:
    #         tmp = pairs[n_template]
    #
    #     for i, pair in enumerate(pairs):
    #         if not isinstance(pair, tuple):
    #             pair = [pair]
    #         if i != n_template:
    #             for j, frame in enumerate(pair):
    #                 if isinstance(tmp, tuple):
    #                     template = tmp[j]
    #                 else:
    #                     template = tmp
    #                 u.debug_print(2, frame.filename.replace("_norm.fits", "_registered.fits"))
    #                 registered = frame.register(
    #                     target=template,
    #                     output_path=os.path.join(
    #                         output_dir,
    #                         frame.filename.replace("_norm.fits", "_registered.fits"))
    #                 )
    #                 self.add_frame_registered(registered)
    #         else:
    #             for j, frame in enumerate(pair):
    #                 registered = frame.copy(
    #                     os.path.join(output_dir, frame.filename.replace("_norm.fits", "_registered.fits")))
    #                 self.add_frame_registered(registered)

    def _sort_after_esoreflex(
            self,
            output_dir: str,
            date_dir: str,
            obj: str,
            mjd: int,
            delete_output: bool,
            subpath: str,
            **kwargs
    ):
        # List directories within 'reduction date' directories.
        # These should represent individual images reduced.

        _, subdirectory = os.path.split(subpath)

        # Get the files within the image directory.
        files = filter(
            lambda d: os.path.isfile(os.path.join(subpath, d)),
            os.listdir(subpath)
        )
        for file_name in files:
            # Retrieve the target object name from the fits file.
            file_path = os.path.join(subpath, file_name)
            inst_file = image.detect_instrument(file_path, fail_quietly=True)
            if inst_file != "vlt-fors2":
                continue
            file = image.from_path(
                path=file_path,
                cls=image.FORS2Image
            )
            file_obj = file.extract_object().lower()
            file_mjd = int(file.extract_header_item('MJD-OBS'))
            file_filter = file.extract_filter()
            # Check the object name and observation date against those of the epoch we're concerned with.
            if file_obj == obj and file_mjd == mjd:
                # Check which type of file we have.
                if file_name.endswith("PHOT_BACKGROUND_SCI_IMG.fits"):
                    file_destination = os.path.join(output_dir, "backgrounds")
                    suffix = "PHOT_BACKGROUND_SCI_IMG.fits"
                    file_type = "background"
                elif file_name.endswith("OBJECT_TABLE_SCI_IMG.fits"):
                    file_destination = os.path.join(output_dir, "obj_tbls")
                    suffix = "OBJECT_TABLE_SCI_IMG.fits"
                    file_type = "object_table"
                elif file_name.endswith("SCIENCE_REDUCED_IMG.fits"):
                    file_destination = os.path.join(output_dir, "science")
                    suffix = "SCIENCE_REDUCED_IMG.fits"
                    file_type = "science"
                else:
                    file_destination = os.path.join(output_dir, "sources")
                    suffix = "SOURCES_SCI_IMG.fits"
                    file_type = "sources"
                # Make this directory, if it doesn't already exist.
                u.mkdir_check(file_destination)
                # Make a subdirectory by filter.
                file_destination = os.path.join(file_destination, file_filter)
                u.mkdir_check(file_destination)
                # Title new file.
                file_destination = os.path.join(
                    file_destination,
                    f"{self.name}_{subdirectory}_{suffix}"
                )
                # Copy file to new location.
                if not self.quiet:
                    print(f"Copying: {file_path} to \n\t {file_destination}")
                file.copy(file_destination)
                if delete_output and os.path.isfile(file_destination):
                    os.remove(file_path)
                img = image.from_path(
                    path=file_destination,
                    cls=image.FORS2Image
                )
                u.debug_print(2, "ESOImagingEpoch._sort_after_esoreflex(): file_type ==", file_type)
                if file_type == "science":
                    self.add_frame_reduced(img)
                elif file_type == "background":
                    self.add_frame_background(img)
        # With the FORS2 substructure we want to search every subdirectory
        return False

    def correct_astrometry_frames(self, output_dir: str, frames: dict = None, **kwargs):
        """

        :param output_dir:
        :param kwargs:
            method: method with which to solve astrometry of epoch. Allowed values are:
                individual: each frame, including separate chips in the same exposure, will be passed to astrometry.net
                    individually. Of the options, this is the most likely to result in an error, especially if the FOV
                    is small; it will also slightly degrade the PSF of the stacked image, as although the accuracy of
                    the WCS of the individual frames is increased, slight errors will be introduced between frames.
                pairwise: the upper-chip image of each pair will first be passed to astrometry.net, and its solution
                    propagated to the bottom chip. If a solution is not found for the top chip, the reverse will be
                    attempted. This method is not recommended, as it will incorrectly capture distortions in the
                    unsolved chip.
                propagate_from_single: Each upper-chip image is passed to astrometry.net until a solution is found; this
                    solution is then propagated to all other upper-chip images. The same is repeated for the lower chip.
        :return:
        """
        self.frames_astrometry = {}
        method = "individual"
        if "method" in kwargs:
            method = kwargs.pop("method")
        upper_only = False
        if "upper_only" in kwargs:
            upper_only = kwargs.pop("upper_only")
        if upper_only and method == "pairwise":
            method = "individual"
        if frames is None:
            frames = self.frames_normalised
        if not self.quiet:
            print()
            print(f"Solving astrometry using method '{method}'")
            print()

        if method == "individual":

            if upper_only:
                frames_upper = {}
                for fil in frames:
                    frames_upper[fil] = []
                    for img in frames[fil]:
                        if img.extract_chip_number() == 1:
                            frames_upper[fil].append(img)
                frames = frames_upper
            super().correct_astrometry_frames(output_dir=output_dir, frames=frames, **kwargs)

        else:
            for fil in frames:
                astrometry_fil_path = os.path.join(output_dir, fil)
                if method == "pairwise":
                    pairs = self.pair_files(frames[fil])
                    reverse_pair = False
                    for pair in pairs:
                        if isinstance(pair, tuple):
                            img_1, img_2 = pair
                            success = False
                            failed_first = False
                            while not success:  # The SystemError should stop this from looping indefinitely.
                                if not reverse_pair:
                                    new_img_1 = img_1.correct_astrometry(
                                        output_dir=astrometry_fil_path,
                                        **kwargs)
                                    # Check if the first astrometry run was successful.
                                    # If it wasn't, we need to be running on the second image of the pair.
                                    if new_img_1 is None:
                                        reverse_pair = True
                                        failed_first = True
                                        self.astrometry_successful[fil][img_1.name] = False
                                        if not self.quiet:
                                            print(
                                                f"Astrometry.net failed to solve {img_1}, trying on opposite chip {img_2}.")
                                    else:
                                        self.add_frame_astrometry(new_img_1)
                                        self.astrometry_successful[fil][img_1.name] = True
                                        new_img_2 = img_2.correct_astrometry_from_other(
                                            new_img_1,
                                            output_dir=astrometry_fil_path,
                                        )
                                        self.add_frame_astrometry(new_img_2)

                                        success = True
                                # We don't use an else statement here because reverse_pair can change within the above
                                # block, and if it does the block below needs to execute.
                                if reverse_pair:
                                    new_img_2 = img_2.correct_astrometry(
                                        output_dir=astrometry_fil_path,
                                        **kwargs)
                                    if new_img_2 is None:
                                        self.astrometry_successful[fil][img_2.name] = False
                                        if failed_first:
                                            raise SystemError(
                                                f"Astrometry.net failed to solve both chips of this pair ({img_1}, {img_2})")
                                        else:
                                            reverse_pair = False
                                    else:
                                        self.add_frame_astrometry(new_img_2)
                                        self.astrometry_successful[fil][img_2.name] = True
                                        new_img_1 = img_1.correct_astrometry_from_other(
                                            new_img_2,
                                            output_dir=astrometry_fil_path,
                                        )
                                        self.add_frame_astrometry(new_img_1)
                                        success = True
                        else:
                            new_img = pair.correct_astrometry(
                                output_dir=astrometry_fil_path,
                                **kwargs)
                            self.add_frame_astrometry(new_img)

                        self.update_output_file()

                elif method == "propagate_from_single":
                    # Sort frames by upper or lower chip.
                    chips = self.sort_by_chip(frames[fil])
                    upper = chips[1]
                    lower = chips[2]
                    if upper_only:
                        lower = []
                    for j, lst in enumerate((upper, lower)):
                        successful = None
                        i = 0
                        while successful is None and i < len(upper):
                            img = lst[i]
                            i += 1
                            new_img = img.correct_astrometry(output_dir=astrometry_fil_path,
                                                             **kwargs)
                            # Check if successful:
                            if new_img is not None:
                                lst.remove(img)
                                self.add_frame_astrometry(new_img)
                                successful = new_img
                                self.astrometry_successful[fil][img.name] = True
                            else:
                                self.astrometry_successful[fil][img.name] = False

                        # If we failed to find a solution on any frame in lst:
                        if successful is None and not self.quiet:
                            print(
                                f"Astrometry.net failed to solve any of the chip {j + 1} images. "
                                f"Chip 2 will not be included in the co-addition.")

                        # Now correct all of the other images in the list with the successful solution.
                        else:
                            for img in lst:
                                new_img = img.correct_astrometry_from_other(
                                    successful,
                                    output_dir=astrometry_fil_path
                                )

                                self.add_frame_astrometry(new_img)

                        self.update_output_file()

                else:
                    raise ValueError(
                        f"Astrometry method {method} not recognised. Must be individual, pairwise or propagate_from_single")

    def estimate_atmospheric_extinction(
            self,
            n: int = 10,
            output: str = None
    ):
        mjd = self.date.mjd
        fils_known = []
        tbls_known = {}

        fils_find = []

        for fil_name in craftutils.observation.filters.eso.vlt_fors2.FORS2Filter.qc1_retrievable:
            fil = filters.Filter.from_params(fil_name, instrument_name="vlt-fors2")
            fil.retrieve_calibration_table()
            fils_known.append(fil)
            tbls_known[fil_name] = fil.get_nearest_calib_rows(mjd=mjd, n=n)

        fils_known.sort(key=lambda f: f.lambda_eff)

        lambdas_known = list(map(lambda f: f.lambda_eff.value, fils_known))

        results_tbl = {
            "mjd": [],
            "curve_err": [],
        }

        for fil_name in self.filters:
            if fil_name not in craftutils.observation.filters.eso.vlt_fors2.FORS2Filter.qc1_retrievable:
                fil = filters.Filter.from_params(fil_name, instrument_name="vlt-fors2")
                fils_find.append(fil)
                results_tbl[f"ext_{fil_name}"] = []
                # results_tbl[f"ext_err_{fil_name}"] = []
                results_tbl[f"stat_err_{fil_name}"] = []

        fils_find.sort(key=lambda f: f.lambda_eff)
        lambdas_find = list(map(lambda f: f.lambda_eff.value, fils_find))

        if output is None:
            output = self.data_path

        for i in range(n):
            extinctions_known = []
            extinctions_known_err = []
            mjd = None
            mjds = []
            for fil in fils_known:
                tbl = tbls_known[fil.name]
                if mjd is None:
                    mjd = tbl[i]["mjd_obs"]
                mjds.append(tbl[i]["mjd_obs"])
                extinctions_known.append(tbl[i]["extinction"].value)
                extinctions_known_err.append(tbl[i]["extinction_err"].value)
            results_tbl["mjd"].append(mjd)
            extinctions_known_err = np.array(extinctions_known_err)
            model_init = models.PowerLaw1D()
            fitter = fitting.LevMarLSQFitter()

            try:
                model = fitter(model_init, np.array(lambdas_known), np.array(extinctions_known),
                               weights=1 / extinctions_known_err)
                curve_err = u.root_mean_squared_error(model_values=model(lambdas_known), obs_values=extinctions_known)
                results_tbl["curve_err"].append(curve_err)
                extinctions_find = model(lambdas_find)
                lambda_eff_fit = np.linspace(3000, 10000)
                plt.close()
                plt.plot(lambda_eff_fit, model(lambda_eff_fit))
                plt.scatter(lambdas_known, extinctions_known, label="Known")
                for j, m in enumerate(mjds):
                    plt.text(lambdas_known[j], extinctions_known[j], fils_known[j])
                plt.scatter(lambdas_find, extinctions_find, label="fitted")
                plt.xlabel("$\lambda_{eff}$ (Ang)")
                plt.ylabel("Extinction (mag)")
                try:
                    plt.savefig(os.path.join(output, f"extinction_fit_mjd_{mjd}.png"))
                except TypeError:
                    pass
                plt.close()

                for fil in fils_find:
                    results_tbl[f"ext_{fil.name}"].append(model(fil.lambda_eff.value))

            except fitting.NonFiniteValueError:
                print("Fitting failed for MJD", mjd)
                results_tbl["curve_err"].append(np.nan)
                for fil in fils_find:
                    results_tbl[f"ext_{fil.name}"].append(np.nan)

        for fil in fils_find:
            results_tbl[f"stat_err_{fil.name}"] = [np.std(results_tbl[f"ext_{fil.name}"])] * n

        results_tbl = table.QTable(results_tbl)
        for fil in fils_find:
            results_tbl[f"ext_err_{fil.name}"] = np.sqrt(
                results_tbl[f"stat_err_{fil.name}"] ** 2 + results_tbl[f"curve_err"] ** 2) * units.mag
            results_tbl[f"stat_err_{fil.name}"] *= units.mag
            results_tbl[f"ext_{fil.name}"] *= units.mag
        results_tbl[f"curve_err"] *= units.mag

        i, nrst = u.find_nearest(results_tbl["mjd"], self.date.mjd)

        results_tbl.write(os.path.join(output, "fitted_extinction.csv"), format="ascii.csv")

        return results_tbl[i], results_tbl

    def photometric_calibration_from_standards(
            self,
            image_dict: dict,
            output_path: str,
    ):

        import craftutils.wrap.esorex as esorex

        ext_row, ext_tbl = self.estimate_atmospheric_extinction(output=output_path)
        # image_dict = self._get_images(image_type=image_type)
        for fil in image_dict:
            img = image_dict[fil]
            if f"ext_{fil}" in ext_row.colnames:
                img.extinction_atmospheric = ext_row[f"ext_{fil}"]
                img.extinction_atmospheric_err = ext_row[f"ext_err_{fil}"]

        # Do esorex reduction of standard images, and attempt esorex zeropoints if there are enough different
        # observations
        # image_dict = self._get_images(image_type)
        # Split up bias images by chip
        bias_sets = self.sort_by_chip(self.frames_bias)

        if 1 in bias_sets and 2 in bias_sets:
            bias_sets = (bias_sets[1], bias_sets[2])
            flat_sets = {}
            std_sets = {}
            # Split up the flats and standards by filter and chip
            for fil in self.filters:
                flat_chips = self.sort_by_chip(self.frames_flat[fil])
                if flat_chips:
                    flat_sets[fil] = flat_chips[1], flat_chips[2]
                std_chips = self.sort_by_chip(self.frames_standard[fil])
                if std_chips:
                    std_sets[fil] = std_chips[1], std_chips[2]

            chips = ("up", "down")
            for i, chip in enumerate(chips):
                bias_set = bias_sets[i]
                # For each chip, generate a master bias image
                try:
                    master_bias = esorex.fors_bias(
                        bias_frames=list(map(lambda b: b.output_file, bias_set)),
                        output_dir=output_path,
                        output_filename=f"master_bias_{chip}.fits",
                        sof_name=f"bias_{chip}.sof"
                    )
                except SystemError:
                    continue

                for fil in image_dict:
                    # Generate master flat per-filter, per-chip
                    if fil not in flat_sets or fil in craftutils.observation.filters.eso.vlt_fors2.FORS2Filter.qc1_retrievable:  # Time-saver
                        continue
                    img = image_dict[fil]
                    if "calib_pipeline" in img.zeropoints:
                        img.zeropoints.pop("calib_pipeline")
                    flat_set = list(map(lambda b: b.output_file, flat_sets[fil][i]))
                    fil_dir = os.path.join(output_path, fil)
                    u.mkdir_check(fil_dir)
                    try:
                        master_sky_flat_img = esorex.fors_img_sky_flat(
                            flat_frames=flat_set,
                            master_bias=master_bias,
                            output_dir=fil_dir,
                            output_filename=f"master_sky_flat_img_{chip}.fits",
                            sof_name=f"flat_{chip}"
                        )
                    except SystemError:
                        continue

                    aligned_phots = []
                    if fil in std_sets:
                        for std in std_sets[fil][i]:
                            # generate or load an appropriate StandardEpoch
                            # (and StandardField in the background)
                            pointing = std.extract_pointing()
                            jname = astm.jname(pointing, 0, 0)
                            if pointing not in self.std_pointings:
                                self.std_pointings.append(pointing)
                            if jname not in self.std_epochs:
                                std_epoch = FORS2StandardEpoch(
                                    centre_coords=pointing,
                                    instrument=self.instrument,
                                    frames_flat=self.frames_flat,
                                    frames_bias=self.frames_bias,
                                    date=self.date
                                )
                                self.std_epochs[jname] = std_epoch
                            else:
                                std_epoch = self.std_epochs[jname]
                            std_epoch.add_frame_raw(std)
                            # For each raw standard, reduce
                            std_dir = os.path.join(fil_dir, std.name)
                            u.mkdir_check(std_dir)
                            aligned_phot, std_reduced = esorex.fors_zeropoint(
                                standard_img=std.output_file,
                                master_bias=master_bias,
                                master_sky_flat_img=master_sky_flat_img,
                                output_dir=std_dir,
                                chip_num=i + 1
                            )
                            aligned_phots.append(aligned_phot)
                            std_epoch.add_frame_reduced(std_reduced)

                        if len(aligned_phots) > 1:
                            try:
                                phot_coeff_table = esorex.fors_photometry(
                                    aligned_phot=aligned_phots,
                                    master_sky_flat_img=master_sky_flat_img,
                                    output_dir=fil_dir,
                                    chip_num=i + 1,
                                )

                                phot_coeff_table = fits.open(phot_coeff_table)[1].data

                                u.debug_print(1, f"Chip {chip}, zeropoint {phot_coeff_table['ZPOINT'][0] * units.mag}")

                                # The intention here is that a chip 1 zeropoint override a chip 2 zeropoint, but
                                # if chip 1 doesn't work a chip 2 one will do.
                                if chip == 1 or "calib_pipeline" not in img.zeropoints:
                                    img.add_zeropoint(
                                        zeropoint=phot_coeff_table["ZPOINT"][0] * units.mag,
                                        zeropoint_err=phot_coeff_table["DZPOINT"][0] * units.mag,
                                        airmass=img.extract_airmass(),
                                        airmass_err=self.airmass_err[fil],
                                        extinction=phot_coeff_table["EXT"][0] * units.mag,
                                        extinction_err=phot_coeff_table["DEXT"][0] * units.mag,
                                        catalogue="calib_pipeline",
                                        n_matches=None,
                                    )

                                # img.update_output_file()
                            except SystemError:
                                if not self.quiet:
                                    print(
                                        "System error encountered while doing esorex processing; possibly impossible value encountered. Skipping.")

                        else:
                            print(f"Insufficient standard observations to calculate esorex zeropoint for {img}")
            if not self.quiet:
                print("Estimating zeropoints from standard observations...")
            for jname in self.std_epochs:
                std_epoch = self.std_epochs[jname]
                std_epoch.photometric_calibration()
                for fil in image_dict:
                    img = image_dict[fil]
                    # We save time by only bothering with non-qc1-obtainable zeropoints.
                    if fil in std_epoch.frames_reduced and fil not in craftutils.observation.filters.eso.vlt_fors2.FORS2Filter.qc1_retrievable:
                        for std in std_epoch.frames_reduced[fil]:
                            img.add_zeropoint_from_other(std)

    def photometric_calibration(
            self,
            output_path: str,
            image_dict: dict,
            **kwargs
    ):

        # if "image_type" in kwargs and kwargs["image_type"] is not None:
        #     image_type = kwargs["image_type"]
        # else:
        #     image_type = "final"

        # skip_retrievable = True
        # if "skip_retrievable" in kwargs and kwargs["skip_retrievable"] is not None:
        #     skip_retrievable = kwargs.pop("skip_retrievable")

        suppress_select = True
        if "suppress_select" in kwargs and kwargs["suppress_select"] is not None:
            suppress_select = kwargs.pop("suppress_select")

        if not self.combined_epoch:
            self.photometric_calibration_from_standards(
                image_dict=image_dict,
                output_path=output_path,
                # skip_retrievable=skip_retrievable
            )

        zeropoints = p.load_params(zeropoint_yaml)
        if zeropoints is None:
            zeropoints = {}

        super().photometric_calibration(
            output_path=output_path,
            suppress_select=True,
            image_dict=image_dict,
            **kwargs
        )

        for fil in image_dict:
            if "preferred_zeropoint" in kwargs and fil in kwargs["preferred_zeropoint"]:
                preferred = kwargs["preferred_zeropoint"][fil]
            else:
                preferred = None
            img = image_dict[fil]

            img.select_zeropoint(suppress_select, preferred=preferred)

            if fil not in zeropoints:
                zeropoints[fil] = {}
            for cat in img.zeropoints:
                if cat not in zeropoints[fil]:
                    zeropoints[fil][cat] = {}
                zeropoints[fil][cat][self.date_str()] = img.zeropoints[cat]

            # Transfer derived zeropoints to the unprojected versions (if they exist and are distinct)
            if self.coadded_unprojected[fil] is not None and self.coadded_unprojected[fil] is not img:
                self.coadded_unprojected[fil].zeropoints = img.zeropoints
                self.coadded_unprojected[fil].zeropoint_best = img.zeropoint_best
                self.coadded_unprojected[fil].update_output_file()
            if self.coadded_subtracted[fil] is not None:
                self.coadded_subtracted[fil].zeropoints = img.zeropoints
                self.coadded_subtracted[fil].zeropoint_best = img.zeropoint_best
                self.coadded_subtracted[fil].update_output_file()

        p.save_params(zeropoint_yaml, zeropoints)

    @classmethod
    def pair_files(cls, images: list):
        pairs = []
        images.sort(key=lambda im: im.name)
        is_paired = True
        for i, img_1 in enumerate(images):
            # If the images are in pairs, it's sufficient to check only the even-numbered ones.
            # If not, is_paired=False should be triggered by the case below.
            if i % 2 == 0 or not is_paired:
                chip_this = img_1.extract_chip_number()
                # If we are at the end of the list and still checking, this must be unpaired.
                if i + 1 == len(images):
                    pair = img_1
                else:
                    # Get the next image in the list.
                    img_2 = images[i + 1]
                    chip_other = img_2.extract_chip_number()
                    # If we have chip
                    if (chip_this == 1 and chip_other == 2) or (chip_this == 2 and chip_other == 1):
                        img_1.other_chip = img_2
                        img_1.update_output_file()
                        img_2.other_chip = img_1
                        img_2.update_output_file()
                        if chip_this == 1:
                            pair = (img_1, img_2)
                        elif chip_this == 2:
                            pair = (img_2, img_1)
                        else:
                            raise ValueError("Image is missing chip.")
                        is_paired = True
                    else:
                        is_paired = False
                        pair = img_1
                if isinstance(pair, tuple):
                    u.debug_print(1, str(pair[0]), ",", str(pair[1]))
                else:
                    u.debug_print(1, pair)
                pairs.append(pair)

        return pairs

    @classmethod
    def from_file(
            cls,
            param_file: Union[str, dict],
            name: str = None,
            old_format: bool = False,
            field: 'fld.Field' = None
    ):

        if old_format:
            if name is None:
                raise ValueError("name must be provided for old_format=True.")
            param_file = cls.convert_old_params(old_epoch_name=name)

        return super().from_file(param_file=param_file, name=name, field=field)

    @classmethod
    def convert_old_params(cls, old_epoch_name: str):
        new_params = cls.new_yaml(name=old_epoch_name, path=None)
        old_params = p.object_params_fors2(old_epoch_name)

        new_epoch_name = f"FRB20{old_epoch_name[3:]}"

        old_field = old_epoch_name[:old_epoch_name.find('_')]
        new_field = new_epoch_name[:new_epoch_name.find('_')]

        old_data_dir = old_params["data_dir"]
        i = old_data_dir.find("MJD")
        mjd = old_data_dir[i + 3:i + 8]
        t = Time(mjd, format="mjd")
        date = t.strftime("%Y-%m-%d")
        new_data_dir = os.path.join(p.config['top_data_dir'], new_field, "imaging", "vlt-fors2",
                                    f"{date}-{new_epoch_name}")

        new_params["instrument"] = "vlt-fors2"
        new_params["data_path"] = new_data_dir
        new_params["field"] = new_field
        new_params["name"] = new_epoch_name
        new_params["date"] = date

        new_params["sextractor"]["aperture_diameters"] = old_params["photometry_apertures"]
        new_params["sextractor"]["dual_mode"] = old_params["do_dual_mode"]
        new_params["sextractor"]["threshold"] = old_params["threshold"]
        new_params["sextractor"]["kron_factor"] = old_params["sextractor_kron_radius"]
        new_params["sextractor"]["kron_radius_min"] = old_params["sextractor_min_radius"]

        filters = filter(lambda k: k.endswith("_star_class_tol"), old_params)
        filters = list(map(lambda f: f[0], filters))

        new_params["background_subtraction"]["renormalise_centre"]["dec"] = old_params["renormalise_centre_dec"]
        new_params["background_subtraction"]["renormalise_centre"]["ra"] = old_params["renormalise_centre_ra"]
        new_params["background_subtraction"]["test_synths"] = []
        if old_params["test_synths"]["ra"]:
            for i, _ in enumerate(old_params["test_synths"]):
                synth_dict = {"position": {}}
                synth_dict["position"]["ra"] = old_params["test_synths"]["ra"][i]
                synth_dict["position"]["dec"] = old_params["test_synths"]["dec"][i]
                synth_dict["mags"] = {}
                synth_dict["mags"]["g"] = old_params["test_synths"]["g_mag"][i]
                synth_dict["mags"]["I"] = old_params["test_synths"]["I_mag"][i]
                new_params["background_subtraction"]["test_synths"].append(synth_dict)

        new_params["skip"]["esoreflex_copy"] = old_params["skip_copy"]
        new_params["skip"]["sextractor_individual"] = not old_params["do_sextractor_individual"]
        new_params["skip"]["astrometry_net"] = old_params["skip_astrometry"]
        new_params["skip"]["sextractor"] = not old_params["do_sextractor"]
        new_params["skip"]["esorex"] = old_params["skip_esorex"]

        instrument_path = os.path.join(p.param_dir, "fields", new_field, "imaging", "vlt-fors2")
        u.mkdir_check(instrument_path)
        output_path = os.path.join(instrument_path, new_epoch_name)
        p.save_params(file=output_path,
                      dictionary=new_params)

        return output_path
