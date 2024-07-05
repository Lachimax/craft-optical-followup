# Code by Lachlan Marnoch, 2021 - 2024

import os
import shutil
import datetime
from typing import Union

from astropy.time import Time
import astropy.units as units

import craftutils.params as p
import craftutils.retrieve as retrieve
import craftutils.utils as u
import craftutils.observation.field as fld
import craftutils.observation.image as image
from craftutils.observation.pipeline import Pipeline
from craftutils.observation.instrument import Instrument

config = p.config

active_epochs = {}


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
            from .imaging.epoch import ImagingEpoch
            epoch = ImagingEpoch.from_params(
                epoch_name,
                instrument=instrument,
                field=field,
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


def expunge_epochs():
    epoch_list = list(active_epochs.keys())
    for epoch_name in epoch_list:
        del active_epochs[epoch_name]


def _output_img_list(lst: list):
    """
    Turns a list of images into a YAML-able list of paths.
    :param lst:
    :return:
    """
    out_list = []
    for img in lst:
        out_list.append(img.path)
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


class Epoch(Pipeline):
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
            do_runtime: Union[list, str] = None,
            **kwargs
    ):

        super().__init__(
            param_path=param_path,
            name=name,
            data_path=data_path,
            do_runtime=do_runtime,
            **kwargs
        )

        # Input attributes

        self.field = field

        u.debug_print(2, f"__init__(): {self.name}.data_path ==", self.data_path)
        self.instrument_name = instrument
        try:
            self.instrument = Instrument.from_params(instrument_name=str(instrument))
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

        self.binning = None
        self.binning_std = None

        # Frames
        self.frames_raw = []
        self.frames_bias = []
        self.frames_standard = {}
        self.frames_science = []
        self.frames_dark = []
        self.frames_flat = {}

        self.frames_reduced = []

        # Master calib files
        self.master_biases = {}
        self.master_flats = {}
        self.fringe_maps = {}

        self.coadded = {}

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

        self.combined_from = []

        self.combined_epoch = False
        if "combined_epoch" in kwargs:
            self.combined_epoch = kwargs["combined_epoch"]

        self.exclude_frames: list = []
        if "exclude_frames" in kwargs and isinstance(kwargs["exclude_frames"], list):
            self.exclude_frames = kwargs["exclude_frames"]

        active_epochs[self.name] = self

    def is_excluded(self, frame: Union[image.Image, str]):
        if isinstance(frame, image.Image):
            ident = frame.name
        else:
            ident = str(frame)
        for fname in self.exclude_frames:
            if fname in ident:
                return True
        return False

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self)

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
        outputs = super().load_output_file(**kwargs)
        if isinstance(outputs, dict):
            if "coadded" in outputs:
                for fil in outputs["coadded"]:
                    if outputs["coadded"][fil] is not None:
                        self.add_coadded_image(img=outputs["coadded"][fil], key=fil, **kwargs)
            if "frames_bias" in outputs:
                for frame in set(outputs["frames_bias"]):
                    if os.path.isfile(frame):
                        self.add_frame_raw(raw_frame=frame)
        return outputs

    def _output_dict(self):
        output_dict = super()._output_dict()
        output_dict.update({
            "combined_from": self.combined_from,
            "coadded": _output_img_dict_single(self.coadded),
            "date": self.date,
            "frames_science": _output_img_dict_list(self.frames_science),
            "frames_flat": _output_img_dict_list(self.frames_flat),
            "frames_std": _output_img_dict_list(self.frames_standard),
            "frames_bias": _output_img_list(self.frames_bias),
            "fringe_maps": self.fringe_maps,
            "log": self.log.to_dict(),
            "master_biases": self.master_biases,
            "master_flats": self.master_flats,
        })
        return output_dict

    # def set_survey(self):

    def _pipeline_init(self, skip_cats: bool = False):
        super()._pipeline_init()
        self.set_path(
            key="download",
            value=os.path.join(self.data_path, "0-download")
        )

    def set_program_id(self, program_id: str):
        self.program_id = program_id
        self.update_param_file("program_id")

    def set_date(self, date: Union[str, Time]):
        if isinstance(date, str):
            date = Time(date)
        if date is not None:
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

    def get_master_bias(self, chip: int):
        return self.master_biases[chip]

    def get_master_flat(self, chip: int, fil: str):
        return self.master_flats[chip][fil]

    def _updateable(self):
        p_dict = super()._updateable()
        p_dict.update({
            "program_id": self.program_id,
            "date": self.date,
            "target": self.target
        })
        return p_dict

    @classmethod
    def sort_by_chip(cls, images: list):
        chips = {}

        images.sort(key=lambda f: f.name)

        for img in images:
            chip_this = img.extract_chip_number()
            img.extract_n_pix()
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
        print("Attempting to add raw frame", raw_frame)
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
            "mode": cls.mode,
            "name": None,
            "exclude_frames": [],
            "field": None,
            "data_path": None,
            "instrument": cls.instrument_name,
            "date": None,
            "target": None,
            "program_id": None,
            "do": {},
            "notes": [],
            "combined_epoch": False,
            "validation_copy_of": None
        }
        # Pull the list of applicable kwargs from the stage information
        stages = cls.stages()
        for stage in stages:
            stage_info = stages[stage]
            if stage_info is not None and "keywords" in stage_info:
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



def _retrieve_eso_epoch(
        epoch: Union['ESOImagingEpoch', 'ESOSpectroscopyEpoch'],
        path: str
):
    from .spectroscopy import ESOSpectroscopyEpoch
    from .imaging import ESOImagingEpoch
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
