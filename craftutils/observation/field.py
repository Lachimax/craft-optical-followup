# Code by Lachlan Marnoch, 2021
import datetime
import os
import warnings
from typing import Union, List, Dict
import shutil

import matplotlib.pyplot as plt
import numpy as np

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as units
import astropy.table as table
import astropy.io.fits as fits
from astropy.modeling import models, fitting
from astropy.visualization import make_lupton_rgb, ImageNormalize

import craftutils.astrometry as astm
import craftutils.fits_files as ff
import craftutils.observation as obs
import craftutils.observation.objects as objects
import craftutils.observation.image as image
import craftutils.observation.instrument as inst
import craftutils.observation.log as log
import craftutils.observation.survey as survey
import craftutils.params as p
import craftutils.plotting as pl
import craftutils.retrieve as retrieve
import craftutils.wrap.pypeit as spec
import craftutils.utils as u
import craftutils.wrap.montage as montage
import craftutils.wrap.dragons as dragons

# pl.latex_setup()

config = p.config

instruments_imaging = p.instruments_imaging
instruments_imaging.sort()
instruments_spectroscopy = p.instruments_spectroscopy
instruments_spectroscopy.sort()
surveys = p.surveys

active_fields = {}
active_epochs = {}

zeropoint_yaml = os.path.join(p.data_path, f"zeropoints.yaml")


def expunge_fields():
    for field_name in active_fields:
        del active_fields[field_name]


def expunge_epochs():
    for epoch_name in active_epochs:
        del active_fields[epoch_name]


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


def select_instrument(mode: str):
    if mode == "imaging":
        options = instruments_imaging
    elif mode == "spectroscopy":
        options = instruments_spectroscopy
    else:
        raise ValueError("Mode must be 'imaging' or 'spectroscopy'.")
    _, instrument = u.select_option("Select an instrument:", options=options)
    return instrument


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


def epoch_from_directory(epoch_name: str):
    directory = load_epoch_directory()
    print(f"Looking for {epoch_name} in directory...")
    epoch = None
    if epoch_name in directory:
        epoch_dict = directory[epoch_name]
        field_name = epoch_dict["field_name"]
        instrument = epoch_dict["instrument"]
        mode = epoch_dict["mode"]
        field = Field.from_params(name=field_name)
        if mode == "imaging":
            epoch = ImagingEpoch.from_params(epoch_name, instrument=instrument, field=field)
        elif mode == "spectroscopy":
            epoch = SpectroscopyEpoch.from_params(name=epoch_name, field=field, instrument=instrument)
        return epoch


def write_epoch_directory(directory: dict):
    """
    Writes the passed dict to the directory.yaml file in param directory.
    :param directory: The updated directory as a dict.
    :return:
    """
    p.save_params(_epoch_directory_path(), directory)


def add_to_epoch_directory(field_name: str, instrument: str, mode: str, epoch_name: str):
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


def list_fields(include_std: bool = False):
    print("Searching for field param files...")
    param_path = os.path.join(config['param_dir'], 'fields')
    fields = list(filter(lambda d: os.path.isdir(os.path.join(param_path, d)) and os.path.isfile(
        os.path.join(param_path, d, f"{d}.yaml")), os.listdir(param_path)))
    if not include_std:
        fields = list(filter(lambda f: "STD" not in f, fields))
    fields.sort()
    return fields


def list_fields_old():
    print("Searching for old-format field param files...")
    param_path = os.path.join(config['param_dir'], 'FRBs')
    if os.path.isdir(param_path):
        fields = filter(
            lambda d: os.path.isfile(os.path.join(param_path, d)) and d.endswith('.yaml'),
            os.listdir(param_path))
        fields = list(map(lambda f: f.split(".")[0], fields))
    else:
        fields = []
    fields.sort()
    return fields


def _retrieve_eso_epoch(epoch: Union['ESOImagingEpoch', 'ESOSpectroscopyEpoch'], path: str):
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


class Field:
    def __init__(
            self,
            name: str = None,
            centre_coords: Union[SkyCoord, str] = None,
            param_path: str = None,
            data_path: str = None,
            objs: Union[List[objects.Object], dict] = None,
            extent: units.Quantity = None,
            **kwargs
    ):
        """

        :param centre_coords:
        :param name:
        :param param_path:
        :param data_path:
        :param objs: a list of objects of interest in the field. The primary object of interest should be first in
        the list.
        """

        # Input attributes

        self.objects = []
        self.objects_dict = {}

        if centre_coords is None:
            if objs is not None:
                centre_coords = objs[0].coords
        if centre_coords is not None:
            self.centre_coords = astm.attempt_skycoord(centre_coords)

        self.name = name
        self.param_path = param_path
        self.param_dir = None
        if self.param_path is not None:
            self.param_dir = os.path.split(self.param_path)[0]
        self.mkdir_params()
        self.data_path = os.path.join(p.data_path, data_path)
        u.debug_print(2, f"Field.__init__(): {self.name}.data_path", self.data_path)
        self.data_path_relative = data_path
        self.mkdir()
        self.output_file = None

        # Derived attributes

        self.epochs_spectroscopy = {}
        self.epochs_spectroscopy_loaded = {}
        self.epochs_imaging = {}
        self.epochs_imaging_loaded = {}

        self.paths = {}

        self.cats = {}
        self.cat_gaia = None
        self.irsa_extinction = None

        if type(objs) is dict:
            for obj_name in objs:
                if obj_name != "<name>":
                    obj_dict = objs[obj_name]
                    if "position" not in obj_dict:
                        obj_dict = {"position": obj_dict}
                    if "name" not in obj_dict:
                        obj_dict["name"] = obj_name
                    self.add_object_from_dict(obj_dict)

        self.load_output_file()

        self.extent = extent

        self.survey = None
        if "survey" in kwargs:
            self.survey = kwargs["survey"]
        if isinstance(self.survey, str):
            self.survey = survey.Survey.from_params(self.survey)

        active_fields[self.name] = self

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return self.__str__()

    def mkdir(self):
        if self.data_path is not None:
            u.mkdir_check(self.data_path)
            u.mkdir_check(os.path.join(self.data_path, "objects"))

    def mkdir_params(self):
        if self.param_dir is not None:
            u.mkdir_check(os.path.join(self.param_dir, "spectroscopy"))
            u.mkdir_check(os.path.join(self.param_dir, "imaging"))
            u.mkdir_check(os.path.join(self.param_dir, "objects"))
        else:
            warnings.warn(f"param_dir is not set for this {type(self)}.")

    def _gather_epochs(self, mode: str = "imaging", quiet: bool = False):
        """
        Helper method for code reuse in gather_epochs_spectroscopy() and gather_epochs_imaging().
        Gathers all of the observation epochs of the given mode for this field.
        :param mode: str, "imaging" or "spectroscopy"
        :return: Dict, with keys being the epoch names and values being nested dictionaries containing the same
        information as the epoch .yaml files.
        """
        if not quiet:
            print(f"Searching for {mode} epoch param files...")
        epochs = {}
        if self.param_dir is not None:
            mode_path = os.path.join(self.param_dir, mode)
            if not quiet:
                print(f"Looking in {mode_path}")
            for instrument in filter(lambda d: os.path.isdir(os.path.join(mode_path, d)), os.listdir(mode_path)):
                instrument_path = os.path.join(mode_path, instrument)
                if not quiet:
                    print(f"Looking in {instrument_path}")
                epoch_params = list(filter(lambda f: f.endswith(".yaml"), os.listdir(instrument_path)))
                epoch_params.sort()
                for epoch_param in epoch_params:
                    epoch_name = epoch_param[:epoch_param.find(".yaml")]
                    param_path = os.path.join(instrument_path, epoch_param)
                    epoch = p.load_params(file=param_path)
                    epoch["format"] = "current"
                    epoch["param_path"] = param_path
                    epochs[epoch_name] = epoch

        add_many_to_epoch_directory(epochs, field_name=self.name, mode=mode)

        return epochs

    def gather_epochs_spectroscopy(self):
        """
        Gathers all of the spectroscopy observation epochs of this field.
        :return: Dict, with keys being the epoch names and values being nested dictionaries containing the same
        information as the epoch .yaml files.
        """
        epochs = self._gather_epochs(mode="spectroscopy")
        self.epochs_spectroscopy.update(epochs)
        return epochs

    def gather_epochs_imaging(self, quiet: bool = False):
        """
        Gathers all of the imaging observation epochs of this field.
        :return: Dict, with keys being the epoch names and values being nested dictionaries containing the same
        information as the epoch .yaml files.
        """
        epochs = self._gather_epochs(mode="imaging", quiet=quiet)
        self.epochs_imaging.update(epochs)
        return epochs

    def epoch_from_params(self, epoch_name: str, instrument: str, old_format: bool = False):
        epoch = ImagingEpoch.from_params(name=epoch_name, field=self, instrument=instrument, old_format=old_format)
        self.epochs_imaging[epoch_name] = epoch
        return epoch

    def select_epoch_imaging(self):
        options = {}
        for epoch in self.epochs_imaging:
            epoch = self.epochs_imaging[epoch]
            date_string = ""
            if epoch["format"] == 'old':
                date_string = " (old format)           "
            elif "date" in epoch and epoch["date"] is not None:
                if isinstance(epoch["date"], str):
                    date_string = f" {epoch['date']}"
                else:
                    date_string = f" {epoch['date'].strftime('%Y-%m-%d')}"
            options[f'{epoch["name"]}\t{date_string}\t{epoch["instrument"]}'] = epoch
        for epoch in self.epochs_imaging_loaded:
            # If epoch is already instantiated.
            epoch = self.epochs_spectroscopy_loaded[epoch]
            options[f'*{epoch.name}\t{epoch.date.isot}\t{epoch.instrument_name}'] = epoch
        options["New epoch"] = "new"
        j, epoch = u.select_option(message="Select epoch.", options=options, sort=True)
        if epoch == "new":
            epoch = self.new_epoch_imaging()
        elif not isinstance(epoch, Epoch):
            old_format = False
            if epoch["format"] == "old":
                old_format = True
            epoch = ImagingEpoch.from_file(epoch, old_format=old_format, field=self)
            self.epochs_imaging_loaded[epoch.name] = epoch
        return epoch

    def select_epoch_spectroscopy(self):
        options = {}
        for epoch in self.epochs_spectroscopy:
            epoch = self.epochs_spectroscopy[epoch]
            options[f"{epoch['name']}\t{epoch['date'].to_datetime().date()}\t{epoch['instrument']}"] = epoch
        for epoch in self.epochs_spectroscopy_loaded:
            epoch = self.epochs_spectroscopy_loaded[epoch]
            options[f'*{epoch.name}\t{epoch.date.isot}\t{epoch.instrument_name}'] = epoch
        options["New epoch"] = "new"
        j, epoch = u.select_option(message="Select epoch.", options=options)
        if epoch == "new":
            epoch = self.new_epoch_spectroscopy()
        elif not isinstance(epoch, Epoch):
            epoch = SpectroscopyEpoch.from_file(epoch, field=self)
            self.epochs_spectroscopy_loaded[epoch.name] = epoch
        return epoch

    def new_epoch_imaging(self):
        return self._new_epoch(mode="imaging")

    def new_epoch_spectroscopy(self):
        return self._new_epoch(mode="spectroscopy")

    def _new_epoch(self, mode: str):
        """
        Helper method for generating a new epoch.
        :param mode:
        :return:
        """
        instrument = select_instrument(mode=mode)
        if mode == "imaging":
            cls = ImagingEpoch.select_child_class(instrument=instrument)
        elif mode == "spectroscopy":
            cls = SpectroscopyEpoch.select_child_class(instrument=instrument)
        else:
            raise ValueError("mode must be 'imaging' or 'spectroscopy'.")
        new_params = cls.default_params()
        if instrument in surveys:
            new_params["name"] = instrument.upper()
            new_params["date"] = None
            new_params["program_id"] = None
            survey = True
        else:
            survey = False
            new_params["name"] = u.user_input("Please enter a name for the epoch.")
            # new_params["date"] = u.enter_time(message="Enter UTC observation date, in iso or isot format:").strftime(
            #     '%Y-%m-%d')
            # new_params["program_id"] = input("Enter the programmme ID for the observation:\n")
        new_params["instrument"] = instrument
        new_params["data_path"] = self._epoch_data_path(
            mode=mode,
            instrument=instrument,
            date=new_params["date"],
            epoch_name=new_params["name"],
            survey=survey)
        new_params["field"] = self.name
        param_path = self._epoch_param_path(mode=mode, instrument=instrument, epoch_name=new_params["name"])

        p.save_params(file=param_path, dictionary=new_params)
        epoch = cls.from_file(param_file=param_path, field=self)

        return epoch

    def _mode_param_path(self, mode: str):
        if self.param_dir is not None:
            path = os.path.join(self.param_dir, mode)
            u.mkdir_check(path)
            return path
        else:
            raise ValueError(f"param_dir is not set for {self}.")

    def _mode_data_path(self, mode: str):
        if self.data_path is not None:
            path = os.path.join(self.data_path_relative, mode)
            path_abs = os.path.join(self.data_path, mode)
            u.mkdir_check(path_abs)
            return path, path_abs
        else:
            raise ValueError(f"data_path is not set for {self}.")

    def _cat_data_path(self, cat: str):
        if self.data_path is not None:
            filename = f"{cat}_{self.name}.csv"
            path = os.path.join(self.data_path, filename)
            return path
        else:
            raise ValueError(f"data_path is not set for {self}.")

    def _instrument_param_path(self, mode: str, instrument: str):
        path = os.path.join(self._mode_param_path(mode=mode), instrument)
        u.mkdir_check(path)
        return path

    def _instrument_data_path(self, mode: str, instrument: str):
        path, path_abs = self._mode_data_path(mode=mode)
        path = os.path.join(path, instrument)
        path_abs = os.path.join(path_abs, instrument)
        u.mkdir_check(path_abs)
        return path, path_abs

    def _epoch_param_path(self, mode: str, instrument: str, epoch_name: str):
        return os.path.join(self._instrument_param_path(mode=mode, instrument=instrument), f"{epoch_name}.yaml")

    def _epoch_data_path(self, mode: str, instrument: str, date: Time, epoch_name: str, survey: bool = False):
        if survey:
            path, path_abs = self._instrument_data_path(mode=mode, instrument=instrument)
        else:
            if date is None:
                name_str = epoch_name
            else:
                name_str = f"{date}-{epoch_name}"
            path, path_abs = self._instrument_data_path(mode=mode, instrument=instrument)
            path = os.path.join(path, name_str)
            path_abs = os.path.join(path_abs, name_str)
        u.mkdir_check(path_abs)
        return path

    def retrieve_catalogues(self, force_update: bool = False):
        for cat_name in retrieve.photometry_catalogues:
            u.debug_print(1, f"Checking for photometry in {cat_name}")
            self.retrieve_catalogue(cat_name=cat_name, force_update=force_update)

    def retrieve_catalogue(self, cat_name: str, force_update: bool = False):
        if isinstance(self.extent, units.Quantity):
            radius = self.extent
        else:
            radius = 0.2 * units.deg
        output = self._cat_data_path(cat=cat_name)
        ra = self.centre_coords.ra.value
        dec = self.centre_coords.dec.value
        if force_update or f"in_{cat_name}" not in self.cats:
            u.debug_print(2, "Field.retrieve_catalogue(): radius ==", radius)
            response = retrieve.save_catalogue(
                ra=ra, dec=dec, output=output, cat=cat_name.lower(),
                radius=radius)
            # Check if a valid response was received; if not, we don't want to erroneously report that
            # the field doesn't exist in the catalogue.
            if isinstance(response, str) and response == "ERROR":
                pass
            else:
                if response is not None:
                    self.cats[f"in_{cat_name}"] = True
                    self.set_path(f"cat_csv_{cat_name}", output)
                else:
                    self.cats[f"in_{cat_name}"] = False
                self.update_output_file()
            return response
        elif self.cats[f"in_{cat_name}"] is True:
            u.debug_print(1, f"There is already {cat_name} data present for this field.")
            return True
        else:
            u.debug_print(1, f"This field is not present in {cat_name}.")

    def load_catalogue(self, cat_name: str):
        if self.retrieve_catalogue(cat_name):
            return retrieve.load_catalogue(cat_name=cat_name, cat=self.get_path(f"cat_csv_{cat_name}"))
        else:
            print("Could not load catalogue; field is outside footprint.")

    def generate_astrometry_indices(
            self,
            cat_name: str = "gaia",
            correct_to_epoch: bool = True
    ):
        self.retrieve_catalogue(cat_name=cat_name)
        if not self.check_cat(cat_name=cat_name):
            print(f"Field is not in {cat_name}; index file could not be created.")
        else:
            cat_path = self.get_path(f"cat_csv_{cat_name}")
            index_path = os.path.join(config["top_data_dir"], "astrometry_index_files")
            u.mkdir_check(index_path)
            cat_index_path = os.path.join(index_path, cat_name)
            prefix = f"{cat_name}_index_{self.name}"
            astm.generate_astrometry_indices(
                cat_name=cat_name,
                cat=cat_path,
                fits_cat_output=cat_path.replace(".csv", ".fits"),
                output_file_prefix=prefix,
                index_output_dir=cat_index_path,
                unique_id_prefix=int(self.name.replace("FRB", ""))
            )

    def get_path(self, key):
        if key in self.paths:
            return self.paths[key]
        else:
            raise KeyError(f"{key} has not been set.")

    def check_cat(self, cat_name: str):
        if f"in_{cat_name}" in self.cats:
            return self.cats[f"in_{cat_name}"]
        else:
            return None

    def set_path(self, key, value):
        self.paths[key] = value

    def load_output_file(self, **kwargs):
        outputs = p.load_output_file(self)
        if outputs is not None:
            if "cats" in outputs:
                self.cats.update(outputs["cats"])
        return outputs

    def _output_dict(self):
        return {
            "paths": self.paths,
            "cats": self.cats,
        }

    def update_output_file(self):
        p.update_output_file(self)

    def add_object(self, obj: objects.Object):
        self.objects.append(obj)
        self.objects_dict[obj.name] = obj
        obj.field = self

    def add_object_from_dict(self, obj_dict: dict):
        obj = objects.Object.from_dict(obj_dict, field=self)
        self.add_object(obj=obj)

    def object_properties(self):
        self.objects.sort(key=lambda o: o.name, reverse=True)
        for obj in self.objects:
            obj.load_output_file()
            obj.update_output_file()
            obj.push_to_table(select=True)
            obj.write_plot_photometry()
            obj.update_output_file()
        self.generate_cigale_photometry()

    def generate_cigale_photometry(self):
        # photometries = {
        #     "Galaxy ID": [],
        #     "z": []
        # }
        photometries = []
        for obj in self.objects:
            if isinstance(obj, objects.Galaxy):
                photometry = {}
                obj.load_output_file()
                obj.photometry_to_table()
                tbl_this = obj.photometry_to_table(best=True)
                photometry["Galaxy ID"] = obj.name
                photometry["z"] = obj.z
                for row in tbl_this:
                    instrument_name = row["instrument"]
                    instrument = inst.Instrument.from_params(instrument_name)
                    if instrument.cigale_name is not None:
                        inst_cig = instrument.cigale_name
                    else:
                        inst_cig = instrument_name

                    if instrument_name == "vlt-fors2":
                        fil_cig = row["band"][0].lower()
                    else:
                        fil_cig = row["band"]
                    band_str = f"{inst_cig}_{fil_cig}"
                    if np.isnan(row["mag_sep_ext_corrected"]):
                        photometry[band_str] = -999.
                        photometry[band_str + "_err"] = -999.
                    else:
                        photometry[band_str] = row["mag_sep_ext_corrected"]
                        photometry[band_str + "_err"] = row["mag_sep_err"]
                for entry in photometries:
                    for name in entry:
                        if name not in photometry:
                            photometry[name] = -999.
                    for name in photometry:
                        if name not in entry:
                            entry[name] = -999.

                photometries.append(photometry)
        tbl_cigale = table.QTable(photometries)
        tbl_cigale.write(
            os.path.join(self.data_path, f"{self.name}_cigale.csv"),
            overwrite=True
        )
        return tbl_cigale

    def unpack_cigale_results(
            self,
            cigale_dir: str
    ):
        results = fits.open(os.path.join(cigale_dir, "results.fits"))
        results_tbl = table.QTable(results[1].data)

        for i, obj_name in enumerate(results_tbl["id"]):
            if obj_name in self.objects_dict:
                model_path = os.path.join(cigale_dir, f"{obj_name}_best_model.fits")
                sfh_path = os.path.join(cigale_dir, f"{obj_name}_SFH.fits")
                obj = self.objects_dict[obj_name]
                obj.load_output_file()
                obj.cigale_results = p.sanitise_yaml_dict(dict(results_tbl[i]))
                obj.mass_stellar = obj.cigale_results["bayes.stellar.m_star"] * units.solMass
                obj.mass_stellar_err = obj.cigale_results["bayes.stellar.m_star_err"] * units.solMass
                obj.sfr = obj.cigale_results["bayes.sfh.sfr"] * units.solMass / units.year
                obj.sfr_err = obj.cigale_results["bayes.sfh.sfr_err"] * units.solMass / units.year
                if os.path.isfile(model_path):
                    obj.cigale_model_path = model_path
                if os.path.isfile(sfh_path):
                    obj.cigale_sfh_path = sfh_path
                obj.update_output_file()

    def load_all_objects(self):
        for obj in self.objects:
            obj.load_output_file()

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "type": "Field",
            "centre": objects.position_dictionary.copy(),
            "objects": [objects.Object.default_params()],
            "extent": 0.1 * units.deg,
            "survey": None
        }
        return default_params

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):
        name, param_file, param_dict = p.params_init(param_file)
        if param_file is None:
            return None
        # Check data_dir path for relevant .yamls (output_values, etc.)

        if param_dict is None:
            raise FileNotFoundError(f"There is no param file at {param_file}")
        field_type = param_dict["type"]
        centre_ra, centre_dec = p.select_coords(param_dict["centre"])
        coord_str = f"{centre_ra} {centre_dec}"

        if field_type == "Field":
            return cls(
                centre_coords=coord_str,
                objs=param_dict["objects"],
                **param_dict
            )
        elif field_type == "FRBField":
            return FRBField(
                centre_coords=coord_str,
                objs=param_dict["objects"],
                **param_dict
            )
        elif field_type == "StandardField":
            return StandardField(
                centre_coords=coord_str,
                **param_dict
            )

    @classmethod
    def from_params(cls, name):
        if name in active_fields:
            return active_fields[name]
        print("Initializing field...")
        path = cls.build_param_path(field_name=name)
        return cls.from_file(param_file=path)

    @classmethod
    def new_yaml(cls, name: str, path: str = None, **kwargs):
        param_dict = cls.default_params()
        param_dict["name"] = name
        param_dict["data_path"] = os.path.join(name, "")
        for kwarg in kwargs:
            param_dict[kwarg] = kwargs[kwarg]
        if path is not None:
            if os.path.isdir(path):
                path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict)
        return param_dict

    @classmethod
    def build_param_path(cls, field_name: str):
        path = u.mkdir_check_args(p.param_dir, "fields", field_name)
        return os.path.join(path, f"{field_name}.yaml")

    @classmethod
    def new_params_from_input(cls, field_name: str, field_param_path: str):
        _, field_class = u.select_option(
            message="Which type of field would you like to create?",
            options={
                "FRB field": FRBField,
                "Standard (calibration) field": StandardField,
                "Normal field": Field
            })

        survey_options = survey.Survey.list_surveys()
        survey_options.append("New survey")
        survey_options.append("None")
        _, survey_name = u.select_option(
            message="Which survey is this field a part of?",
            options=survey_options
        )
        if survey_name == "New survey":
            survey_name = survey.Survey.new_param_from_input()
        elif survey_name == "None":
            survey_name = None

        pos_coord = None
        while pos_coord is None:
            ra = u.user_input(
                "Please enter the Right Ascension of the field target, in the format 00h00m00.0s or as a decimal number of degrees"
                " (for an FRB field, this should be the FRB coordinates). Eg: 13h19m14.08s, 199.80867d")
            ra_err = 0.0
            if field_class is FRBField:
                ra_err = u.user_input("If you know the uncertainty in the FRB localisation RA, you can enter "
                                      "that now (in true arcseconds, not in RA units). Otherwise, leave blank.")
                if ra_err in ["", " ", 'None']:
                    ra_err = 0.0
            dec = u.user_input(
                "Please enter the Declination of the field target, in the format 00d00m00.0s or as a decimal number of degrees"
                " (for an FRB field, this should be the FRB coordinates). Eg: -18d50m16.7s, -18.83797222d")
            dec_err = 0.0
            if field_class is FRBField:
                dec_err = u.user_input(
                    "If you know the uncertainty in the FRB localisation Dec, you can enter "
                    "that now, in arcseconds. Otherwise, leave blank.")
                if dec_err in ["", " ", 'None']:
                    dec_err = 0.0
            try:
                pos_coord = astm.attempt_skycoord((ra, dec))
            except ValueError:
                print("Invalid values encountered when parsing coordinates. Please try again.")

        position = objects.skycoord_to_position_dict(skycoord=pos_coord)

        field_param_path_yaml = os.path.join(field_param_path, f"{field_name}.yaml")

        yaml_dict = field_class.new_yaml(
            path=field_param_path,
            name=field_name,
            centre=position,
            survey=survey_name
        )
        if field_class is FRBField:
            if ra_err is None:
                ra_err = 0.
            if dec_err is None:
                dec_err = 0.
            date = u.user_input(
                "If you have a precise FRB arrival time, please enter that now; otherwise, leave blank."
            )
            if date in ["", " ", 'None']:
                date = objects.FRB._date_from_name(field_name)
            yaml_dict["frb"]["date"] = date
            yaml_dict["frb"]["position"] = position
            yaml_dict["frb"]["position_err"]["a"]["stat"] = float(ra_err)
            yaml_dict["frb"]["position_err"]["b"]["stat"] = float(dec_err)
            yaml_dict["frb"]["host_galaxy"]["position"] = position

            p.save_params(field_param_path_yaml, yaml_dict)

        print(f"Template parameter file created at '{field_param_path_yaml}'")
        input("Please edit this file before proceeding, then press Enter to continue.")


class StandardField(Field):
    def __init__(
            self,
            centre_coords: Union[SkyCoord, str] = None,
            **kwargs
    ):
        jname = astm.jname(
            coord=centre_coords,
            ra_precision=0,
            dec_precision=0
        )
        name = f"STD-{jname}"

        param_path = os.path.join(p.param_dir, "fields", name, f"{name}.yaml")
        if not os.path.isfile(param_path):
            u.mkdir_check_nested(param_path)
            self.new_yaml(
                name=name,
                path=param_path,
                centre=objects.skycoord_to_position_dict(centre_coords)
            )

        super().__init__(
            name=name,
            centre_coords=centre_coords,
            data_path=os.path.join(p.data_path, name),
            param_path=param_path
        )

        self.retrieve_catalogues()


class FRBField(Field):
    def __init__(
            self,
            name: str = None,
            centre_coords: Union[SkyCoord, str] = None,
            param_path: str = None,
            data_path: str = None,
            objs: List[objects.Object] = None,
            frb: Union[objects.FRB, dict] = None,
            extent: units.Quantity = None,
            **kwargs
    ):
        if centre_coords is None:
            if frb is not None:
                centre_coords = frb.position

        # Input attributes
        super().__init__(name=name,
                         centre_coords=centre_coords,
                         param_path=param_path,
                         data_path=data_path,
                         objs=objs,
                         extent=extent,
                         **kwargs
                         )

        self.frb = frb
        if self.frb is not None:

            if isinstance(self.frb, dict):
                self.frb = objects.FRB.from_dict(self.frb, field=self)

            self.frb.field = self
            if self.frb.host_galaxy is not None:
                self.add_object(self.frb.host_galaxy)
        self.epochs_imaging_old = {}

    def plot_host_colour(
            self,
            output_path: str,
            red: image.ImagingImage,
            blue: image.ImagingImage,
            green: image.ImagingImage = None,
            fig: plt.Figure = None,
            centre: SkyCoord = None,
            show_frb: bool = True,
            show_coords: bool = True,
            frame: units.Quantity = 30 * units.pix,
            n: int = 1, n_x: int = 1, n_y: int = 1,
            frb_kwargs: dict = {},
            imshow_kwargs: dict = {},
            ext: Union[tuple, int] = (0, 0, 0),
            vmaxes: tuple = (None, None, None),
            vmins: tuple = (None, None, None),
            scale_to_jansky: bool = False,
            scale_to_rgb: bool = False,
            **kwargs
    ):
        pl.latex_setup()

        if not isinstance(self.frb, objects.FRB):
            raise TypeError("self.frb has not been set properly for this FRBField.")
        if centre is None:
            centre = self.frb.host_galaxy.position
        if fig is None:
            fig = plt.figure()
        if isinstance(ext, int):
            ext = (ext, ext, ext)

        path_split = os.path.split(output_path)[-1]

        frame = u.check_quantity(frame, unit=units.pix)

        red_data, red_trimmed = red.prep_for_colour(
            output_path=output_path.replace(path_split, f"{red.name}_trimmed.fits"),
            frame=frame,
            centre=centre,
            vmax=vmaxes[0],
            vmin=vmins[0],
            ext=ext[0],
            scale_to_jansky=scale_to_jansky
        )

        blue_data, _ = blue.prep_for_colour(
            output_path=output_path.replace(path_split, f"{blue.name}_trimmed.fits"),
            frame=frame,
            centre=centre,
            vmax=vmaxes[1],
            vmin=vmins[1],
            ext=ext[1],
            scale_to_jansky=scale_to_jansky
        )

        if green is None:
            green_data = (red_data + blue_data) / 2
        else:
            green_data, _ = green.prep_for_colour(
                output_path=output_path.replace(path_split, f"{green.name}_trimmed.fits"),
                frame=frame,
                centre=centre,
                vmax=vmaxes[2],
                vmin=vmins[2],
                ext=ext[2],
                scale_to_jansky=scale_to_jansky
            )

        if scale_to_rgb:
            max_all = max(np.max(red_data), np.max(green_data), np.max(blue_data))
            factor = max_all / 255
            red_data /= factor
            green_data /= factor
            blue_data /= factor

        colour = make_lupton_rgb(
            red_data,
            green_data,
            blue_data,
            Q=7,
            stretch=30
        )

        if "origin" not in imshow_kwargs:
            imshow_kwargs["origin"] = "lower"

        if show_coords:
            projection = red_trimmed.wcs
        else:
            projection = None
        ax = fig.add_subplot(n_x, n_y, n, projection=projection)

        if not show_coords:
            ax.get_xaxis().set_visible(False)
            ax.set_yticks([])
            ax.invert_yaxis()

        ax.imshow(
            colour,
            **imshow_kwargs,
        )
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        # ax.set_xlabel("Right Ascension (J2000)", size=16)
        # ax.set_ylabel("Declination (J2000)", size=16, rotation=0, labelpad=-20)
        ax.tick_params(labelsize=10)
        # ax.yaxis.set_label_position("right")
        # plt.tight_layout()

        if show_frb:
            self.frb_ellipse_to_plot(ext=ext[0], frb_kwargs=frb_kwargs, img=red_trimmed, plot=ax)

        fig.savefig(output_path)
        return ax, fig, colour

    def plot_host(
            self,
            img: image.ImagingImage,
            ext: int = 0,
            fig: plt.Figure = None,
            ax: plt.Axes = None,
            centre: SkyCoord = None,
            show_frb: bool = True,
            frame: units.Quantity = 30 * units.pix,
            n: int = 1, n_x: int = 1, n_y: int = 1,
            # show_cbar: bool = False,
            # show_grid: bool = False,
            # ticks: int = None, interval: str = 'minmax',
            # font_size: int = 12,
            # reverse_y=False,
            frb_kwargs: dict = {},
            imshow_kwargs: dict = {},
            normalize_kwargs: dict = {},
            output_path: str = None,
            show_legend: bool = False,
            **kwargs
    ):
        pl.latex_setup()
        if not isinstance(self.frb, objects.FRB):
            raise TypeError("self.frb has not been set properly for this FRBField.")
        if centre is None:
            centre = self.frb.host_galaxy.position

        plot, fig, other_args = img.plot_subimage(
            centre=centre,
            frame=frame,
            ext=ext,
            fig=fig,
            ax=ax,
            n=n, n_x=n_x, n_y=n_y,
            imshow_kwargs=imshow_kwargs,
            normalize_kwargs=normalize_kwargs,
            **kwargs
        )

        if show_frb:
            self.frb_ellipse_to_plot(ext=ext, frb_kwargs=frb_kwargs, img=img, plot=plot)
            if show_legend:
                plot.legend()

        if output_path is not None:
            fig.savefig(output_path)

        return plot, fig

    def frb_ellipse_to_plot(
            self,
            plot,
            img: image.ImagingImage,
            ext: int = 0,
            colour: str = "white",
            frb_kwargs: dict = {},
            plot_centre: bool = False
    ):
        from matplotlib.patches import Ellipse
        img.load_headers()
        frb = self.frb.position
        x, y = img.world_to_pixel(frb, 0)
        uncertainty = self.frb.position_err
        a, b = uncertainty.uncertainty_quadrature()
        if a == 0 * units.arcsec or b == 0 * units.arcsec:
            a, b = uncertainty.uncertainty_quadrature_equ()
        theta = uncertainty.theta.to(units.deg)
        rotation_angle = img.extract_rotation_angle(ext=ext)
        theta = theta - rotation_angle
        img_err = img.extract_astrometry_err()
        img.extract_pixel_scale()
        if "edgecolor" not in frb_kwargs:
            frb_kwargs["edgecolor"] = colour
        if "facecolor" not in frb_kwargs:
            frb_kwargs["facecolor"] = "none"
        if img_err is not None:
            a = np.sqrt(a ** 2 + img_err ** 2)
            b = np.sqrt(b ** 2 + img_err ** 2)
        e = Ellipse(
            xy=(x, y),
            width=2 * a.to(units.pix, img.pixel_scale_y).value,
            height=2 * b.to(units.pix, img.pixel_scale_y).value,
            angle=theta.value,
            **frb_kwargs
        )
        # e.set_edgecolor(color)
        plot.add_artist(e)
        if plot_centre:
            plot.scatter(x, y, c=frb_kwargs["edgecolor"], marker="x")

    @classmethod
    def default_params(cls):
        default_params = super().default_params()

        default_params.update({
            "type": "FRBField",
            "frb": objects.FRB.default_params(),
            "subtraction":
                {
                    "template_epochs":
                        {
                            "des": None,
                            "fors2": None,
                            "xshooter": None,
                            "sdss": None
                        }
                },
        })

        return default_params

    @classmethod
    def new_yaml(cls, name: str, path: str = None, **kwargs) -> dict:
        """
        Generates a new parameter .yaml file for an FRBField.
        :param name: Name of the field.
        :param path: Path to write .yaml to.
        :param kwargs: Other keywords to insert or replace in the output yaml.
        :return: dict reflecting content of yaml file.
        """
        param_dict = super().new_yaml(name=name, path=None)
        param_dict["frb"]["name"] = name
        param_dict["frb"]["type"] = "FRB"
        if "FRB" in name:
            param_dict["frb"]["host_galaxy"]["name"] = name.replace("FRB", "HG")
        else:
            param_dict["frb"]["host_galaxy"]["name"] = name + " Host"

        for kwarg in kwargs:
            param_dict[kwarg] = kwargs[kwarg]
        if path is not None:
            path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict)
        u.debug_print(2, "FRBField.new_yaml(): param_dict:", param_dict)
        return param_dict

    @classmethod
    def yaml_from_furby_dict(
            cls,
            furby_dict: dict,
            output_path: str,
            healpix_path: str = None) -> dict:
        """
        Constructs a param .yaml file from a dict representing a FURBY json file.
        :param furby_dict: the .json file read in as a dict
        :param output_path: The path to write output yaml file to.
        :param healpix_path: Optional, path to FITS file containing healpix information.
        :return: Dictionary containing the same information as the written .yaml
        """

        u.mkdir_check(output_path)

        field_name = furby_dict["Name"]
        frb = objects.FRB.default_params()
        coords = objects.position_dictionary.copy()

        ra = furby_dict["RA"]
        dec = furby_dict["DEC"]

        pos_coord = astm.attempt_skycoord((ra * units.deg, dec * units.deg))
        ra_str, dec_str = astm.coord_string(pos_coord)

        coords["ra"]["decimal"] = ra
        coords["dec"]["decimal"] = dec
        coords["ra"]["hms"] = ra_str
        coords["dec"]["dms"] = dec_str

        obs.load_furby_table()
        row, _ = obs.get_row_furby(field_name)
        if row is not None:
            frb["position_err"]["a"]["stat"] = row["sig_ra"]
            frb["position_err"]["b"]["stat"] = row["sig_dec"]

        frb["dm"] = furby_dict["DM"] * objects.dm_units
        frb["name"] = field_name
        frb["position"] = coords.copy()
        frb["position_err"]["healpix_path"] = healpix_path
        frb["host_galaxy"]["name"] = "HG" + field_name[3:]
        param_dict = cls.new_yaml(
            name=field_name,
            path=output_path,
            centre=coords,
            frb=frb,
            snr=furby_dict["S/N"],
            survey="furby"
        )

        return param_dict

    @classmethod
    def param_from_furby_json(cls, json_path: str, healpix_path: str = None):
        """
        Constructs a param .yaml file from a FURBY json file and places it in the default location.
        :param json_path: The path to the FURBY .json file.
        :param healpix_path: Optional, path to FITS file containing healpix information.
        :return:
        """
        furby_dict = p.load_json(json_path)
        u.debug_print(2, "FRBField.param_from_furby_json(): json_path ==", json_path)
        u.debug_print(2, "FRBField.param_from_furby_json(): furby_dict ==", furby_dict)
        field_name = furby_dict["Name"]
        output_path = os.path.join(p.param_dir, "fields", field_name)

        param_dict = cls.yaml_from_furby_dict(
            furby_dict=furby_dict,
            healpix_path=healpix_path,
            output_path=output_path
        )
        return param_dict

    @classmethod
    def convert_old_param(cls, frb: str):

        new_frb = f"FRB20{frb[3:]}"
        new_params = cls.new_yaml(name=new_frb, path=None)
        old_params = p.object_params_frb(frb)

        new_params["name"] = new_frb

        new_params["centre"]["dec"]["decimal"] = old_params["burst_dec"]
        new_params["centre"]["dec"]["dms"] = old_params["burst_dec_str"]

        new_params["centre"]["ra"]["decimal"] = old_params["burst_ra"]
        new_params["centre"]["ra"]["hms"] = old_params["burst_ra_str"]

        old_data_dir = old_params["data_dir"]
        if isinstance(old_data_dir, str):
            new_params["data_path"] = old_data_dir.replace(frb, new_frb)

        new_params["frb"]["name"] = new_frb

        new_params["frb"]["host_galaxy"]["position"]["dec"]["decimal"] = old_params["hg_dec"]
        new_params["frb"]["host_galaxy"]["position"]["ra"]["decimal"] = old_params["hg_ra"]

        new_params["frb"]["host_galaxy"]["position_err"]["dec"]["stat"] = old_params["hg_err_y"]
        new_params["frb"]["host_galaxy"]["position_err"]["ra"]["stat"] = old_params["hg_err_x"]

        new_params["frb"]["host_galaxy"]["z"] = old_params["z"]

        new_params["frb"]["mjd"] = old_params["mjd_burst"]

        new_params["frb"]["position"]["dec"]["decimal"] = old_params["burst_dec"]
        new_params["frb"]["position"]["dec"]["dms"] = old_params["burst_dec_str"]

        new_params["frb"]["position"]["ra"]["decimal"] = old_params["burst_ra"]
        new_params["frb"]["position"]["ra"]["hms"] = old_params["burst_ra_str"]

        new_params["frb"]["position_err"]["a"]["stat"] = old_params["burst_err_stat_a"]
        new_params["frb"]["position_err"]["a"]["sys"] = old_params["burst_err_sys_a"]

        new_params["frb"]["position_err"]["b"]["stat"] = old_params["burst_err_stat_b"]
        new_params["frb"]["position_err"]["b"]["sys"] = old_params["burst_err_sys_b"]

        new_params["frb"]["position_err"]["dec"]["stat"] = old_params["burst_err_stat_dec"]
        new_params["frb"]["position_err"]["dec"]["sys"] = old_params["burst_err_sys_dec"]

        new_params["frb"]["position_err"]["ra"]["stat"] = old_params["burst_err_stat_ra"]
        new_params["frb"]["position_err"]["ra"]["sys"] = old_params["burst_err_sys_ra"]

        new_params["frb"]["position_err"]["theta"] = old_params["burst_err_theta"]

        if "other_objects" in old_params and type(old_params["other_objects"]) is dict:
            for obj in old_params["other_objects"]:
                if obj != "<name>":
                    obj_dict = objects.Object.default_params()
                    obj_dict["name"] = obj
                    obj_dict["position"] = objects.position_dictionary.copy()
                    obj_dict["position"]["dec"]["decimal"] = old_params["other_objects"][obj]["dec"]
                    obj_dict["position"]["ra"]["decimal"] = old_params["other_objects"][obj]["ra"]
                    new_params["objects"].append(obj_dict)

        new_params["subtraction"]["template_epochs"]["des"] = old_params["template_epoch_des"]
        new_params["subtraction"]["template_epochs"]["fors2"] = old_params["template_epoch_fors2"]
        new_params["subtraction"]["template_epochs"]["sdss"] = old_params["template_epoch_sdss"]
        new_params["subtraction"]["template_epochs"]["xshooter"] = old_params["template_epoch_xshooter"]

        param_path_upper = os.path.join(p.param_dir, "fields", new_frb)
        u.mkdir_check(param_path_upper)
        p.save_params(file=os.path.join(param_path_upper, f"{new_frb}.yaml"), dictionary=new_params)

    def gather_epochs_old(self):
        print("Searching for old-format imaging epoch param files...")
        epochs = {}
        param_dir = p.param_dir
        for instrument_path in filter(lambda d: d.startswith("epochs_"), os.listdir(param_dir)):
            instrument = instrument_path.split("_")[-1]
            instrument_path = os.path.join(param_dir, instrument_path)
            gathered = list(filter(
                lambda f: (f.startswith(self.name) or f.startswith(f"FRB{self.name[5:]}")) and f.endswith(".yaml"),
                os.listdir(instrument_path)))
            gathered.sort()
            for epoch_param in gathered:
                epoch_name = epoch_param[:epoch_param.find('.yaml')]
                if f"{self.name}_{epoch_name[-1]}" not in self.epochs_imaging:
                    param_path = os.path.join(instrument_path, epoch_param)
                    epoch = p.load_params(file=param_path)
                    epoch["format"] = "old"
                    epoch["name"] = epoch_name
                    epoch["instrument"] = instrument
                    epoch["param_path"] = param_path
                    epochs[epoch_name] = epoch
        self.epochs_imaging.update(epochs)
        return epochs


epoch_stage_dirs = {"0-download": "0-data_with_raw_calibs",
                    "2-pypeit": "2-pypeit",
                    }


class Epoch:
    instrument_name = "dummy-instrument"
    mode = "dummy_mode"
    frame_class = image.Image

    def __init__(
            self,
            param_path: str = None,
            name: str = None,
            field: Union[str, Field] = None,
            data_path: str = None,
            instrument: str = None,
            date: Union[str, Time] = None,
            program_id: str = None,
            target: str = None,
            do_stages: Union[list, str] = None,
            **kwargs
    ):

        # Input attributes
        self.param_path = param_path
        self.name = name
        self.field = field
        self.data_path = None
        self.data_path_relative = None
        if data_path is not None:
            self.data_path = os.path.join(p.data_path, data_path)
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

        add_to_epoch_directory(
            field_name=self.field.name,
            instrument=self.instrument_name,
            mode=self.mode,
            epoch_name=self.name)

        self.param_file = kwargs

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

            # Check if we should do this stage
            if do_this and (no_query or self.query_stage(
                    message=message,
                    n=n,
                    stage_name=name
            )):
                print(f"Performing processing step {n}: {name}")
                # Construct path; if dir_name is None then the step is pathless.
                dir_name = f"{n}-{name}"
                output_dir = os.path.join(self.data_path, dir_name)
                output_dir_backup = output_dir + "_backup"
                u.rmtree_check(output_dir_backup)
                u.move_check(output_dir, output_dir_backup)
                u.mkdir_check_nested(output_dir, remove_last=False)
                self.set_path(name, output_dir)

                if name in self.param_file:
                    stage_kwargs = self.param_file[name]
                else:
                    stage_kwargs = {}

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
            "log": self.log.to_dict()
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

    def query_stage(self, message: str, stage_name: str, n: float):
        """
        Helper method for asking the user if we need to do this stage of processing.
        If self.do is True, skips the query and returns True.
        :param message: Message to display.
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
            message = f"{n}. {message}"
            done = self.check_done(stage=stage_name)
            u.debug_print(2, "Epoch.query_stage(): done ==", done)
            if done is not None:
                time_since = (Time.now() - done).sec * units.second
                time_since = u.relevant_timescale(time_since)
                message += f" (last performed at {done.isot}, {time_since.round(1)} ago)"
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

    def get_path(self, key):
        if key in self.paths:
            return self.paths[key]
        else:
            raise KeyError(f"{key} has not been set.")

    def set_path(self, key, value):
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
                cls = image.CoaddedImage.select_child_class(instrument=self.instrument_name)
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
            "notes": []
        }
        # Pull the list of applicable kwargs from the stage information
        stages = cls.stages()
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
    def _from_params_setup(cls, name: str, field: Union[Field, str] = None):
        field_name = None
        if isinstance(field, Field):
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
        field = StandardField(centre_coords=centre_coords)
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

    def __init__(
            self,
            name: str = None,
            field: Union[str, Field] = None,
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
        self.coadded_final = None

        self.gaia_catalogue = None

        self.frame_stats = {}
        self.astrometry_stats = {}
        self.psf_stats = {}

        # self.load_output_file(mode="imaging")

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
                    "skip_indices": False
                }
            },
            "frame_diagnostics": {
                "method": cls.proc_frame_diagnostics,
                "message": "Run diagnostics on individual frames?",
                "default": False,
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
                    "astroalign_template": None
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
                    "image_type": "coadded_trimmed",
                    "preferred_zeropoint": {},
                    "suppress_select": False
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

        skip = False
        if "skip_indices" in kwargs:
            skip = kwargs.pop("skip_indices")

        if "correct_to_epoch" in kwargs:
            print(f"correct_to_epoch 1: {kwargs['correct_to_epoch']}")
            correct_to_epoch = kwargs.pop("correct_to_epoch")
            print(f"correct_to_epoch 2: {correct_to_epoch}")
        else:
            correct_to_epoch = True

        u.debug_print(2, kwargs)
        u.debug_print(1, "skip ==", skip)

        if not skip:
            self.generate_astrometry_indices(correct_to_epoch=correct_to_epoch)

        self.frames_astrometry = {}

        if "register_frames" in self.do_kwargs and self.do_kwargs["register_frames"]:
            self.correct_astrometry_frames(
                output_dir=output_dir,
                frames=self.frames_registered,
                **kwargs)
        else:
            self.correct_astrometry_frames(
                output_dir=output_dir,
                frames=self.frames_normalised,
                **kwargs)

    def correct_astrometry_frames(
            self,
            output_dir: str,
            frames: dict = None,
            am_params: dict = {},
            **kwargs
    ):
        self.frames_astrometry = {}

        if frames is None:
            frames = self.frames_reduced

        for fil in frames:
            frames_by_chip = self.sort_by_chip(frames[fil])
            for chip in frames_by_chip:
                print()
                print(f"Processing frames for chip {chip} in astrometry.net:")
                print()
                first_success = None
                astrometry_fil_path = os.path.join(output_dir, fil)
                for frame in frames_by_chip[chip]:
                    new_frame = frame.correct_astrometry(
                        output_dir=astrometry_fil_path,
                        am_params=am_params,
                        **kwargs
                    )

                    if new_frame is not None:
                        print(f"{frame} astrometry successful.")
                        self.add_frame_astrometry(new_frame)
                        self.astrometry_successful[fil][frame.name] = "astrometry.net"
                        if first_success is None:
                            first_success = new_frame
                    else:
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

                print()
                print(f"Re-processing failed frames for chip {chip} with astroalign, with template {first_success}:")
                print()
                for frame in frames_by_chip[chip]:
                    if not self.astrometry_successful[fil][frame.name]:
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
                        print(f"\tAdding {frame_lists[chip][i]}")
                        self.add_frame_diagnosed(frame_lists[chip][i])
                else:
                    print(f"Median PSF FWHM {fwhm_median} > upper limit {upper_limit}")

    def proc_coadd(self, output_dir: str, **kwargs):
        if "frames" not in kwargs:
            kwargs["frames"] = self.frames_final
        self.coadd(output_dir, **kwargs)

    def coadd(self, output_dir: str, frames: str = "astrometry", sigma_clip: float = 1.5):
        """
        Use Montage to coadd individual frames.
        :param output_dir: Directory in which to write data products.
        :param frames: Name of frames list to coadd.
        :return:
        """
        import ccdproc
        u.mkdir_check(output_dir)
        frame_dict = self._get_frames(frame_type=frames)
        input_frames = self._get_frames(frames)

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
                coadd_types=["median"],
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
                ancestors=input_frames[fil]
            )
            ccds = []
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
                ancestors=input_frames[fil]
            )
            combined_img.write_fits_file()
            combined_img.update_output_file()

            self.add_coadded_image(sigclip_path, key=fil, mode="imaging")

    def proc_correct_astrometry_coadded(self, output_dir: str, **kwargs):
        self.generate_astrometry_indices()
        self.correct_astrometry_coadded(
            output_dir=output_dir,
            images=self.coadded,
            **kwargs
        )

    def correct_astrometry_coadded(self, output_dir: str, images: dict, **kwargs):
        self.coadded_astrometry = {}

        if images is None:
            images = self.coadded

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

    def trim_coadded(self, output_dir: str, images: dict = None, reproject: bool = False):
        if images is None:
            images = self.coadded
        u.mkdir_check(output_dir)
        template = None
        for fil in images:
            img = images[fil]
            output_path = os.path.join(output_dir, img.filename.replace(".fits", "_trimmed.fits"))
            u.debug_print(2, "trim_coadded img.path:", img.path)
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
        print("Extracting sources for", image_type, "with", len(list(images.keys())))
        for fil in images:
            img = images[fil]
            print(f"Extracting sources from {img}")
            configs = self.source_extractor_config

            img.psfex_path = None
            img.source_extraction_psf(
                output_dir=output_dir,
                phot_autoparams=f"{configs['kron_factor']},{configs['kron_radius_min']}"
            )
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

        if "distance_tolerance" in kwargs and kwargs["distance_tolerance"] is not None:
            kwargs["distance_tolerance"] = u.check_quantity(kwargs["distance_tolerance"], units.arcsec, convert=True)
        if "snr_min" not in kwargs or kwargs["snr_min"] is None:
            kwargs["snr_min"] = 3.
        if "suppress_select" not in kwargs:
            kwargs["suppress_select"] = False

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

        self.deepest_filter = deepest.filter_name
        self.deepest = deepest

        print("DEEPEST FILTER:", self.deepest_filter, self.deepest.depth["secure"]["SNR_PSF"]["5-sigma"])

    def zeropoint(
            self,
            image_dict: dict,
            output_path: str,
            distance_tolerance: units.Quantity = None,
            snr_min: float = 3.,
            star_class_tolerance: int = 0.95,
            suppress_select: bool = False,
            **kwargs
    ):

        deepest = image_dict[self.filters[0]]
        for fil in self.filters:
            img = image_dict[fil]
            for cat_name in retrieve.photometry_catalogues:
                if cat_name == "gaia":
                    continue
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
        self.get_photometry(output_dir, image_type=image_type)

    def get_photometry(
            self,
            path: str,
            image_type: str = "final",
            dual: bool = False,
            match_tolerance: units.Quantity = 3 * units.arcsec
    ):
        """
        Retrieve photometric properties of key objects and write to disk.
        :param path: Path to which to write the data products.
        :return:
        """

        print(f"Getting finalised photometry for key objects, in {image_type}.")
        obs.load_master_objects_table()

        image_dict = self._get_images(image_type=image_type)
        u.mkdir_check(path)
        # Loop through filters
        for fil in image_dict:
            fil_output_path = os.path.join(path, fil)
            u.mkdir_check(fil_output_path)
            img = image_dict[fil]
            if "secure" not in img.depth:
                img.estimate_depth()
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
                        good_image_path=self.coadded_unprojected[fil].path,
                        do_mask=img.mask_nearby()
                    )
                else:
                    u.debug_print(2, "ImagingImage.get_photometry(): nearest.colnames ==", nearest.colnames)
                    err = nearest[f'MAGERR_AUTO_ZP_best']
                    print("FILTER:", fil)
                    print(f"MAG_AUTO = {nearest['MAG_AUTO_ZP_best']} +/- {err}")
                    print(f"A = {nearest['A_WORLD'].to(units.arcsec)}; B = {nearest['B_WORLD'].to(units.arcsec)}")
                    img.plot_source_extractor_object(
                        nearest,
                        output=os.path.join(fil_output_path, f"{obj.name_filesys}.png"),
                        show=False,
                        title=f"{obj.name}, {fil}-band, {nearest['MAG_AUTO_ZP_best'].round(3).value} ± {err.round(3)}"
                    )
                    obj.cat_row = nearest
                    print()

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
                        good_image_path=self.coadded_unprojected[fil].path,
                        do_mask=img.mask_nearby()
                    )

                    if isinstance(self.field, FRBField):
                        if "frame" in obj.plotting_params and obj.plotting_params["frame"] is not None:
                            frame = obj.plotting_params["frame"]
                        else:
                            frame = img.nice_frame(row=obj.cat_row)

                        normalize_kwargs = None
                        if fil in obj.plotting_params:
                            if "normalize" in obj.plotting_params[fil]:
                                normalize_kwargs = obj.plotting_params[fil]["normalize"]

                        centre = obj.position_from_cat_row()
                        fig = plt.figure(figsize=(6, 5))
                        plot, fig = self.field.plot_host(
                            img=img,
                            fig=fig,
                            centre=centre,
                            show_frb=True,
                            frame=frame,
                            imshow_kwargs={
                                "cmap": "plasma"
                            },
                            normalize_kwargs=normalize_kwargs
                        )
                        output_path = os.path.join(fil_output_path, f"{obj.name_filesys}_{fil}.pdf")
                        name = obj.name
                        img.extract_filter()
                        if img.filter is None:
                            f_name = fil
                        else:
                            f_name = img.filter.nice_name()
                        plot.set_title(u.latex_sanitise(f"{name}, {f_name}"))
                        fig.savefig(output_path)
                        fig.savefig(output_path.replace(".pdf", ".png"))
                        plt.close(fig)
                        pl.latex_off()

            tbl = table.vstack(rows)
            tbl.add_column(names, name="NAME")
            tbl.add_column(separations, name="OFFSET_FROM_TARGET")
            tbl.add_column(ra_target, name="RA_TARGET")
            tbl.add_column(dec_target, name="DEC_TARGET")

            tbl.write(
                os.path.join(fil_output_path, f"{self.field.name}_{self.name}_{fil}.ecsv"),
                format="ascii.ecsv")
            tbl.write(
                os.path.join(fil_output_path, f"{self.field.name}_{self.name}_{fil}.csv"),
                format="ascii.csv")

        for fil in self.coadded_unprojected:

            img = self.coadded_unprojected[fil]
            img_projected = image_dict[fil]

            if img is None:
                continue

            if isinstance(self.instrument, inst.Instrument):
                inst_name = self.instrument.nice_name().replace('/', '-')
            else:
                inst_name = self.instrument_name
            nice_name = f"{self.field.name}_{inst_name}_{fil.replace('_', '-')}_{self.date_str()}.fits"

            if img != img_projected:
                astm_rms = img_projected.extract_astrometry_err().value
                psf_fwhm = img_projected.extract_header_item(key="PSF_FWHM")
                psf_fwhm_err = img_projected.extract_header_item(key="PSF_FWHM_ERR")
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

            img.copy_with_outputs(os.path.join(
                self.data_path,
                nice_name)
            )

            if isinstance(self.field.survey, survey.Survey):
                refined_path = self.field.survey.refined_stage_path

                if refined_path is not None:
                    if self.field.survey.name == "FURBY":
                        cwd = os.getcwd()
                        os.chdir(refined_path)

                        u.system_command_verbose(
                            f"furby_archive {self.field.name} {img.filter.band_name} {img.path} --clobber",
                            error_on_exit_code=False
                        )

                        os.chdir(cwd)
                    else:
                        img.copy_with_outputs(
                            os.path.join(
                                refined_path,
                                # self.field.name,
                                # f"{self.instrument_name}_{fil}",
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

        for fil in images:
            img = images[fil]
            img.load_source_cat()
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
            print(f"Performing PSF measurements on {img}...")
            self.psf_stats[fil], _, _, _ = img.psf_diagnostics()
            self.psf_stats[fil]["file_path"] = img.path

        self.update_output_file()
        return self.psf_stats

    def _get_images(self, image_type: str) -> Dict[str, image.CoaddedImage]:
        """
        A helper method for finding the desired coadded image dictionary.
        :param image_type: "trimmed", "coadded", "unprojected" or "astrometry"
        :return: dict with filter names as keys and CoaddedImage objects as values.
        """
        if image_type == "final":
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

        if frame_type == "science":
            image_dict = self.frames_science
        elif frame_type == "reduced":
            image_dict = self.frames_reduced
        elif frame_type == "trimmed":
            image_dict = self.frames_trimmed
        elif frame_type == "normalised":
            image_dict = self.frames_normalised
        elif frame_type == "registered":
            image_dict = self.frames_registered
        elif frame_type == "astrometry":
            image_dict = self.frames_astrometry
        elif frame_type == "diagnosed":
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
            "std_pointings": self.std_pointings,
            "frames_final": self.frames_final,
            "frames_raw": _output_img_list(self.frames_raw),
            "frames_reduced": _output_img_dict_list(self.frames_reduced),
            "frames_normalised": _output_img_dict_list(self.frames_normalised),
            "frames_registered": _output_img_dict_list(self.frames_registered),
            "frames_astrometry": _output_img_dict_list(self.frames_astrometry),
            "frames_diagnosed": _output_img_dict_list(self.frames_diagnosed),
            "exp_time_mean": self.exp_time_mean,
            "exp_time_err": self.exp_time_err,
            "airmass_mean": self.airmass_mean,
            "airmass_err": self.airmass_err,
            "astrometry_successful": self.astrometry_successful,
            "astrometry_stats": self.astrometry_stats,
            "psf_stats": self.psf_stats
        })
        return output_dict

    def load_output_file(self, **kwargs):
        outputs = super().load_output_file(**kwargs)
        if isinstance(outputs, dict):
            frame_cls = image.ImagingImage.select_child_class(instrument=self.instrument_name, mode='imaging')
            coadd_class = image.CoaddedImage.select_child_class(instrument=self.instrument_name, mode='imaging')
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
            if "frames_raw" in outputs:
                for frame in set(outputs["frames_raw"]):
                    if os.path.isfile(frame):
                        self.add_frame_raw(raw_frame=frame)
            if "frames_reduced" in outputs:
                for fil in outputs["frames_reduced"]:
                    if outputs["frames_reduced"][fil] is not None:
                        for frame in set(outputs["frames_reduced"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_reduced(reduced_frame=frame)
            if "frames_normalised" in outputs:
                for fil in outputs["frames_normalised"]:
                    if outputs["frames_normalised"][fil] is not None:
                        for frame in set(outputs["frames_normalised"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_normalised(norm_frame=frame)
            if "frames_registered" in outputs:
                for fil in outputs["frames_registered"]:
                    if outputs["frames_registered"][fil] is not None:
                        for frame in set(outputs["frames_registered"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_registered(registered_frame=frame)
            if "frames_astrometry" in outputs:
                for fil in outputs["frames_astrometry"]:
                    if outputs["frames_astrometry"][fil] is not None:
                        for frame in set(outputs["frames_astrometry"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_astrometry(astrometry_frame=frame)
            if "frames_diagnosed" in outputs:
                for fil in outputs["frames_diagnosed"]:
                    if outputs["frames_diagnosed"][fil] is not None:
                        for frame in set(outputs["frames_diagnosed"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_diagnosed(diagnosed_frame=frame)
            if "coadded" in outputs:
                for fil in outputs["coadded"]:
                    if outputs["coadded"][fil] is not None:
                        self.add_coadded_image(img=outputs["coadded"][fil], key=fil, **kwargs)
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
            correct_to_epoch: bool = True
    ):
        print(f"correct_to_epoch 3: {correct_to_epoch}")
        if not isinstance(self.field, Field):
            raise ValueError("field has not been set for this observation.")
        self.field.retrieve_catalogue(cat_name=cat_name)
        index_path = os.path.join(config["top_data_dir"], "astrometry_index_files")
        u.mkdir_check(index_path)
        cat_index_path = os.path.join(index_path, cat_name)
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

        astm.generate_astrometry_indices(
            cat_name=cat_name,
            cat=cat,
            output_file_prefix=f"{cat_name}_index_{self.field.name}",
            index_output_dir=cat_index_path,
            fits_cat_output=csv_path.replace(".csv", ".fits"),
            p_lower=-1,
            p_upper=2,
            unique_id_prefix=unique_id_prefix,
        )

        return cat

    def epoch_gaia_catalogue(
            self,
            correct_to_epoch: bool = True
    ):
        print(f"correct_to_epoch 4: {correct_to_epoch}")
        if self.date is None:
            raise ValueError(f"{self}.date not set; needed to correct Gaia cat to epoch.")
        if correct_to_epoch:
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
                cls = image.ImagingImage.select_child_class(instrument=self.instrument_name)
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

    def add_frame_raw(self, raw_frame: Union[image.ImagingImage, str]):
        raw_frame, fil = self._check_frame(frame=raw_frame, frame_type="raw")
        self.check_filter(fil)
        if raw_frame is None:
            return None
        if raw_frame not in self.frames_raw:
            self.frames_raw.append(raw_frame)
        self.sort_frame(raw_frame, sort_key=fil)
        return raw_frame

    def add_frame_reduced(self, reduced_frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=reduced_frame, frames_dict=self.frames_reduced, frame_type="reduced")

    def add_frame_trimmed(self, trimmed_frame: image.ImagingImage):
        self._add_frame(frame=trimmed_frame, frames_dict=self.frames_trimmed, frame_type="reduced")

    def add_frame_registered(self, registered_frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=registered_frame, frames_dict=self.frames_registered, frame_type="registered")

    def add_frame_astrometry(self, astrometry_frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=astrometry_frame, frames_dict=self.frames_astrometry, frame_type="astrometry")

    def add_frame_diagnosed(self, diagnosed_frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=diagnosed_frame, frames_dict=self.frames_diagnosed, frame_type="diagnosed")

    def add_frame_normalised(self, norm_frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=norm_frame, frames_dict=self.frames_normalised, frame_type="reduced")

    def add_coadded_trimmed_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_trimmed)

    def add_coadded_unprojected_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_unprojected)

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

            obs.add_epoch(
                epoch_name=self.name,
                fil=fil,
                entry=entry
            )

        obs.write_master_imaging_table()

    @classmethod
    def from_params(cls, name: str, instrument: str, field: Union[Field, str] = None, old_format: bool = False):
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
        return cls.from_file(param_file=path, field=field)

    @classmethod
    def build_param_path(cls, instrument_name: str, field_name: str, epoch_name: str):
        path = u.mkdir_check_args(p.param_dir, "fields", field_name, "imaging", instrument_name)
        return os.path.join(path, f"{epoch_name}.yaml")

    @classmethod
    def build_data_path_absolute(cls, field: Field, instrument_name: str, name: str, date: Time = None):
        if date is not None:
            name_str = f"{date.isot}-{name}"
        else:
            name_str = name

        return u.mkdir_check_args(field.data_path, "imaging", instrument_name, name_str)

    @classmethod
    def from_file(cls, param_file: Union[str, dict], old_format: bool = False, field: Field = None):
        print("Initializing epoch...")

        name, param_file, param_dict = p.params_init(param_file)

        if param_dict is None:
            raise FileNotFoundError(f"There is no param file at {param_file}")

        pdict_backup = param_dict.copy()

        if old_format:
            instrument = "vlt-fors2"
        else:
            instrument = param_dict.pop("instrument").lower()

        if field is None:
            field = param_dict.pop("field")
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
            "skip":
                {"esoreflex_copy": False,
                 "sextractor_individual": False,
                 "sextractor": False,
                 "esorex": False,
                 },
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
        elif instrument in instruments_imaging:
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
            suppress_select: bool = False,
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
                            iterate_uncertainty=True
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
            field: Union[str, Field] = None,
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
            overwrite=overwrite)

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
                overwrite=overwrite)

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
            print(f"Adding {flat} to database.")
            sys_str = f"caldb add {flat}"
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
    def from_file(cls, param_file: Union[str, dict], name: str = None, field: Field = None):

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
            field: Field = None,
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
                print(f"Performing PSF measurements on {img}...")
                self.psf_stats[fil], _, _, _ = img.psf_diagnostics()

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
    def from_file(cls, param_file: Union[str, dict], name: str = None, field: Field = None):

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
            field: Union[str, Field] = None,
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
    def from_file(cls, param_file: Union[str, dict], name: str = None, field: Field = None):
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


class ESOImagingEpoch(ImagingEpoch):
    instrument_name = "dummy-instrument"
    mode = "imaging"

    def __init__(
            self,
            name: str = None,
            field: Field = None,
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

        self.frames_science = {}
        self.frames_flat = {}
        self.frames_bias = []
        self.frames_raw = []

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

        for row in tbl_full:
            path = os.path.join(raw_dir, row["identifier"])
            cls = image.ImagingImage.select_child_class(instrument=self.instrument_name, mode="imaging")
            img = image.from_path(path, cls=cls)
            img.extract_frame_type()
            img.extract_filter()
            u.debug_print(1, self.instrument_name, cls, img.name, img.frame_type)
            # The below will also update the filter list.
            u.debug_print(
                2,
                f"_initial_setup(): Adding frame {img.name}, type {img.frame_type}/{type(img)}, to {self}, type {type(self)}")
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
                print(f"Copying {file} to ESOReflex input directory...")
                origin = os.path.join(raw_dir, file)
                shutil.copy(origin, os.path.join(config["esoreflex_input_dir"], inst_reflex_dir))
                print("Done.")

                if survey_raw_path is not None:
                    survey_raw_path_file = os.path.join(
                        survey_raw_path,
                        file
                    )
                    print(f"Copying {file} to {survey_raw_path_file}...")
                    shutil.copy(
                        origin,
                        survey_raw_path_file
                    )
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
        self._sort_after_esoreflex(output_dir=output_dir, **kwargs)

    def _sort_after_esoreflex(self, output_dir: str, **kwargs):

        self.frames_reduced = {}
        self.frames_esoreflex_backgrounds = {}

        # Check for alternate directory.
        if "alternate_dir" in kwargs and isinstance(kwargs["alternate_dir"], str):
            eso_dir = kwargs["alternate_dir"]
            expect_sorted = True
            if "expect_sorted" in kwargs and isinstance(kwargs["expect_sorted"], bool):
                expect_sorted = kwargs["expect_sorted"]

        else:
            eso_dir = p.config['esoreflex_output_dir']
            expect_sorted = False

        if "delete_eso_output" in kwargs:
            delete_output = kwargs["delete_eso_output"]
        else:
            delete_output = False

        if os.path.isdir(eso_dir):
            data_dir = self.data_path

            if expect_sorted:
                print(f"Copying files from {eso_dir} to {output_dir}")
                shutil.rmtree(output_dir)
                shutil.copytree(
                    eso_dir,
                    output_dir,
                )

                science = os.path.join(output_dir, "science")
                for fil in filter(lambda d: os.path.isdir(os.path.join(science, d)), os.listdir(science)):
                    output_subdir = os.path.join(science, fil)
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

                print(f"Looking for data with object '{obj}' and MJD of observation {mjd} inside {eso_dir}")
                # Look for files with the appropriate object and MJD, as recorded in output_values

                # List directories in eso_output_dir; these are dates on which data was reduced using ESOReflex.
                date_dirs = filter(lambda d: os.path.isdir(os.path.join(eso_dir, d)), os.listdir(eso_dir))
                date_dirs = map(lambda d: os.path.join(eso_dir, d), date_dirs)
                for date_dir in date_dirs:
                    # List directories within 'reduction date' directories.
                    # These should represent individual images reduced.

                    print(f"Searching {date_dir}")
                    eso_subdirs = filter(
                        lambda d: os.path.isdir(os.path.join(date_dir, d)),
                        os.listdir(date_dir))
                    for subdirectory in eso_subdirs:
                        subpath = os.path.join(date_dir, subdirectory)
                        print(f"\tSearching {subpath}")
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
                                    f"{self.name}_{subdirectory}_{suffix}")
                                # Copy file to new location.
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
        else:
            raise IOError(f"ESO output directory '{eso_dir}' not found.")

        if not self.frames_reduced:
            u.debug_print(2, "ESOImagingEpoch._sort_after_esoreflex(): kwargs ==", kwargs)
            print(f"WARNING: No reduced frames were found in the target directory {eso_dir}.")

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
            frame_type="reduced")

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
                            self.add_frame_trimmed(trimmed_frame=frame)
            if "frames_esoreflex_backgrounds" in outputs:
                for fil in outputs["frames_esoreflex_backgrounds"]:
                    if outputs["frames_esoreflex_backgrounds"][fil] is not None:
                        for frame in outputs["frames_esoreflex_backgrounds"][fil]:
                            self.add_frame_background(background_frame=frame)

        return outputs

    @classmethod
    def from_file(cls, param_file: Union[str, dict], name: str = None, field: Field = None):

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


class FORS2ImagingEpoch(ESOImagingEpoch):
    instrument_name = "vlt-fors2"
    frame_class = image.FORS2Image
    coadded_class = image.FORS2CoaddedImage

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
            "coadd": ie_stages["coadd"],
            "correct_astrometry_coadded": ie_stages["correct_astrometry_coadded"],
            "trim_coadded": ie_stages["trim_coadded"],
            "source_extraction": ie_stages["source_extraction"],
            "photometric_calibration": ie_stages["photometric_calibration"],
            "dual_mode_source_extraction": ie_stages["dual_mode_source_extraction"],
            "get_photometry": ie_stages["get_photometry"],
            # "get_photometry_all": ie_stages["get_photometry_all"]
        }

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
                        if successful is None:
                            print(f"Astrometry.net failed to solve any of the chip {j + 1} images. "
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

    def estimate_atmospheric_extinction(self, n: int = 10, output: str = None):
        mjd = self.date.mjd
        fils_known = []
        tbls_known = {}

        fils_find = []

        for fil_name in inst.FORS2Filter.qc1_retrievable:
            fil = inst.Filter.from_params(fil_name, instrument_name="vlt-fors2")
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
            if fil_name not in inst.FORS2Filter.qc1_retrievable:
                fil = inst.Filter.from_params(fil_name, instrument_name="vlt-fors2")
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
            model = fitter(model_init, lambdas_known, extinctions_known, weights=1 / extinctions_known_err)

            curve_err = u.root_mean_squared_error(model_values=model(lambdas_known), obs_values=extinctions_known)

            results_tbl["curve_err"].append(curve_err)

            extinctions_find = model(lambdas_find)

            lambda_eff_fit = np.linspace(3000, 10000)
            plt.close()
            plt.plot(lambda_eff_fit, model(lambda_eff_fit))
            plt.scatter(lambdas_known, extinctions_known, label="Known")
            for j, m in enumerate(mjds):
                plt.text(lambdas_known[j], extinctions_known[j], m)
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

    def photometric_calibration(
            self,
            output_path: str,
            image_dict: dict,
            **kwargs
    ):
        import craftutils.wrap.esorex as esorex

        # if "image_type" in kwargs and kwargs["image_type"] is not None:
        #     image_type = kwargs["image_type"]
        # else:
        #     image_type = "final"

        suppress_select = False
        if "suppress_select" in kwargs and kwargs["suppress_select"] is not None:
            suppress_select = kwargs.pop("suppress_select")

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
                    bias_frames=list(map(lambda b: b.path, bias_set)),
                    output_dir=output_path,
                    output_filename=f"master_bias_{chip}.fits",
                    sof_name=f"bias_{chip}.sof"
                )
            except SystemError:
                continue

            for fil in image_dict:
                # Generate master flat per-filter, per-chip
                if fil not in flat_sets or fil in inst.FORS2Filter.qc1_retrievable:  # Time-saver
                    continue
                img = image_dict[fil]
                if "calib_pipeline" in img.zeropoints:
                    img.zeropoints.pop("calib_pipeline")
                flat_set = list(map(lambda b: b.path, flat_sets[fil][i]))
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
                            standard_img=std.path,
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
                            print(
                                "System error encountered while doing esorex processing; possibly impossible value encountered. Skipping.")

                    else:
                        print(f"Insufficient standard observations to calculate esorex zeropoint for {img}")

        print("Estimating zeropoints from standard observations...")
        for jname in self.std_epochs:
            std_epoch = self.std_epochs[jname]
            std_epoch.photometric_calibration()
            for fil in image_dict:
                img = image_dict[fil]
                # We save time by only bothering with non-qc1-obtainable zeropoints.
                if fil in std_epoch.frames_reduced and fil not in inst.FORS2Filter.qc1_retrievable:
                    for std in std_epoch.frames_reduced[fil]:
                        img.add_zeropoint_from_other(std)

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

            if self.coadded_unprojected[fil] is not None:
                self.coadded_unprojected[fil].zeropoints = img.zeropoints
                self.coadded_unprojected[fil].zeropoint_best = img.zeropoint_best
                self.coadded_unprojected[fil].update_output_file()

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
    def from_file(cls, param_file: Union[str, dict], name: str = None, old_format: bool = False, field: Field = None):

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


class SpectroscopyEpoch(Epoch):
    instrument_name = "dummy-instrument"
    mode = "spectrocopy"
    grisms = {}
    frame_class = image.Spectrum
    coadded_class = image.Coadded1DSpectrum

    def __init__(self,
                 param_path: str = None,
                 name: str = None,
                 field: Union[str, Field] = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 program_id: str = None,
                 target: str = None,
                 grism: str = None,
                 decker: str = None
                 ):
        super().__init__(param_path=param_path, name=name, field=field, data_path=data_path, instrument=instrument,
                         date=date, program_id=program_id, target=target)

        self.decker = decker
        self.decker_std = decker
        self.grism = grism
        if grism is None or grism not in self.grisms:
            warnings.warn("grism not configured.")

        self.obj = target

        self._path_2_pypeit()
        self.standards_raw = []
        self._instrument_pypeit = self.instrument_name.replace('-', '_')

        self._pypeit_file = None
        self._pypeit_sorted_file = None
        self._pypeit_coadd1d_file = None

    def _path_0_raw(self):
        if self.data_path is not None and "raw_dir" not in self.paths:
            self.paths["raw_dir"] = os.path.join(self.data_path, epoch_stage_dirs["0-download"])

    def proc_pypeit_flux(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do fluxing with PypeIt?", stage_name='4-pypeit_flux_calib'):
            self._pypeit_flux()
            self.stages_complete['4-pypeit_flux_calib'] = Time.now()
            self.update_output_file()

    def _pypeit_flux(self):
        pypeit_run_dir = self.get_path("pypeit_run_dir")
        pypeit_science_dir = os.path.join(pypeit_run_dir, "Science")
        std_reduced_filename = filter(lambda f: "spec1d" in f and "STD,FLUX" in f and f.endswith(".fits"),
                                      os.listdir(pypeit_science_dir)).__next__()
        std_reduced_path = os.path.join(pypeit_science_dir, std_reduced_filename)
        print(f"Using {std_reduced_path} for fluxing.")

        sensfunc_path = os.path.join(pypeit_run_dir, "sens.fits")
        # Generate sensitivity function from standard observation
        spec.pypeit_sensfunc(spec1dfile=std_reduced_path, outfile=sensfunc_path)
        # Generate flux setup file.
        spec.pypeit_flux_setup(sci_path=pypeit_science_dir, run_dir=pypeit_run_dir)
        flux_setup_path = os.path.join(pypeit_run_dir, self.pypeit_flux_title())
        # Insert name of sensitivity file to flux setup file.
        with open(flux_setup_path, "r") as flux_setup:
            flux_lines = flux_setup.readlines()
        file_first = flux_lines.index("flux read\n") + 1
        flux_lines[file_first] = flux_lines[file_first][:-1] + " " + sensfunc_path + "\n"
        # Write back to file.
        u.write_list_to_file(path=flux_setup_path, file=flux_lines)
        # Run pypeit_flux_calib
        os.system(f"pypeit_flux_calib {flux_setup_path}")

        self.set_path("pypeit_sensitivity_file", sensfunc_path)
        self.set_path("pypeit_std_reduced", std_reduced_path)
        self.set_path("pypeit_science_dir", pypeit_science_dir)
        self.set_path("pypeit_flux_setup", flux_setup_path)

    def pypeit_flux_title(self):
        return f"{self._instrument_pypeit}.flux"

    def read_pypeit_sorted_file(self):
        if "pypeit_setup_dir" in self.paths and self.paths["pypeit_setup_dir"] is not None:
            setup_files = self.paths["pypeit_setup_dir"]
            sorted_path = os.path.join(setup_files,
                                       filter(lambda f: f.endswith(".sorted"), os.listdir(setup_files)).__next__())
            with open(sorted_path) as sorted_file:
                self._pypeit_sorted_file = sorted_file.readlines()
        else:
            raise KeyError("pypeit_setup_dir has not been set.")

    def setup_info(self, setup: str):
        """
        Pulls setup info from a pypeit .sorted file.
        :param setup:
        :return:
        """
        file = self._get_pypeit_sorted_file()
        # Find start of setup description
        setup_start = file.index(f"Setup {setup}\n")
        setup_dict = {}
        i = setup_start + 1
        line = file[i]
        # Assemble a dictionary of the setup parameters.
        while line != "#---------------------------------------------------------\n":
            while line[0] == " ":
                line = line[1:]
            line = line[:-1]
            key, value = line.split(": ")
            setup_dict[key] = value
            i += 1
            line = file[i]
        return setup_dict

    def _read_pypeit_file(self, filename):
        pypeit_run_dir = os.path.join(self.paths['pypeit_dir'], filename)
        self.set_path("pypeit_run_dir", pypeit_run_dir)
        self.set_path("pypeit_file", os.path.join(pypeit_run_dir, f"{filename}.pypeit"))
        # Retrieve text from .pypeit file
        with open(self.get_path("pypeit_file"), 'r') as pypeit_file:
            pypeit_lines = pypeit_file.readlines()
            self._set_pypeit_file(pypeit_lines)
        f_start = pypeit_lines.index("data read\n") + 3
        f_end = pypeit_lines.index("data end\n")
        for line in pypeit_lines[f_start:f_end]:
            raw = image.RawSpectrum.from_pypeit_line(line=line, pypeit_raw_path=self.paths["raw_dir"])
            self.add_frame_raw(raw)
        return pypeit_lines

    def read_pypeit_file(self, setup: str):
        if "pypeit_dir" in self.paths and self.paths["pypeit_dir"] is not None:
            filename = f"{self._instrument_pypeit}_{setup}"
            self._read_pypeit_file(filename=filename)
            return self._pypeit_file
        else:
            raise KeyError("pypeit_run_dir has not been set.")

    def write_pypeit_file_science(self):
        """
        Rewrites the stored .pypeit file to disk at its original path.
        :return: path of .pypeit file.
        """
        pypeit_lines = self._get_pypeit_file()
        if pypeit_lines is not None:
            pypeit_file_path = self.get_path("pypeit_file")
            u.write_list_to_file(path=pypeit_file_path, file=pypeit_lines)
        else:
            raise ValueError("pypeit_file has not yet been read.")
        return pypeit_file_path

    def add_pypeit_user_param(self, param: list, value: str, file_type: str = "pypeit"):
        """
        Inserts a parameter for the PypeIt run at the correct point in the stored .pypeit file.
        :param param: For m
        :param value:
        :return:
        """
        if file_type == "pypeit":
            pypeit_file = self._get_pypeit_file()
        elif file_type == "coadd1d":
            pypeit_file = self._get_pypeit_coadd1d_file()
        else:
            raise ValueError(f"file_type {file_type} not recognised.")

        if pypeit_file is not None:
            # Build the final line of the setting specially.
            setting = "\t" * (len(param) - 1) + f"{param.pop()} = {value}\n"
            p_start = pypeit_file.index("# User-defined execution parameters\n") + 1
            insert_here = False
            # For each level of the param list, look to see if it's already there.
            for i, line in pypeit_file[p_start]:
                if param[0] in line:
                    p_start = i
                    break

            for i, par in enumerate(param):
                # Encase each level of the parameter in the correct number of square brackets and tabs.
                par = "\t" * i + "[" * (i + 1) + par + "]" * (i + 1) + "\n"
                # First, check if param sub-headings are already there:
                if par in pypeit_file and not insert_here:
                    p_start = pypeit_file.index(par) + 1
                else:
                    # Insert the line at correct position.
                    pypeit_file.insert(p_start, par)
                    p_start += 1
                    insert_here = True
            # Insert the final line.
            pypeit_file.insert(p_start, setting)

            if file_type == "pypeit":
                self._set_pypeit_file(lines=pypeit_file)
            elif file_type == "coadd1d":
                self._set_pypeit_coadd1d_file(lines=pypeit_file)

        else:
            raise ValueError("pypeit_file has not yet been read.")

    def add_pypeit_file_lines(self, lines: list):
        if self._get_pypeit_file() is not None:
            # Remove last two lines of file ("data end")
            pypeit_lines = self._pypeit_file[:-2]
            # Insert desired lines
            pypeit_lines += lines
            # Reinsert last two lines.
            pypeit_lines += ["data end\n", "\n"]
            self._pypeit_file = pypeit_lines
        else:
            raise ValueError("pypeit_file has not yet been read.")

    def _output_dict(self):
        output_dict = super()._output_dict()
        output_dict.update({"binning": self.binning,
                            "decker": self.decker})
        return output_dict

    def proc_pypeit_setup(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do PypeIt setup?", stage_name='2-pypeit_setup'):
            pass

    def proc_pypeit_run(self, no_query: bool = False, do_not_reuse_masters=False, **kwargs):
        if no_query or self.query_stage("Run PypeIt?", stage_name='3-pypeit_run'):
            pass

    def proc_pypeit_coadd(self, no_query: bool = False, **kwargs):
        pass

    def proc_convert_to_marz_format(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Convert co-added 1D spectra to Marz format?", stage_name='6-marz-format'):
            pass

    def _path_2_pypeit(self):
        if self.data_path is not None and "pypeit_dir" not in self.paths:
            self.paths["pypeit_dir"] = os.path.join(self.data_path, epoch_stage_dirs["2-pypeit"])

    def find_science_attributes(self):
        frames = self.get_frames_science()
        if frames:
            frame = frames[0]
            self.set_binning(frame.binning)
            self.set_decker(frame.decker)
        else:
            raise ValueError(f"Science frames list is empty.")
        return frame

    def find_std_attributes(self):
        frames = self.get_frames_standard()
        if frames:
            frame = frames[0]
            self.set_binning_std(frame.binning)
            self.set_decker_std(frame.decker)
        else:
            raise ValueError(f"Standard frames list is empty.")
        return frame

    def get_frames_science(self):
        return self.frames_science

    def get_frames_standard(self):
        return self.frames_standard

    def get_decker(self):
        return self.decker

    def set_decker(self, decker: str):
        self.decker = decker
        return decker

    def get_decker_std(self):
        return self.decker_std

    def set_decker_std(self, decker: str):
        self.decker_std = decker
        return decker

    def _set_pypeit_file(self, lines: list):
        self._pypeit_file = lines

    def _get_pypeit_file(self):
        return self._pypeit_file

    def _set_pypeit_coadd1d_file(self, lines: list):
        self._pypeit_coadd1d_file = lines

    def _get_pypeit_coadd1d_file(self):
        return self._pypeit_coadd1d_file

    def _get_pypeit_sorted_file(self):
        return self._pypeit_sorted_file

    @classmethod
    def stages(cls):
        param_dict = super().stages()
        param_dict.update({
            "2-pypeit_setup": None,
            "3-pypeit_run": None,
            "4-pypeit_flux_calib": None
        })
        return param_dict

    @classmethod
    def select_child_class(cls, instrument: str):
        instrument = instrument.lower()
        if instrument == "vlt-fors2":
            return FORS2SpectroscopyEpoch
        elif instrument == "vlt-xshooter":
            return XShooterSpectroscopyEpoch
        elif instrument in instruments_spectroscopy:
            return SpectroscopyEpoch
        else:
            raise ValueError(f"Unrecognised instrument {instrument}")

    @classmethod
    def from_file(cls, param_file: Union[str, dict], field: Field = None):
        name, param_file, param_dict = p.params_init(param_file)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")
        instrument = param_dict["instrument"].lower()
        if field is None:
            field = param_dict.pop("field")
        if 'target' in param_dict:
            target = param_dict.pop('target')
        else:
            target = None
        sub_cls = cls.select_child_class(instrument=instrument)
        # if sub_cls is SpectroscopyEpoch:
        return sub_cls(
            name=name,
            field=field,
            param_path=param_file,
            data_path=os.path.join(config["top_data_dir"], param_dict['data_path']),
            instrument=instrument,
            date=param_dict["date"],
            program_id=param_dict["program_id"],
            target=target,
            **param_dict
        )
        # else:
        # return sub_cls.from_file(param_file=param_file, field=field)

    @classmethod
    def from_params(cls, name, field: Union[Field, str] = None, instrument: str = None):
        if name in active_epochs:
            return active_epochs[name]
        print("Initializing epoch...")
        instrument = instrument.lower()
        field_name, field = cls._from_params_setup(name=name, field=field)
        path = cls.build_param_path(
            field_name=field_name,
            instrument_name=instrument,
            epoch_name=name)
        return cls.from_file(param_file=path, field=field)

    @classmethod
    def build_param_path(cls, field_name: str, instrument_name: str, epoch_name: str):
        return os.path.join(p.param_dir, "fields", field_name, "spectroscopy", instrument_name, epoch_name)


class ESOSpectroscopyEpoch(SpectroscopyEpoch):
    def __init__(self,
                 param_path: str = None,
                 name: str = None,
                 field: Union[str, Field] = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 program_id: str = None,
                 grism: str = None
                 ):
        super().__init__(param_path=param_path,
                         name=name,
                         field=field,
                         data_path=data_path,
                         instrument=instrument,
                         date=date,
                         program_id=program_id,
                         grism=grism)
        # Data reduction paths

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)
        if "do_not_reuse_masters" in kwargs:
            do_not_reuse_masters = kwargs["do_not_reuse_masters"]
        else:
            do_not_reuse_masters = False
        self.proc_raw()
        self.proc_initial_setup()
        self.proc_pypeit_setup()
        self.proc_pypeit_run(do_not_reuse_masters=do_not_reuse_masters)
        self.proc_pypeit_flux()
        self.proc_pypeit_coadd()
        self.proc_convert_to_marz_format()

    def proc_raw(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Download raw data from ESO archive?", stage_name='0-download'):
            self._path_0_raw()
            r = self.retrieve()
            if r:
                self.stages_complete['0-download'] = Time.now()
                self.update_output_file()

    def _initial_setup(self):
        self._path_0_raw()
        m_path = os.path.join(self.paths["raw_dir"], "M")
        u.mkdir_check(m_path)
        os.system(f"mv {os.path.join(self.paths['raw_dir'], 'M.')}* {m_path}")
        image.fits_table_all(input_path=self.paths["raw_dir"],
                             output_path=os.path.join(self.data_path, f"{self.name}_fits_table_science.csv"))
        image.fits_table_all(input_path=self.paths["raw_dir"],
                             output_path=os.path.join(self.data_path, f"{self.name}_fits_table_all.csv"),
                             science_only=False)

    def retrieve(self):
        """
        Check ESO archive for the epoch raw frames, and download those frames and associated files.
        :return:
        """
        r = []
        if "raw_dir" in self.paths:
            r = _retrieve_eso_epoch(self, path=self.paths["raw_dir"])
        else:
            warnings.warn("raw_dir has not been set. Retrieve could not be run.")
        return r

    @classmethod
    def stages(cls):
        param_dict = super().stages()
        param_dict.update({"0-download": None})
        return param_dict


class FORS2SpectroscopyEpoch(ESOSpectroscopyEpoch):
    instrument_name = "vlt-fors2"
    _instrument_pypeit = "vlt_fors2"
    grisms = {
        "GRIS_300I": {
            "lambda_min": 6000 * units.angstrom,
            "lambda_max": 11000 * units.angstrom
        }}

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)

    def proc_pypeit_setup(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do PypeIt setup?", stage_name='2-pypeit_setup'):
            self._path_2_pypeit()
            setup_files = os.path.join(self.paths["pypeit_dir"], 'setup_files', '')
            self.paths["pypeit_setup_dir"] = setup_files
            os.system(f"rm {setup_files}*")
            # Generate .sorted file and others
            spec.pypeit_setup(root=self.paths['raw_dir'], output_path=self.paths['pypeit_dir'],
                              spectrograph=self._instrument_pypeit)
            # Generate files to use for run. Set cfg_split to "A" because that corresponds to Chip 1, which is the only
            # one we need to worry about.
            spec.pypeit_setup(root=self.paths['raw_dir'], output_path=self.paths['pypeit_dir'],
                              spectrograph=self._instrument_pypeit, cfg_split="A")
            # Read .sorted file
            self.read_pypeit_sorted_file()
            # Retrieve bias files from .sorted file.
            bias_lines = list(filter(lambda s: "bias" in s and "CHIP1" in s, self._pypeit_sorted_file))
            # Find line containing information for standard observation.
            std_line = filter(lambda s: "standard" in s and "CHIP1" in s, self._pypeit_sorted_file).__next__()
            std_raw = image.RawSpectrum.from_pypeit_line(std_line, pypeit_raw_path=self.paths['raw_dir'])
            self.standards_raw.append(std_raw)
            std_start_index = self._pypeit_sorted_file.index(std_line)
            # Find last line of the std-obs configuration (encapsulating the required calibration files)
            std_end_index = self._pypeit_sorted_file[std_start_index:].index(
                "##########################################################\n") + std_start_index
            std_lines = self._pypeit_sorted_file[std_start_index:std_end_index]
            # Read in .pypeit file
            self.read_pypeit_file(setup="A")
            # Add lines to set slit prediction to "nearest" in .pypeit file.
            self.add_pypeit_user_param(param=["calibrations", "slitedges", "sync_predict"], value="nearest")
            # Insert bias lines from .sorted file
            self.add_pypeit_file_lines(lines=bias_lines + std_lines)
            # Write modified .pypeit file back to disk.
            self.write_pypeit_file_science()

            self.stages_complete['2-pypeit_setup'] = Time.now()
            self.update_output_file()

    def proc_pypeit_run(self, no_query: bool = False, do_not_reuse_masters: bool = False, **kwargs):
        if no_query or self.query_stage("Run PypeIt?", stage_name='3-pypeit_run'):
            spec.run_pypeit(pypeit_file=self.paths['pypeit_file'],
                            redux_path=self.paths['pypeit_run_dir'],
                            do_not_reuse_masters=do_not_reuse_masters)
            self.stages_complete['3-pypeit_run'] = Time.now()
            self.update_output_file()

    def proc_pypeit_coadd(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage(
                "Do coaddition with PypeIt?\nYou should first inspect the 2D spectra to determine which objects to co-add.",
                stage_name='5-pypeit_coadd'):
            for file in filter(lambda f: "spec1d" in f, os.listdir(self.paths["pypeit_science_dir"])):
                path = os.path.join(self.paths["pypeit_science_dir"], file)
                os.system(f"pypeit_show_1dspec {path}")


class XShooterSpectroscopyEpoch(ESOSpectroscopyEpoch):
    _instrument_pypeit = "vlt_xshooter"
    grisms = {'uvb': {"lambda_min": 300 * units.nm,
                      "lambda_max": 550 * units.nm},
              "vis": {"lambda_min": 550 * units.nm,
                      "lambda_max": 1000 * units.nm},
              "nir": {"lambda_min": 1000 * units.nm,
                      "lambda_max": 2500 * units.nm}}

    def __init__(self,
                 param_path: str = None,
                 name: str = None,
                 field: Union[str, Field] = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 program_id: str = None,
                 ):

        super().__init__(param_path=param_path,
                         name=name,
                         field=field,
                         data_path=data_path,
                         instrument=instrument,
                         date=date,
                         program_id=program_id
                         )

        self.frames_raw = {"uvb": [],
                           "vis": [],
                           "nir": []}
        self.frames_bias = {"uvb": [],
                            "vis": [],
                            "nir": []}
        self.frames_standard = {"uvb": [],
                                "vis": [],
                                "nir": []}
        self.frames_science = {"uvb": [],
                               "vis": [],
                               "nir": []}
        self.frames_dark = {"uvb": [],
                            "vis": [],
                            "nir": []}
        self._pypeit_file = {"uvb": None,
                             "vis": None,
                             "nir": None}
        self._pypeit_file_std = {"uvb": None,
                                 "vis": None,
                                 "nir": None}
        self._pypeit_sorted_file = {"uvb": None,
                                    "vis": None,
                                    "nir": None}
        self._pypeit_coadd1d_file = {"uvb": None,
                                     "vis": None,
                                     "nir": None}
        self._pypeit_user_param_start = {"uvb": None,
                                         "vis": None,
                                         "nir": None}
        self._pypeit_user_param_end = {"uvb": None,
                                       "vis": None,
                                       "nir": None}

        self.binning = {"uvb": None,
                        "vis": None,
                        "nir": None}
        self.binning_std = {"uvb": None,
                            "vis": None,
                            "nir": None}
        self.decker = {"uvb": None,
                       "vis": None,
                       "nir": None}
        self.decker_std = {"uvb": None,
                           "vis": None,
                           "nir": None}

        self._cfg_split_letters = {"uvb": None,
                                   "vis": None,
                                   "nir": None}

        self.load_output_file()
        self._current_arm = None

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)
        # self.proc_pypeit_coadd()

    def proc_pypeit_setup(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do PypeIt setup?", stage_name='2-pypeit_setup'):
            self._path_2_pypeit()
            setup_files = os.path.join(self.paths["pypeit_dir"], 'setup_files', '')
            self.paths["pypeit_setup_dir"] = setup_files
            os.system(f"rm {setup_files}*")
            for arm in self.grisms:
                self._current_arm = arm
                spec.pypeit_setup(root=self.paths['raw_dir'], output_path=self.paths['pypeit_dir'],
                                  spectrograph=f"{self._instrument_pypeit}_{arm}")
                # Read .sorted file
                self.read_pypeit_sorted_file()
                setup = self._cfg_split_letters[arm]
                spec.pypeit_setup(root=self.paths['raw_dir'], output_path=self.paths['pypeit_dir'],
                                  spectrograph=f"{self._instrument_pypeit}_{arm}", cfg_split=setup)
                # Retrieve text from .pypeit file
                self.read_pypeit_file(setup=setup)
                # Add parameter to use dark frames for NIR reduction.
                if arm == "nir":
                    self.add_pypeit_user_param(param=["calibrations", "pixelflatframe", "process", "use_darkimage"],
                                               value="True")
                    self.add_pypeit_user_param(param=["calibrations", "illumflatframe", "process", "use_darkimage"],
                                               value="True")
                    self.add_pypeit_user_param(param=["calibrations", "traceframe", "process", "use_darkimage"],
                                               value="True")
                self.find_science_attributes()
                # For X-Shooter, we need to reduce the standards separately due to the habit of observing them with
                # different decker (who knows)
                self.find_std_attributes()
                # Remove incompatible binnings and frametypes
                print(f"\nRemoving incompatible files for {arm} arm:")
                pypeit_file = self._get_pypeit_file()
                # pypeit_file_std = pypeit_file.copy()
                decker = self.get_decker()
                binning = self.get_binning()
                decker_std = self.get_decker_std()
                for raw_frame in self.frames_raw[arm]:
                    # Remove all frames with frame_type "None" from both science and standard lists.
                    if raw_frame.frame_type == "None":
                        pypeit_file.remove(raw_frame.pypeit_line)
                        # pypeit_file_std.remove(raw_frame.pypeit_line)
                    else:
                        # Remove files with incompatible binnings from science reduction list.
                        if raw_frame.binning != binning \
                                or raw_frame.decker not in ["Pin_row", decker, decker_std]:
                            pypeit_file.remove(raw_frame.pypeit_line)
                    # Special behaviour for NIR arm
                    if arm == "nir":
                        # For the NIR arm, PypeIt only works if you use the Science frames for the arc, tilt calib.
                        if raw_frame.frame_type in ["arc,tilt", "tilt,arc"]:
                            pypeit_file.remove(raw_frame.pypeit_line)
                            # pypeit_file_std.remove(raw_frame.pypeit_line)
                        elif raw_frame.frame_type == "science":
                            raw_frame.frame_type = "science,arc,tilt"
                            # Find original line in PypeIt file
                            to_replace = pypeit_file.index(raw_frame.pypeit_line)
                            # Rewrite pypeit line.
                            raw_frame.pypeit_line = raw_frame.pypeit_line.replace("science", "science,arc,tilt")
                            pypeit_file[to_replace] = raw_frame.pypeit_line
                        elif raw_frame.frame_type == "standard":
                            raw_frame.frame_type = "standard,arc,tilt"
                            # Find original line in PypeIt file
                            to_replace = pypeit_file.index(raw_frame.pypeit_line)
                            # Rewrite pypeit line.
                            raw_frame.pypeit_line = raw_frame.pypeit_line.replace("standard", "standard,arc,tilt")
                            pypeit_file[to_replace] = raw_frame.pypeit_line
                self._set_pypeit_file(pypeit_file)
                # self._set_pypeit_file_std(pypeit_file_std)
                self.write_pypeit_file_science()
                # std_path = os.path.join(self.paths["pypeit_dir"], self.get_path("pypeit_run_dir"), "Flux_Standards")
                # u.mkdir_check(std_path)
                # self.set_path("pypeit_dir_std", std_path)
                # self.set_path("pypeit_file_std",
                #              os.path.join(self.get_path("pypeit_run_dir"), f"vlt_xshooter_{arm}_std.pypeit"))
                # self.write_pypeit_file_std()
            self._current_arm = None
            self.stages_complete['2-pypeit_setup'] = Time.now()
            self.update_output_file()

    def proc_pypeit_run(self, no_query: bool = False, do_not_reuse_masters: bool = False, **kwargs):
        for i, arm in enumerate(self.grisms):
            # UVB not yet implemented in PypeIt, so we skip.
            if arm == "uvb":
                continue
            self._current_arm = arm
            if no_query or self.query_stage(f"Run PypeIt for {arm.upper()} arm?",
                                            stage_name=f'3.{i + 1}-pypeit_run_{arm}'):
                spec.run_pypeit(pypeit_file=self.get_path('pypeit_file'),
                                redux_path=self.get_path('pypeit_run_dir'),
                                do_not_reuse_masters=do_not_reuse_masters)
                self.stages_complete[f'3.{i + 1}-pypeit_run_{arm}'] = Time.now()
                self.update_output_file()
            # if arm != "nir" and self.query_stage(f"Run PypeIt on flux standards for {arm.upper()} arm?",
            #                                      stage=f'3.{i + 1}-pypeit_run_{arm}_std'):
            #     print(self.get_path('pypeit_file_std'))
            #     spec.run_pypeit(pypeit_file=self.get_path('pypeit_file_std'),
            #                     redux_path=self.get_path('pypeit_dir_std'),
            #                     do_not_reuse_masters=do_not_reuse_masters)
            #     self.stages_complete[f'3.{i + 1}-pypeit_run_{arm}'] = Time.now()
            #     self.update_output_file()
        self._current_arm = None

    def proc_pypeit_flux(self, no_query: bool = False, **kwargs):
        for i, arm in enumerate(self.grisms):
            # UVB not yet implemented in PypeIt, so we skip.
            if arm == "uvb":
                continue
            self._current_arm = arm
            if no_query or self.query_stage(f"Do PypeIt fluxing for {arm.upper()} arm?",
                                            stage_name=f'4.{i + 1}-pypeit_flux_calib_{arm}'):
                self._current_arm = arm
                self._pypeit_flux()
            self.stages_complete[f'4.{i + 1}-pypeit_flux_calib_{arm}'] = Time.now()
        self._current_arm = None
        self.update_output_file()

    def proc_pypeit_coadd(self, no_query: bool = False, **kwargs):
        for i, arm in enumerate(self.grisms):
            # UVB not yet implemented in PypeIt, so we skip.
            if arm == "uvb":
                continue
            self._current_arm = arm
            if no_query or self.query_stage(f"Do PypeIt coaddition for {arm.upper()} arm?",
                                            stage_name=f'5.{i + 1}-pypeit_coadd_{arm}'):
                run_dir = self.get_path("pypeit_run_dir")
                coadd_file_path = os.path.join(run_dir, f"{self._instrument_pypeit}_{arm}.coadd1d")
                self.set_path("pypeit_coadd1d_file", coadd_file_path)
                with open(coadd_file_path) as file:
                    coadd_file_lines = file.readlines()
                output_path = os.path.join(run_dir, f"{self.name}_{arm}_coadded.fits")
                sensfunc_path = self.get_path("pypeit_sensitivity_file")
                # Remove non-science files
                for line in coadd_file_lines[coadd_file_lines.index("coadd1d read\n"):]:
                    if "STD,FLUX" in line or "STD,TELLURIC" in line:
                        coadd_file_lines.remove(line)

                self._set_pypeit_coadd1d_file(coadd_file_lines)
                # Re-insert parameter lines
                self.add_pypeit_user_param(param=["coadd1d", "coaddfile"], value=output_path, file_type="coadd1d")
                self.add_pypeit_user_param(param=["coadd1d", "sensfuncfile"], value=sensfunc_path, file_type="coadd1d")
                self.add_pypeit_user_param(param=["coadd1d", "wave_method"], value="velocity", file_type="coadd1d")
                u.write_list_to_file(coadd_file_path, self._get_pypeit_coadd1d_file())
                spec.pypeit_coadd_1dspec(coadd1d_file=coadd_file_path)
                self.add_coadded_image(coadd_file_path, key=arm)

            self.stages_complete[f'5.{i + 1}-pypeit_coadd_{arm}'] = Time.now()

        self._current_arm = None

    def proc_convert_to_marz_format(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Convert co-added 1D spectra to Marz format?",
                                        stage_name='6-convert_to_marz_format'):
            for arm in self.coadded:
                self.coadded[arm].convert_to_marz_format()
            self.stages_complete[f'6-convert_to_marz_format'] = Time.now()

    def add_coadded_image(self, img: Union[str, image.Coadded1DSpectrum], **kwargs):
        arm = kwargs["key"]
        if isinstance(img, str):
            img = image.from_path(
                path=img,
                grism=arm,
                cls=image.Coadded1DSpectrum
            )
        img.epoch = self
        self.coadded[arm] = img
        return img

    def add_frame_raw(self, raw_frame: image.Image):
        arm = self._get_current_arm()
        self.frames_raw[arm].append(raw_frame)
        self.sort_frame(raw_frame)

    def sort_frame(self, frame: image.Image):
        arm = self._get_current_arm()
        if frame.frame_type == "bias":
            self.frames_bias[arm].append(frame)
        elif frame.frame_type == "science":
            self.frames_science[arm].append(frame)
        elif frame.frame_type == "standard":
            self.frames_standard[arm].append(frame)
        elif frame.frame_type == "dark":
            self.frames_dark[arm].append(frame)

    def read_pypeit_sorted_file(self):
        arm = self._get_current_arm()
        if "pypeit_setup_dir" in self.paths and self.paths["pypeit_setup_dir"] is not None:
            setup_files = self.paths["pypeit_setup_dir"]
            sorted_path = os.path.join(setup_files,
                                       filter(lambda f: f"vlt_xshooter_{arm}" in f and f.endswith(".sorted"),
                                              os.listdir(setup_files)).__next__())
            with open(sorted_path) as sorted_file:
                file = sorted_file.readlines()
            self._pypeit_sorted_file[arm] = file
            for setup in ["A", "B", "C"]:
                info = self.setup_info(setup=setup)
                arm_this = info["arm"].lower()
                self._cfg_split_letters[arm_this] = setup
        else:
            raise KeyError("pypeit_setup_dir has not been set.")

    def read_pypeit_file(self, setup: str):
        if "pypeit_dir" in self.paths and self.paths["pypeit_dir"] is not None:
            arm = self._get_current_arm()
            filename = f"{self._instrument_pypeit}_{arm}_{setup}"
            self._read_pypeit_file(filename=filename)
            return self._pypeit_file[arm]
        else:
            raise KeyError("pypeit_run_dir has not been set.")

    def pypeit_flux_title(self):
        return f"{self._instrument_pypeit}_{self._get_current_arm()}.flux"

    def get_path(self, key):
        key = self._get_key_arm(key)
        return self.paths[key]

    def set_path(self, key: str, value: str):
        key = self._get_key_arm(key)
        self.paths[key] = value

    def get_frames_science(self):
        return self.frames_science[self._get_current_arm()]

    def get_frames_standard(self):
        return self.frames_standard[self._get_current_arm()]

    def get_binning(self):
        return self.binning[self._get_current_arm()]

    def set_binning(self, binning: str):
        self.binning[self._get_current_arm()] = binning
        return binning

    def get_binning_std(self):
        return self.binning_std[self._get_current_arm()]

    def set_binning_std(self, binning: str):
        self.binning_std[self._get_current_arm()] = binning
        return binning

    def get_decker(self):
        return self.decker[self._get_current_arm()]

    def set_decker(self, decker: str):
        self.decker[self._get_current_arm()] = decker
        return decker

    def get_decker_std(self):
        return self.decker_std[self._get_current_arm()]

    def set_decker_std(self, decker: str):
        self.decker_std[self._get_current_arm()] = decker
        return decker

    def _get_current_arm(self):
        if self._current_arm is not None:
            return self._current_arm
        else:
            raise ValueError("self._current_arm is not set (no arm currently active).")

    def _set_pypeit_file(self, lines: list):
        self._pypeit_file[self._get_current_arm()] = lines

    def _get_pypeit_file(self):
        return self._pypeit_file[self._get_current_arm()]

    def _get_pypeit_sorted_file(self):
        return self._pypeit_sorted_file[self._get_current_arm()]

    def _set_pypeit_file_std(self, lines: list):
        self._pypeit_file_std[self._get_current_arm()] = lines

    def _get_pypeit_file_std(self):
        return self._pypeit_file_std[self._get_current_arm()]

    def _set_pypeit_coadd1d_file(self, lines: list):
        self._pypeit_coadd1d_file[self._get_current_arm()] = lines

    def _get_pypeit_coadd1d_file(self):
        return self._pypeit_coadd1d_file[self._get_current_arm()]

    def _get_key_arm(self, key):
        arm = self._get_current_arm()
        key = f"{arm}_{key}"
        return key

    @classmethod
    def stages(cls):
        param_dict = super().stages()
        param_dict.update({
            "2-pypeit_setup": None,
            "3.1-pypeit_run_uvb": None,
            "3.2-pypeit_run_vis": None,
            "3.3-pypeit_run_nir": None,
            "4.1-pypeit_flux_calib_uvb": None,
            "4.2-pypeit_flux_calib_vis": None,
            "4.3-pypeit_flux_calib_nir": None,
            "5.1-pypeit_coadd_uvb": None,
            "5.2-pypeit_coadd_vis": None,
            "5.3-pypeit_coadd_nir": None,
            "6-convert_to_marz_format": None
        })
        return param_dict

    def _output_dict(self):
        return {
            "stages": self.stages_complete,
            "paths": self.paths,
        }

    def load_output_file(self, **kwargs):
        outputs = super().load_output_file(mode="spectroscopy", **kwargs)
        if outputs not in [None, True, False]:
            self.stages_complete.update(outputs["stages"])
            if "paths" in outputs:
                self.paths.update(outputs[f"paths"])
        return outputs

    def write_pypeit_file_std(self):
        """
        Rewrites the stored .pypeit file to disk.
        :return: path of .pypeit file.
        """
        pypeit_lines = self._get_pypeit_file_std()
        if pypeit_lines is not None:
            pypeit_file_path = os.path.join(self.get_path("pypeit_file_std"), )
            u.write_list_to_file(path=pypeit_file_path, file=pypeit_lines)
        else:
            raise ValueError("pypeit_file_std has not yet been read.")
        return pypeit_file_path
