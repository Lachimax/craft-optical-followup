# Code by Lachlan Marnoch, 2021 - 2023
import os
import warnings
from typing import Union, List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as units
import astropy.table as table
import astropy.io.fits as fits
from astropy.visualization import make_lupton_rgb

import craftutils.astrometry as astm
import craftutils.observation as obs
import craftutils.observation.objects as objects
import craftutils.observation.image as image
import craftutils.observation.instrument as inst
import craftutils.observation.epoch as ep
import craftutils.observation.survey as survey
import craftutils.params as p
import craftutils.plotting as pl
import craftutils.retrieve as retrieve
import craftutils.utils as u

__all__ = []

# pl.latex_setup()

config = p.config

instruments_imaging = p.instruments_imaging
instruments_imaging.sort()
instruments_spectroscopy = p.instruments_spectroscopy
instruments_spectroscopy.sort()
surveys = p.surveys

active_fields = {}


@u.export
def expunge_fields():
    field_list = active_fields.keys()
    for field_name in field_list:
        del active_fields[field_name]


def select_instrument(mode: str):
    if mode == "imaging":
        options = instruments_imaging
    elif mode == "spectroscopy":
        options = instruments_spectroscopy
    else:
        raise ValueError("Mode must be 'imaging' or 'spectroscopy'.")
    _, instrument = u.select_option("Select an instrument:", options=options)
    return instrument


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


@u.export
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
        self.data_path = os.path.join(p.data_dir, data_path)
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

        if objs is not None:
            for obj in objs:
                if type(objs) is dict:
                    if obj != "<name>":
                        obj_dict = objs[obj]
                        if "name" not in obj_dict or obj_dict["name"] is None:
                            obj_dict["name"] = obj
                    else:
                        continue
                elif type(objs) is list:
                    obj_dict = obj
                else:
                    break
                if "position" not in obj_dict:
                    obj_dict = {"position": obj_dict}
                if obj_dict["name"] is None:
                    continue
                self.add_object_from_dict(obj_dict)

        self.gather_objects()

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

    def gather_objects(self, quiet: bool = True):
        if not quiet:
            print(f"Searching for object param files...")

        if self.param_dir is not None:
            obj_path = self._obj_path()
            if not quiet:
                print(f"Looking in {obj_path}")

            obj_params = list(
                filter(lambda f: f.endswith(".yaml") and not f.endswith("backup.yaml"), os.listdir(obj_path)))
            obj_params.sort()
            for obj_param in obj_params:
                obj_name = obj_param[:obj_param.find(".yaml")]
                param_path = os.path.join(obj_path, obj_param)
                obj_dict = p.load_params(file=param_path)
                obj_dict["param_path"] = param_path
                if "name" not in obj_dict:
                    obj_dict["name"] = obj_name
                self.add_object_from_dict(obj_dict=obj_dict)

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
                epoch_params = list(
                    filter(
                        lambda f: f.endswith(".yaml") and not f.endswith("backup.yaml"),
                        os.listdir(instrument_path)
                    ))
                epoch_params.sort()
                for epoch_param in epoch_params:
                    epoch_name = epoch_param[:epoch_param.find(".yaml")]
                    param_path = os.path.join(instrument_path, epoch_param)
                    epoch = p.load_params(file=param_path)
                    epoch["format"] = "current"
                    epoch["param_path"] = param_path
                    epochs[epoch_name] = epoch

        ep.add_many_to_epoch_directory(epochs, field_name=self.name, mode=mode)

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
        epoch = ep.ImagingEpoch.from_params(name=epoch_name, field=self, instrument=instrument, old_format=old_format)
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
        elif not isinstance(epoch, ep.Epoch):
            old_format = False
            if epoch["format"] == "old":
                old_format = True
            epoch = ep.ImagingEpoch.from_file(epoch, old_format=old_format, field=self)
            self.epochs_imaging_loaded[epoch.name] = epoch
        return epoch

    def select_epoch_spectroscopy(self):
        options = {}
        for epoch in self.epochs_spectroscopy:
            date_string = ""
            if "date" in epoch and epoch["date"] is not None:
                if isinstance(epoch["date"], str):
                    date_string = f" {epoch['date']}"
                else:
                    date_string = f" {epoch['date'].strftime('%Y-%m-%d')}"
            epoch = self.epochs_spectroscopy[epoch]
            options[f"{epoch['name']}\t{date_string}\t{epoch['instrument']}"] = epoch
        for epoch in self.epochs_spectroscopy_loaded:
            epoch = self.epochs_spectroscopy_loaded[epoch]
            options[f'*{epoch.name}\t{epoch.date.isot}\t{epoch.instrument_name}'] = epoch
        options["New epoch"] = "new"
        j, epoch = u.select_option(message="Select epoch.", options=options)
        if epoch == "new":
            epoch = self.new_epoch_spectroscopy()
        elif not isinstance(epoch, ep.Epoch):
            epoch = ep.SpectroscopyEpoch.from_file(epoch, field=self)
            self.epochs_spectroscopy_loaded[epoch.name] = epoch

        return epoch

    def _obj_path(self):
        obj_path = os.path.join(self.param_dir, "objects")
        return obj_path

    def new_object(
            self,
            name: str,
            obj_type: str,
            position: Union[SkyCoord, str, tuple, list, np.ndarray],
            **kwargs
    ):
        objects_path = self._obj_path()
        obj_path = os.path.join(objects_path, f"{name}.yaml")
        obj_type = objects.Object.select_child_class(obj_type=obj_type)
        obj_dict = obj_type.default_params()
        obj_dict["name"] = name
        obj_dict["field"] = self.name
        obj_dict["position"] = astm.attempt_skycoord(position)
        obj_dict.update(kwargs)
        p.save_params(file=obj_path, dictionary=obj_dict)
        return self.add_object_from_dict(obj_dict)

    def new_epoch_imaging(self):
        return self._new_epoch(mode="imaging")

    def new_epoch_spectroscopy(self):
        return self._new_epoch(mode="spectroscopy")

    def _new_epoch(self, mode: str) -> 'Epoch':
        """
        Helper method for generating a new epoch.
        :param mode:
        :return:
        """
        # User selects instrument from those available in param directory, and we set up the relevant Epoch object

        current_epochs = self.gather_epochs_imaging()
        current_epochs.update(self.gather_epochs_spectroscopy())

        instrument = select_instrument(mode=mode)
        is_combined = False
        if mode == "imaging":
            cls = ep.ImagingEpoch.select_child_class(instrument=instrument)
            if len(self.epochs_imaging) > 1:
                is_combined = u.select_yn("Create a pseudo-epoch combining other epochs for maximum depth?")
        elif mode == "spectroscopy":
            cls = ep.SpectroscopyEpoch.select_child_class(instrument=instrument)
        else:
            raise ValueError("mode must be 'imaging' or 'spectroscopy'.")
        new_params = cls.default_params()

        if instrument in surveys:
            new_params["name"] = instrument.upper()
            new_params["date"] = None
            new_params["program_id"] = None
            is_survey = True
        else:
            is_survey = False
            default_prefix = f"{self.name}_{instrument.upper()[instrument.find('-') + 1:]}"
            others_like = list(filter(
                lambda string: string.startswith(default_prefix) and string[-1].isnumeric(),
                current_epochs
            ))
            next_n = 1
            if others_like:
                others_like.sort()
                next_n = int(others_like[-1][-1]) + 1
                print("Other epochs for this instrument are:")
                for st in others_like:
                    print(f"\t", st)
            default = f"{default_prefix}_{next_n}"
            name = None
            while name is None:
                name = u.user_input("Please enter a name for the epoch.", default=default)
                if name in current_epochs:
                    print(f"The epoch {name} already exists.")
                    name = None
            new_params["name"] = name
            # new_params["date"] = u.enter_time(message="Enter UTC observation date, in iso or isot format:").strftime(
            #     '%Y-%m-%d')
            # new_params["program_id"] = input("Enter the programmme ID for the observation:\n")
        new_params["instrument"] = instrument
        new_params["data_path"] = self._epoch_data_path(
            mode=mode,
            instrument=instrument,
            date=new_params["date"],
            epoch_name=new_params["name"],
            survey=is_survey
        )
        new_params["field"] = self.name
        # For a combined epoch, we want to skip the reduction stages and take frames already reduced.
        if is_combined:
            new_params["combined_epoch"] = True
            for stage_name in cls.skip_for_combined:
                new_params["do"][stage_name] = False
        param_path = self._epoch_param_path(mode=mode, instrument=instrument, epoch_name=new_params["name"])

        p.save_params(file=param_path, dictionary=new_params)
        epoch = cls.from_file(param_file=param_path, field=self)

        # Set up a combined epoch by gathering reduced frames from other epochs.
        if is_combined:
            dates = []
            print(f"Gathering reduced frames from all {instrument} epochs")
            # Get list of other epochs
            epochs = self.gather_epochs_imaging()
            frame_type = epoch.frames_for_combined
            this_frame_dict = epoch._get_frames(frame_type=frame_type)
            for other_epoch_name in epochs:
                # Loop over gathered epochs
                other_epoch_dict = epochs[other_epoch_name]
                other_instrument = other_epoch_dict["instrument"]
                # Check if instrument is compatible and that it isn't the same epoch
                if other_instrument.lower() == instrument.lower() and other_epoch_name != epoch.name:
                    # If so, add to internal 'combined_from' list
                    other_epoch = self.epoch_from_params(other_epoch_name, instrument)
                    if other_epoch.date is not None:
                        dates.append(other_epoch.date)
                    epoch.combined_from.append(other_epoch.name)
                    # Get appropriate frame dictionary from other epoch
                    frame_dict = other_epoch._get_frames(frame_type)
                    # Loop through and add to this epoch's dict
                    for fil in frame_dict:
                        if len(frame_dict[fil]) > 0:
                            epoch.check_filter(fil)
                        for frame in frame_dict[fil]:
                            epoch._add_frame(
                                frame,
                                frames_dict=this_frame_dict,
                                frame_type=frame_type
                            )
            epoch.update_output_file()
            epoch.date = Time(np.mean(list(map(lambda d: d.mjd, dates))), format="mjd")

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

    def retrieve_catalogue(
            self,
            cat_name: str,
            force_update: bool = False,
            data_release: int = None
    ):
        """
        Retrieves and saves a catalogue of this field.

        :param cat_name: Name of catalogue; must match one of those available in craftutils.retrieve
        :param force_update: If True, retrieves the catalogue even if one is already on disk.
        :return:
        """

        if isinstance(self.extent, units.Quantity) and self.extent > 0.5 * units.deg:
            radius = self.extent
        else:
            radius = 0.5 * units.deg
        output = self._cat_data_path(cat=cat_name)
        ra = self.centre_coords.ra.value
        dec = self.centre_coords.dec.value

        if force_update or f"in_{cat_name}" not in self.cats or not os.path.isfile(output):
            u.debug_print(2, "Field.retrieve_catalogue(): radius ==", radius)
            response = retrieve.save_catalogue(
                ra=ra,
                dec=dec,
                output=output,
                cat=cat_name.lower(),
                radius=radius,
                data_release=data_release
            )
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

    def load_catalogue(self, cat_name: str, **kwargs):
        if self.retrieve_catalogue(cat_name):
            if cat_name == "gaia":
                if "data_release" in kwargs:
                    data_release = kwargs["data_release"]
                else:
                    data_release = 3
            else:
                data_release = None
            return retrieve.load_catalogue(
                cat_name=cat_name,
                cat=self.get_path(f"cat_csv_{cat_name}"),
                data_release=data_release
            )
        else:
            print("Could not load catalogue; field is outside footprint.")

    def generate_astrometry_indices(
            self,
            cat_name: str = "gaia",
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
        if isinstance(obj, dict):
            self.add_object_from_dict(obj)
        self.objects.append(obj)
        self.objects_dict[obj.name] = obj
        obj.field = self

    def add_object_from_dict(self, obj_dict: dict):
        obj_dict["field"] = self
        obj = objects.Object.from_dict(obj_dict)
        self.add_object(obj=obj)
        return obj

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

    def get_object(self, name: str) -> objects.Object:
        """
        Retrieves the named object from the field's object dictionary.

        :param name: Name of object.
        :return: Requested object.
        """
        if name in self.objects_dict:
            return self.objects_dict[name]
        else:
            raise ValueError(f"No object with name '{name}' found in field '{self.name}'.")

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "type": "Field",
            "centre": objects.position_dictionary.copy(),
            # "objects": [objects.Object.default_params()],
            "extent": 0.3 * units.deg,
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

        if "objects" in param_dict:
            objs = param_dict.pop("objects")
        else:
            objs = None

        if field_type == "Field":
            return cls(
                centre_coords=coord_str,
                objs=objs,
                **param_dict
            )
        elif field_type == "FRBField":
            return FRBField(
                centre_coords=coord_str,
                objs=objs,
                **param_dict
            )
        elif field_type == "StandardField":
            return StandardField(
                centre_coords=coord_str,
                **param_dict
            )

    @classmethod
    def from_params(cls, name, quiet: bool = False):
        if name in active_fields:
            return active_fields[name]
        if not quiet:
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
            tns_name = None
            if field_name[-1].isalpha():
                if u.select_yn(
                        message=f"Is '{field_name}' the TNS name of the FRB?",
                ):
                    tns_name = field_name
                else:
                    tns_name = u.user_input(
                        message=f"Please enter the FRB TNS name, if it has one. Otherwise, leave blank.",
                    )
                    if tns_name in ["", " ", 'None']:
                        tns_name = None
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
            yaml_dict["frb"]["tns_name"] = tns_name

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
            data_path=os.path.join(p.data_dir, name),
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
        super().__init__(
            name=name,
            centre_coords=centre_coords,
            param_path=param_path,
            data_path=data_path,
            objs=objs,
            extent=extent,
            **kwargs
        )

        self.frb = frb
        if self.frb is not None:
            if isinstance(self.frb, str):
                self.frb = self.objects_dict[self.frb]
            if isinstance(self.frb, dict):
                self.frb = objects.FRB.from_dict(self.frb)
            self.frb.field = self
            self.frb.get_host()
            if self.frb.host_galaxy not in self.objects and self.frb.host_galaxy is not None:
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
            projection = red_trimmed.wcs[ext[0]]
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
            self.frb_ellipse_to_plot(ext=ext[0], frb_kwargs=frb_kwargs, img=red_trimmed, ax=ax)

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
            # ticks: int = None, interval: str = 'minmax',
            # font_size: int = 12,
            # reverse_y=False,
            frb_kwargs: dict = {},
            imshow_kwargs: dict = {},
            normalize_kwargs: dict = {},
            output_path: str = None,
            show_legend: bool = False,
            latex_kwargs: dict = {},
            draw_scale_bar: bool = False,
            scale_bar_kwargs: dict = {},
            include_img_err: bool = True,
            **kwargs
    ) -> Tuple[plt.Axes, plt.Figure, dict]:
        """

        :param img:
        :param ext:
        :param fig:
        :param ax:
        :param centre:
        :param show_frb:
        :param frame:
        :param n:
        :param n_x:
        :param n_y:
        :param frb_kwargs:
        :param imshow_kwargs:
        :param normalize_kwargs:
        :param output_path:
        :param show_legend:
        :param latex_kwargs:
        :param kwargs:
        :return: ax, figure
        """
        pl.latex_setup(**latex_kwargs)
        if not isinstance(self.frb, objects.FRB):
            raise TypeError("self.frb has not been set properly for this FRBField.")
        if centre is None:
            centre = self.frb.host_galaxy.position

        if draw_scale_bar:
            kwargs["scale_bar_object"] = self.frb.host_galaxy

        ax, fig, other_args = img.plot_subimage(
            centre=centre,
            frame=frame,
            ext=ext,
            fig=fig,
            ax=ax,
            n=n, n_x=n_x, n_y=n_y,
            imshow_kwargs=imshow_kwargs,
            normalize_kwargs=normalize_kwargs,
            scale_bar_kwargs=scale_bar_kwargs,
            obj=self.frb.host_galaxy,
            **kwargs
        )

        if show_frb:
            self.frb_ellipse_to_plot(
                ext=ext,
                frb_kwargs=frb_kwargs,
                img=img,
                ax=ax,
                include_img_err=include_img_err
            )
            if show_legend:
                ax.legend()

        if output_path is not None:
            fig.savefig(output_path)

        return ax, fig, other_args

    def frb_ellipse_to_plot(
            self,
            ax,
            img: image.ImagingImage,
            ext: int = 0,
            colour: str = None,
            frb_kwargs: dict = {},
            plot_centre: bool = False,
            include_img_err: bool = True
    ):
        from matplotlib.patches import Ellipse
        img.load_headers()
        frb = self.frb.position
        uncertainty = self.frb.position_err
        a, b = uncertainty.uncertainty_quadrature()
        if a == 0 * units.arcsec or b == 0 * units.arcsec:
            a, b = uncertainty.uncertainty_quadrature_equ()
        theta = uncertainty.theta.to(units.deg)
        rotation_angle = img.extract_rotation_angle(ext=ext)
        theta = theta - rotation_angle
        img.extract_pixel_scale()

        if "include_img_err" in frb_kwargs:
            include_img_err = frb_kwargs.pop("include_img_err")
        if "edgecolor" not in frb_kwargs:
            if colour is None:
                colour = "white"
            frb_kwargs["edgecolor"] = colour
        if "facecolor" not in frb_kwargs:
            frb_kwargs["facecolor"] = "none"
        img_err = None
        if include_img_err:
            img_err = img.extract_astrometry_err()
        if img_err is not None:
            a = np.sqrt(a ** 2 + img_err ** 2)
            b = np.sqrt(b ** 2 + img_err ** 2)
        ax = img.plot_ellipse(
            ax=ax,
            coord=frb,
            a=a, b=b,
            theta=theta,
            plot_centre=plot_centre,
            centre_kwargs=dict(
                c=frb_kwargs["edgecolor"],
                marker="x"
            ),
            **frb_kwargs,
        )

        return ax

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
        param_dict["frb"]["host_galaxy"] = objects.Galaxy.default_params()
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
