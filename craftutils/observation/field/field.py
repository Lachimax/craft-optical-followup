# Code by Lachlan Marnoch, 2021 - 2024
import os
import warnings
import shutil
from typing import Union, List

import numpy as np

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as units
import astropy.table as table
import astropy.io.fits as fits

import craftutils.observation.objects as objects
import craftutils.observation.image as image
import craftutils.observation.instrument as inst
import craftutils.observation.epoch as ep
import craftutils.observation.survey as survey
import craftutils.observation.filters as filters
import craftutils.astrometry as astm
import craftutils.params as p
import craftutils.retrieve as retrieve
import craftutils.utils as u
from craftutils.observation.pipeline import Pipeline

dm_units = units.parsec * units.cm ** -3

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
    field_list = list(active_fields.keys())
    for field_name in field_list:
        del active_fields[field_name]


@u.export
def select_instrument(mode: str):
    if mode == "imaging":
        options = instruments_imaging
    elif mode == "spectroscopy":
        options = instruments_spectroscopy
    else:
        raise ValueError("Mode must be 'imaging' or 'spectroscopy'.")
    _, instrument = u.select_option("Select an instrument:", options=options)
    return instrument


@u.export
def list_fields(include_std: bool = False):
    print("Searching for field param files...")
    param_path = os.path.join(config['param_dir'], 'fields')
    fields = list(filter(lambda d: os.path.isdir(os.path.join(param_path, d)) and os.path.isfile(
        os.path.join(param_path, d, f"{d}.yaml")), os.listdir(param_path)))
    if not include_std:
        fields = list(filter(lambda f: "STD" not in f, fields))
    fields.sort()
    return fields


@u.export
class Field(Pipeline):
    stage_output_dirs = False

    def __init__(
            self,
            centre_coords: Union[SkyCoord, str] = None,
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

        super().__init__(
            **kwargs
        )
        self.mkdir()

        # Input attributes

        self.objects = {}

        if centre_coords is None:
            if objs is not None:
                centre_coords = objs[0].coords
        if centre_coords is not None:
            self.centre_coords = astm.attempt_skycoord(centre_coords)

        self.param_dir = None
        if self.param_path is not None:
            self.param_dir = os.path.split(self.param_path)[0]
        self.mkdir_params()

        # Derived attributes

        self.epochs_spectroscopy = {}
        self.epochs_spectroscopy_loaded = {}
        self.epochs_imaging = {}
        self.epochs_imaging_loaded = {}
        self.epochs_loaded = {}

        self.imaging = {}

        self.paths = {}

        self.cats = {}
        self.cat_gaia = None

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

    @classmethod
    def stages(cls):
        return {
            "finalise_imaging": {
                "method": cls.proc_finalise_imaging,
                "message": "Finalise imaging from all epochs?"
            },
            "update_photometry": {
                "method": cls.proc_update_photometry,
                "message": "Pull photometry from all epochs?"
            },
            "refine_photometry": {
                "method": cls.proc_refine_photometry,
                "message": "Do matched photometry?"
            },
            "galfit": {
                "method": cls.proc_galfit,
                "message": "Perform basic GALFIT on best images?",
                "keywords": {
                    "galfit_img": None,
                    "force": False
                }
            },
            "send_to_table": {
                "method": cls.proc_push_to_table,
                "message": "Send photometry and other properties to data tables?"
            }

        }

    def proc_finalise_imaging(self, output_dir: str, **kwargs):
        # def fin_fil(name: str):
        #     return "validation" not in name
        self.force_stage_all_epochs(
            stage="finalise",
            exclude_validation=True,
            **kwargs
        )

    def proc_update_photometry(self, output_dir: str, **kwargs):
        self.force_stage_all_epochs(
            stage="get_photometry",
            skip_plots=True,
            skip_path=True,
            **kwargs
        )

    def force_stage_all_epochs(self, stage: str, exclude_validation: bool = True, **kwargs):
        epochs = self.gather_epochs_imaging(exclude_validation=exclude_validation)
        for epoch_name in epochs:
            epoch = ep.epoch_from_directory(epoch_name)
            epoch.do_runtime = [stage]
            if stage not in epoch.param_file:
                epoch.param_file[stage] = {}
            epoch.param_file[stage].update(kwargs)
            # Run only this stage of each epoch pipeline
            print()
            print(f"RUNNING {stage} FOR EPOCH", epoch_name)
            print(epoch.param_file[stage])
            print("=" * 30)

            epoch.pipeline(skip_cats=True)

    def proc_refine_photometry(self, output_dir: str, **kwargs):
        object_list: List[objects.Object] = list(self.objects.values())
        object_list.sort(key=lambda o: o.name, reverse=False)
        for obj in object_list:
            if not obj.optical:
                continue
            obj.load_output_file()
            obj.update_output_file()
            obj.get_good_photometry()
            # obj.push_to_table(select=True)
            obj.write_plot_photometry()
            obj.update_output_file()
        self.generate_cigale_photometry()

    def proc_galfit(self, output_dir: str, **kwargs):
        if "galfit_img" in kwargs and kwargs["galfit_img"] is not None:
            kwargs["use_img"] = kwargs["galfit_img"]
        self.galfit(**kwargs)

    def get_images_band(
            self,
            fil: Union[str, filters.Filter],
            instrument: Union[str, inst.Instrument] = None
    ):
        if isinstance(fil, str):
            if isinstance(instrument, inst.Instrument):
                instrument = instrument.name
            elif not isinstance(instrument, str):
                raise TypeError(
                    f"If fil is provided as a string, instrument must also be provided as str or Instrument, not {type(instrument)}")
            fil = filters.Filter.from_params(
                instrument_name=instrument,
                filter_name=fil
            )

        return list(
            filter(
                lambda d: d["filter"] == fil.name and d["instrument"] == fil.instrument.name,
                self.imaging.values()
            )
        )

    def get_filters(self, instrument: str = None):
        self.load_imaging()
        # Build a list of (filters, instruments) using all available epochs
        all_filters = list(map(lambda i: (i["filter"], i["instrument"]), self.imaging.values()))
        # Use a set to cull it to unique values
        fil_list = list(set(all_filters))
        # Filter (no pun intended) out unwanted instruments
        if instrument is not None:
            fil_list = list(filter(lambda f, i: i["instrument"] == instrument, fil_list))
        # Build a list of Filter objects from the strings
        fil_list_2 = []
        for fil, instr in fil_list:
            fil = filters.Filter.from_params(filter_name=fil, instrument_name=instr)
            fil_list_2.append(fil)
        return fil_list_2

    def deepest_in_band(
            self,
            fil: Union[str, filters.Filter],
            instrument: Union[str, inst.Instrument] = None
    ) -> dict:
        img_list = self.get_images_band(fil, instrument=instrument)
        img_list.sort(key=lambda d: d["depth"])
        img_dict = img_list[-1]
        return img_dict

    def galfit(
            self,
            apply_filter=None,
            use_img: str = None,
            use_subbed: bool = False,
            **kwargs
    ):
        from craftutils.observation.objects import Galaxy
        if "force" in kwargs:
            force = kwargs["force"]
        else:
            force = False
        if apply_filter is None:
            obj_list = list(self.objects.values())
        else:
            obj_list = list(filter(apply_filter, self.objects.values()))

        obj_list += list(filter(lambda o: isinstance(o, Galaxy) and o.do_galfit, self.objects.values()))

        filter_list = self.load_imaging(instrument="vlt-fors2")
        if not filter_list:
            filter_list = self.load_imaging()
        print(filter_list)
        print("use_img", use_img)

        for obj in obj_list:
            print(obj.name, type(obj), obj.do_galfit)
            if isinstance(obj, objects.Galaxy) and obj.do_galfit:
                obj.load_output_file()

                if use_img is None:
                    phot = obj.select_deepest_sep()
                    best_img_path = phot["good_image_path"]
                    best_img = image.Image.from_fits(best_img_path)
                    # print("You shouldn't be getting here right now.")
                else:
                    print(self.imaging[use_img])
                    best_img = self.imaging[use_img]["image"]

                img_dict = {}

                for fil in filter_list:
                    print()
                    print("=" * 100)
                    print("best_img", best_img.name)
                    img = self.deepest_in_band(fil=fil)["image"]
                    print("img", img.name)
                    img_dict[fil.name] = img

                best_fil = best_img.filter
                img_dict[best_fil.name] = best_img

                for fil, img in img_dict.items():

                    print(f"Doing GALFIT with image {img.name}")

                    param_guesses, kwargs = obj.galfit_guess_dict(img=img)
                    print(param_guesses)

                    # TODO: REMOVE
                    # old_dir = os.path.join(obj.data_path, "GALFIT")
                    # n_files = sum([os.path.isfile(os.path.join(old_dir, f)) for f in os.listdir(old_dir)])
                    # if n_files > 0:
                    #     shutil.rmtree(old_dir)

                    if obj.galfit_models and not force:
                        print("Found existing models; skipping actual GALFIT run.")
                        if fil in obj.galfit_models:
                            results = obj.galfit_models[fil]
                        else:
                            results = None
                    else:
                        results = img.galfit_object(
                            obj=obj,
                            model_guesses=[param_guesses],
                            output_prefix=obj.name,
                            **kwargs
                        )
                    print(img.name, best_img.name)
                    print(img.path, "\n", best_img.path)
                    if img.name == best_img.name and results is not None:
                        obj.galfit_models["best"] = results
                    print("=" * 100)
                    print()

            obj.update_output_file()

    def proc_push_to_table(self, output_dir: str, **kwargs):
        object_list: List[objects.Object] = list(self.objects.values())
        for obj in object_list:
            obj.load_output_file()
            print("Tabulating", obj.name)
            obj.load_output_file()
            row = obj.push_to_table(select=True)
            obj.update_output_file()

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
                filter(lambda f: f.endswith(".yaml") and not f.endswith("backup.yaml") and not f == "None.yaml", os.listdir(obj_path)))
            obj_params.sort()
            for obj_param in obj_params:
                obj_name = obj_param[:obj_param.find(".yaml")]
                param_path = os.path.join(obj_path, obj_param)
                obj_dict = p.load_params(file=param_path)
                obj_dict["param_path"] = param_path
                if "name" not in obj_dict:
                    obj_dict["name"] = obj_name
                self.add_object_from_dict(obj_dict=obj_dict)

    def _gather_epochs(
            self,
            mode: str = "imaging",
            quiet: bool = False,
            instrument: str = None,
            exclude_validation: bool = False
    ):
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

            if isinstance(instrument, str):
                instruments = [instrument]
            else:
                instruments = filter(lambda d: os.path.isdir(os.path.join(mode_path, d)), os.listdir(mode_path))

            for instrument in instruments:
                instrument_path = os.path.join(mode_path, instrument)
                if not quiet:
                    print(f"Looking in {instrument_path}")
                if os.path.isdir(instrument_path):
                    epoch_params = list(
                        filter(
                            lambda f: f.endswith(".yaml") and not f.endswith("backup.yaml"),
                            os.listdir(instrument_path)
                        )
                    )
                else:
                    epoch_params = []
                epoch_params.sort()
                for epoch_param in epoch_params:
                    epoch_name = epoch_param[:epoch_param.find(".yaml")]
                    param_path = os.path.join(instrument_path, epoch_param)
                    epoch = p.load_params(file=param_path)
                    epoch["format"] = "current"
                    epoch["param_path"] = param_path
                    is_validation = False
                    if "validation_copy_of" in epoch:
                        is_validation = isinstance(epoch["validation_copy_of"], str)
                    if not is_validation or not exclude_validation:
                        epochs[epoch_name] = epoch

        ep.add_many_to_epoch_directory(epochs, field_name=self.name, mode=mode)

        return epochs

    def gather_epochs_spectroscopy(self, quiet: bool = False, instrument: str = None):
        """
        Gathers all of the spectroscopy observation epochs of this field.
        :return: Dict, with keys being the epoch names and values being nested dictionaries containing the same
        information as the epoch .yaml files.
        """
        epochs = self._gather_epochs(mode="spectroscopy", quiet=quiet, instrument=instrument)
        self.epochs_spectroscopy.update(epochs)
        return epochs

    def gather_epochs_imaging(
            self,
            quiet: bool = False,
            instrument: str = None,
            exclude_validation: bool = False
    ):
        """
        Gathers all of the imaging observation epochs of this field.
        :return: Dict, with keys being the epoch names and values being nested dictionaries containing the same
        information as the epoch .yaml files.
        """
        epochs = self._gather_epochs(
            mode="imaging",
            quiet=quiet,
            instrument=instrument,
            exclude_validation=exclude_validation
        )
        self.epochs_imaging.update(epochs)
        return epochs

    def epoch_from_params(self, epoch_name: str, instrument: str):
        epoch = ep.ImagingEpoch.from_params(name=epoch_name, field=self, instrument=instrument)
        self.epochs_imaging[epoch_name] = epoch
        return epoch

    def select_epoch(self, mode: str, instrument: str = None, allow_new: bool = True):
        options = {}
        epoch_dict = {
            "imaging": self.epochs_imaging,
            "spectroscopy": self.epochs_spectroscopy
        }[mode]
        loaded_dict = {
            "imaging": self.epochs_imaging_loaded,
            "spectroscopy": self.epochs_spectroscopy_loaded
        }[mode]
        for epoch_name, epoch in epoch_dict.items():
            if isinstance(epoch["date"], str):
                date_string = f" {epoch['date']}"
            elif isinstance(epoch["date"], Time):
                date_string = f" {epoch['date'].strftime('%Y-%m-%d')}"
            else:
                date_string = "--"
            options[f'{epoch["name"]}\t{date_string}\t{epoch["instrument"]}'] = epoch
        for epoch in loaded_dict:
            # If epoch is already instantiated.
            if isinstance(instrument, str) and epoch.instrument_name != instrument:
                continue
            epoch = loaded_dict[epoch]
            options[f'*{epoch.name}\t{epoch.date.isot}\t{epoch.instrument_name}'] = epoch
        if allow_new:
            options["New epoch"] = "new"
        j, epoch = u.select_option(message="Select epoch.", options=options, sort=True)
        if epoch == "new":
            epoch = self._new_epoch(instrument=instrument, mode=mode)
        elif not isinstance(epoch, ep.Epoch):
            epoch = ep.ImagingEpoch.from_file(epoch, field=self)
            loaded_dict[epoch.name] = epoch
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

    def new_epoch_imaging(self, instrument: str = None):
        return self._new_epoch(mode="imaging", instrument=instrument)

    def new_epoch_spectroscopy(self, instrument: str = None):
        return self._new_epoch(mode="spectroscopy", instrument=instrument)

    def _new_epoch(self, mode: str, instrument: str = None) -> 'Epoch':
        """Helper method for generating a new epoch.

        :param mode:
        :return:
        """
        # User selects instrument from those available in param directory, and we set up the relevant Epoch object
        if not isinstance(instrument, str):
            instrument = select_instrument(mode=mode)

        current_epochs = self.gather_epochs_imaging(instrument=instrument)
        current_epochs.update(self.gather_epochs_spectroscopy(instrument=instrument))

        copy_of_epoch = None
        is_combined = False
        is_validation = False
        if mode == "imaging":
            cls = ep.ImagingEpoch.select_child_class(instrument=instrument)
            if len(self.epochs_imaging) > 1:
                is_combined = u.select_yn("Create a pseudo-epoch combining other epochs for maximum depth?",
                                          default=False)
            if not is_combined:
                is_validation = u.select_yn("Make a copy of another epoch to perform validation_checks checks?",
                                            default=False)
                if is_validation:
                    print("Which epoch would you like to use?")
                    copy_of_epoch = self.select_epoch(mode=mode, instrument=instrument, allow_new=False)
        elif mode == "spectroscopy":
            cls = ep.SpectroscopyEpoch.select_child_class(instrument=instrument)
        else:
            raise ValueError("mode must be 'imaging' or 'spectroscopy'.")

        if is_validation:
            new_params = p.load_params(os.path.join(self.param_dir, mode, instrument, copy_of_epoch.name + ".yaml"))
        else:
            new_params = cls.default_params()

        if instrument in surveys:
            new_params["name"] = instrument.upper()
            new_params["date"] = None
            new_params["program_id"] = None
            is_survey = True
        else:
            is_survey = False
            default_prefix = f"{self.name}_{instrument.upper()[instrument.find('-') + 1:]}"
            if is_combined:
                default = default_prefix + "_combined"
            elif is_validation:
                default = copy_of_epoch.name + "_validation"
            else:
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
        if isinstance(copy_of_epoch, ep.ImagingEpoch):
            new_params["validation_copy_of"] = copy_of_epoch.name
        new_params["data_path"] = self._epoch_data_path(
            mode=mode,
            instrument=instrument,
            date=new_params["date"],
            epoch_name=new_params["name"],
            survey=is_survey
        )
        new_params["field"] = self.name
        # For a combined epoch, we want to skip the reduction stages and take frames already reduced.
        new_params["combined_epoch"] = is_combined
        if is_combined or is_validation:
            for stage_name in cls.skip_for_combined:
                new_params["do"][stage_name] = False
        if is_validation:
            for stage_name in cls.validation_stages:
                new_params["do"][stage_name] = True
                new_params["source_extraction"]["do_astrometry_diagnostics"] = False
                new_params["source_extraction"]["do_psf_diagnostics"] = False
        param_path = self._epoch_param_path(mode=mode, instrument=instrument, epoch_name=new_params["name"])

        p.save_params(file=param_path, dictionary=new_params)
        epoch = cls.from_file(param_file=param_path, field=self)

        # Set up a combined epoch by gathering reduced frames from other epochs.
        if is_combined:
            dates = []
            frame_type = epoch.frames_for_combined
            print(f"Gathering {frame_type} frames from all {instrument} epochs")
            # Get list of other epochs
            epochs = self.gather_epochs_imaging(exclude_validation=True)
            this_frame_dict = epoch._get_frames(frame_type=frame_type)
            for other_epoch_name in epochs:
                if other_epoch_name == epoch.name:
                    continue
                print(f"\tChecking {other_epoch_name}...")
                # Loop over gathered epochs
                other_epoch_dict = epochs[other_epoch_name]
                other_instrument = other_epoch_dict["instrument"]
                # Check if instrument is compatible and that it isn't the same epoch
                print(f"\t\tInstrument is {other_instrument.lower()}.")
                if other_instrument.lower() == instrument.lower():
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
            print(f"New epoch is combined from: {epoch.combined_from}")
            print(f"With filters: {epoch.filters}")
            epoch.set_date(Time(np.mean(list(map(lambda d: d.mjd, dates))), format="mjd"))
            epoch.update_output_file()

        elif is_validation:
            dates = []
            frame_type = epoch.frames_for_combined
            print(f"Gathering {frame_type} frames from {copy_of_epoch.name}")
            # Get list of other epochs
            this_frame_dict = epoch._get_frames(frame_type=frame_type)
            print(f"\tChecking {copy_of_epoch.name}...")
            # If so, add to internal 'combined_from' list
            if copy_of_epoch.date is not None:
                dates.append(copy_of_epoch.date)
            # Get appropriate frame dictionary from other epoch
            frame_dict = copy_of_epoch._get_frames(frame_type)
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

    def _cat_data_path(self, cat: str, data_release: int = None):
        if self.data_path is not None:
            filename = f"{cat}_{self.name}.csv"
            if data_release is not None:
                filename.replace(".csv", f"_dr{data_release}.csv")
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
                name_str = f"{date.strftime('%Y-%m-%d')}-{epoch_name}"
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
        output = self._cat_data_path(cat=cat_name, data_release=data_release)
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
        if self.retrieve_catalogue(cat_name, **kwargs):
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
            id_prefix = astm.jname(coord=self.centre_coords)[1:].replace("+", "").replace("-", "").replace(".", "")
            astm.generate_astrometry_indices(
                cat_name=cat_name,
                cat=cat_path,
                fits_cat_output=cat_path.replace(".csv", ".fits"),
                output_file_prefix=prefix,
                index_output_dir=cat_index_path,
                unique_id_prefix=int(id_prefix)
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

    def load_output_file(self, **kwargs):
        outputs = p.load_output_file(self)
        if outputs is not None:
            if "cats" in outputs:
                self.cats.update(outputs["cats"])
            if "imaging" in outputs:
                self.imaging.update(outputs["imaging"])
        return outputs

    def load_imaging(self, instrument: str = None):
        filter_list = []
        for img_name, img_dict in self.imaging.items():
            if instrument is not None and instrument != img_dict["instrument"]:
                continue
            cls = image.CoaddedImage.select_child_class(
                instrument_name=img_dict["instrument"]
            )
            img = image.from_path(path=img_dict["path"], cls=cls)
            img_dict["image"] = img
            img.extract_filter()
            filter_list.append(img.filter)
        filter_list = list(set(filter_list))
        return filter_list

    def _output_dict(self):
        imaging = self.imaging.copy()
        for img_dict in imaging.values():
            if "image" in img_dict:
                img_dict.pop("image")

        output_dict = super()._output_dict()
        output_dict.update({
            "cats": self.cats,
            "imaging": imaging
        })
        return output_dict

    def add_image(
            self,
            img: image.ImagingImage,
            epoch_name: str = None
    ):
        img.extract_filter()
        img.filter.load_instrument()
        fil = img.filter
        fil_name = img.filter.machine_name()
        depth, _ = img.select_depth()
        self.imaging[img.name] = {
            "path": img.path,
            "depth": depth,
            "instrument": fil.instrument.name,
            "filter": fil.name,
            "name": img.name,
            "epoch": epoch_name
        }

    def add_object(self, obj: Union[dict, objects.Object]):
        if isinstance(obj, dict):
            self.add_object_from_dict(obj)
        elif obj.name in self.objects and self.objects[obj.name] is not obj:
            warnings.warn(
                f"An object with name {obj.name}, z={self.objects[obj.name].z} already exists in field {self.name}; it is being overwritten (z={obj.z}).")
        self.objects[obj.name] = obj
        obj.field = self
        return obj

    def remove_object(self, obj: Union[objects.Object, str]):
        if isinstance(obj, str):
            name = obj
        elif isinstance(obj, objects.Object):
            name = obj.name
        else:
            name = None
        if name in self.objects:
            obj = self.objects.pop(name)

    def add_object_from_dict(self, obj_dict: dict):
        obj_dict["field"] = self
        obj = objects.Object.from_dict(obj_dict)
        self.add_object(obj=obj)
        return obj

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
                if tbl_this is None:
                    continue
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
        if photometries:
            tbl_cigale = table.QTable(photometries)
            tbl_cigale.write(
                os.path.join(self.data_path, f"{self.name}_cigale.csv"),
                overwrite=True
            )
            return tbl_cigale
        else:
            return None

    def unpack_cigale_results(
            self,
            cigale_dir: str
    ):
        results = fits.open(os.path.join(cigale_dir, "results.fits"))
        results_tbl = table.QTable(results[1].data)

        for i, obj_name in enumerate(results_tbl["id"]):
            if obj_name in self.objects:
                model_path = os.path.join(cigale_dir, f"{obj_name}_best_model.fits")
                sfh_path = os.path.join(cigale_dir, f"{obj_name}_SFH.fits")
                obj = self.objects[obj_name]
                obj.load_output_file()
                obj.cigale_results = p.sanitise_yaml_dict(dict(results_tbl[i]))
                obj.log_mass_stellar = np.log10(obj.cigale_results["bayes.stellar.m_star"])
                obj.log_mass_stellar_err_plus = obj.log_mass_stellar_err_minus = u.uncertainty_log10(
                    arg=obj.cigale_results["bayes.stellar.m_star"],
                    uncertainty_arg=obj.cigale_results["bayes.stellar.m_star_err"]
                )
                obj.sfr = obj.cigale_results["bayes.sfh.sfr"] * units.solMass / units.year
                obj.sfr_err = obj.cigale_results["bayes.sfh.sfr_err"] * units.solMass / units.year
                if os.path.isfile(model_path):
                    obj.cigale_model_path = model_path
                if os.path.isfile(sfh_path):
                    obj.cigale_sfh_path = sfh_path
                obj.update_output_file()

    def load_all_objects(self):
        for obj in self.objects.values():
            obj.load_output_file()

    def get_object(self, name: str, allow_missing: bool) -> Union[objects.Object, None]:
        """
        Retrieves the named object from the field's object dictionary.

        :param name: Name of object.
        :param allow_missing: return None for a missing object instead of throwing an error.
        :return: Requested object.
        """
        if name in self.objects:
            return self.objects[name]
        elif not allow_missing:
            raise ValueError(f"No object with name '{name}' found in field '{self.name}'.")
        else:
            return None

    def object_table(self, tbl_type: str = None, **kwargs):
        self.load_all_objects()
        rows = []
        for name, obj in self.objects.items():
            row_dict, type_ = obj.assemble_row(**kwargs)
            if tbl_type is None or tbl_type == type_:
                rows.append(row_dict)
        return table.QTable(rows)

    @classmethod
    def default_params(cls):
        from craftutils.observation.objects.position import position_dictionary
        default_params = {
            "name": None,
            "type": "Field",
            "centre": position_dictionary.copy(),
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
            from .frb import FRBField
            return FRBField(
                centre_coords=coord_str,
                objs=objs,
                **param_dict
            )
        elif field_type == "StandardField":
            from .std import StandardField
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
        path = cls.build_param_path(field_name=name, mkdir=False)
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
    def build_param_path(cls, field_name: str, mkdir: bool = True):
        path = os.path.join(p.param_dir, "fields", field_name)
        if mkdir:
            os.makedirs(path, exist_ok=True)
        return os.path.join(path, f"{field_name}.yaml")

    @classmethod
    def new_params_from_input(cls, field_name: str, field_param_path: str):
        from .frb import FRBField
        from .std import StandardField
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
        object_param_path_yaml = os.path.join(field_param_path, f"objects")
        u.mkdir_check(object_param_path_yaml)

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
            if tns_name is None:
                tns_name = u.user_input(
                    message=f"Please enter the FRB TNS name, if it has one. Otherwise, leave blank.",
                )
                if tns_name in ["", " ", 'None']:
                    tns_name = None

            dm = u.user_input(
                "If you know the burst DM, please enter that now in units of pc / cm^3. Otherwise, leave blank.",
                input_type=float
            )
            if dm in ["", " ", 'None']:
                dm = 0 * dm_units
            else:
                dm *= dm_units

            date = u.user_input(
                "If you have a precise FRB arrival time, please enter that now; otherwise, leave blank."
            )
            if date in ["", " ", 'None']:
                date = objects.FRB._date_from_name(field_name)

            frb_dict = objects.FRB.default_params()
            host_dict = objects.FRB.default_host_params(
                frb_name=field_name,
                position=position
            )
            yaml_dict["frb"] = field_name

            frb_dict["name"] = field_name
            frb_dict["date"] = date
            frb_dict["dm"] = dm
            frb_dict["position"] = position
            frb_dict["position_err"]["alpha"]["total"] = float(ra_err)
            frb_dict["position_err"]["delta"]["total"] = float(dec_err)
            frb_dict["tns_name"] = tns_name
            frb_dict["field"] = field_name

            host_dict["field"] = field_name
            host_name = host_dict["name"]

            p.save_params(field_param_path_yaml, yaml_dict)
            p.save_params(os.path.join(object_param_path_yaml, f"{field_name}.yaml"), frb_dict)
            p.save_params(os.path.join(object_param_path_yaml, f"{host_name}.yaml"), host_dict)

        print(f"Template parameter file created at '{field_param_path_yaml}'")
        input("Please edit this file before proceeding, then press Enter to continue.")
