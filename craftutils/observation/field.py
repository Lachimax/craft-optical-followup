import os.path
import warnings
from typing import Union, List

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as units

import craftutils.astrometry as a
import craftutils.utils as u
import craftutils.params as p
import craftutils.retrieve as retrieve
import craftutils.spectroscopy as spec
import craftutils.observation.objects as objects
import craftutils.observation.image as image
import craftutils.fits_files as ff

config = p.config

instruments_imaging = ["vlt-fors2", "vlt-xshooter", "mgb-imacs"]
instruments_spectroscopy = ["vlt-fors2", "vlt-xshooter"]


def select_instrument(mode: str):
    if mode == "imaging":
        options = instruments_imaging
    elif mode == "spectroscopy":
        options = instruments_spectroscopy
    else:
        raise ValueError("Mode must be 'imaging' or 'spectroscopy'.")
    _, instrument = u.select_option("Select an instrument:", options=options)
    return instrument


def list_fields():
    print("Searching for field param files...")
    param_path = os.path.join(config['param_dir'], 'fields')
    print(param_path)
    fields = list(filter(lambda d: os.path.isdir(os.path.join(param_path, d)) and os.path.isfile(
        os.path.join(param_path, d, f"{d}.yaml")), os.listdir(param_path)))
    fields.sort()
    return fields


def list_fields_old():
    print("Searching for old-format field param files...")
    param_path = os.path.join(config['param_dir'], 'FRBs')
    fields = filter(lambda d: os.path.isfile(os.path.join(param_path, d)) and d.endswith('.yaml'),
                    os.listdir(param_path))
    fields = list(map(lambda f: f.split(".")[0], fields))
    fields.sort()
    return fields


def _params_init(param_file: Union[str, dict]):
    if type(param_file) is str:
        # Load params from .yaml at path.
        param_file = u.sanitise_file_ext(filename=param_file, ext="yaml")
        param_dict = p.load_params(file=param_file)
        if param_dict is None:
            return None, None, None  # raise FileNotFoundError(f"No parameter file found at {param_file}.")
        name = u.get_filename(path=param_file, include_ext=False)
        param_dict["param_path"] = param_file
    else:
        param_dict = param_file
        name = param_dict["name"]
        param_file = param_dict["param_path"]

    return name, param_file, param_dict


def _retrieve_eso_epoch(epoch: Union['ESOImagingEpoch', 'ESOSpectroscopyEpoch'], path: str):
    if epoch.program_id is None:
        epoch.set_program_id(input("A program ID is required to retrieve ESO data. Enter here:\n"))
    if epoch.date is None:
        epoch.set_date(Time(
            input("An observation date is required to retrieve ESO data. Enter here, in iso or isot format:\n")))
    if isinstance(epoch, ESOImagingEpoch):
        mode = "imaging"
    elif isinstance(epoch, ESOSpectroscopyEpoch):
        mode = "spectroscopy"
    else:
        raise TypeError("epoch must be either an ESOImagingEpoch or an ESOSpectroscopyEpoch.")
    if epoch.target is None:
        obj = u.user_input(
            "Specifying an object might help find the correct observation. Enter here, as it appears in the archive OBJECT field, or leave blank:\n")
        if obj not in ["", " "]:
            epoch.set_target(obj)

    u.mkdir_check(path)
    instrument = epoch.instrument.split('-')[-1]
    r = retrieve.save_eso_raw_data_and_calibs(output=path, date_obs=epoch.date,
                                              program_id=epoch.program_id, instrument=instrument, mode=mode,
                                              obj=epoch.target)
    if r:
        os.system(f"uncompress {path}/*.Z -f")
    return r


def _check_do_list(do: Union[list, str]):
    if type(do) is str:
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
    return do


class Field:
    def __init__(self,
                 name: str = None,
                 centre_coords: Union[SkyCoord, str] = None,
                 param_path: str = None,
                 data_path: str = None,
                 objs: Union[List[objects.Object], dict] = None
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

        if objs is dict:
            obj_list = []
            for name in objs:
                obj = objects.Object.from_dict(objs[name])
                obj_list.append(obj)
            objs = obj_list

        self.objects = objs

        if centre_coords is None:
            if objs is not None:
                centre_coords = objs[0].coords
        if centre_coords is not None:
            self.centre_coords = a.attempt_skycoord(centre_coords)

        self.name = name
        self.param_path = param_path
        self.param_dir = None
        if self.param_path is not None:
            self.param_dir = os.path.split(self.param_path)[0]
        self.mkdir_params()
        self.data_path = data_path
        self.mkdir()

        # Derived attributes

        self.epochs_spectroscopy = {}
        self.epochs_spectroscopy_loaded = {}
        self.epochs_imaging = {}
        self.epochs_imaging_loaded = {}

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return self.__str__()

    def mkdir(self):
        if self.data_path is not None:
            u.mkdir_check(self.data_path)

    def mkdir_params(self):
        if self.param_dir is not None:
            u.mkdir_check(os.path.join(self.param_dir, "spectroscopy"))
            u.mkdir_check(os.path.join(self.param_dir, "imaging"))
        else:
            warnings.warn(f"param_dir is not set for this {type(self)}.")

    def _gather_epochs(self, mode: str = "imaging"):
        print(f"Searching for {mode} epoch param files...")
        epochs = {}
        if self.param_dir is not None:
            mode_path = os.path.join(self.param_dir, mode)
            print(f"Looking in {mode_path}")
            for instrument in filter(lambda d: os.path.isdir(os.path.join(mode_path, d)), os.listdir(mode_path)):
                instrument_path = os.path.join(mode_path, instrument)
                print(f"Looking in {instrument_path}")
                for epoch_param in filter(lambda f: f.endswith(".yaml"), os.listdir(instrument_path)):
                    epoch_name = epoch_param[:epoch_param.find(".yaml")]
                    param_path = os.path.join(instrument_path, epoch_param)
                    epoch = p.load_params(file=param_path)
                    epoch["format"] = "current"
                    epoch["param_path"] = param_path
                    epochs[epoch_name] = epoch

        return epochs

    def gather_epochs_spectroscopy(self):
        epochs = self._gather_epochs(mode="spectroscopy")
        self.epochs_spectroscopy.update(epochs)
        return epochs

    def gather_epochs_imaging(self):
        epochs = self._gather_epochs(mode="imaging")
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
                date_string = f" {epoch['date'].to_datetime().date()}"
            options[f'{epoch["name"]}\t{date_string}\t{epoch["instrument"]}'] = epoch
        for epoch in self.epochs_imaging_loaded:
            # If epoch is already instantiated.
            epoch = self.epochs_spectroscopy_loaded[epoch]
            options[f'*{epoch.name}\t{epoch.date.isot}\t{epoch.instrument}'] = epoch
        options["New epoch"] = "new"
        j, epoch = u.select_option(message="Select epoch.", options=options)
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
            options[f'*{epoch.name}\t{epoch.date.isot}\t{epoch.instrument}'] = epoch
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
        new_params["name"] = u.user_input("Please enter a name for the epoch.")
        new_params["instrument"] = instrument
        new_params["date"] = u.enter_time(message="Enter UTC observation date, in iso or isot format:")
        new_params["program_id"] = input("Enter the programmme ID for the observation:\n")
        new_params["data_path"] = self._epoch_data_path(mode=mode, instrument=instrument, date=new_params["date"],
                                                        epoch_name=new_params["name"])
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
            path = os.path.join(self.data_path, mode)
            u.mkdir_check(path)
            return path
        else:
            raise ValueError(f"data_path is not set for {self}.")

    def _instrument_param_path(self, mode: str, instrument: str):
        path = os.path.join(self._mode_param_path(mode=mode), instrument)
        u.mkdir_check(path)
        return path

    def _instrument_data_path(self, mode: str, instrument: str):
        path = os.path.join(self._mode_data_path(mode=mode), instrument)
        u.mkdir_check(path)
        return path

    def _epoch_param_path(self, mode: str, instrument: str, epoch_name: str):
        return os.path.join(self._instrument_param_path(mode=mode, instrument=instrument), f"{epoch_name}.yaml")

    def _epoch_data_path(self, mode: str, instrument: str, date: Time, epoch_name: str):
        path = os.path.join(self._instrument_data_path(mode=mode, instrument=instrument),
                            f"{date.to_datetime().date()}-{epoch_name}")
        u.mkdir_check(path)
        return path

    @classmethod
    def default_params(cls):
        default_params = {"name": None,
                          "type": "Field",
                          "centre": objects.position_dictionary.copy(),
                          "objects": {"<name>": objects.position_dictionary.copy()
                                      }
                          }
        return default_params

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):
        name, param_file, param_dict = _params_init(param_file)
        if param_file is None:
            return None
        # Check data_dir path for relevant .yamls (output_values, etc.)

        field_type = param_dict["type"]
        centre_ra, centre_dec = p.select_coords(param_dict["centre"])

        if field_type == "Field":
            return cls(name=name,
                       centre_coords=f"{centre_ra} {centre_dec}",
                       param_path=param_file,
                       data_path=param_dict["data_path"],
                       objs=param_dict["objects"]
                       )
        elif field_type == "FRBField":
            return FRBField.from_file(param_file)

    @classmethod
    def from_params(cls, name):
        path = os.path.join(p.param_path, "fields", name, name)
        return cls.from_file(param_file=path)

    @classmethod
    def new_yaml(cls, name: str, path: str = None, quiet: bool = False):
        param_dict = cls.default_params()
        param_dict["name"] = name
        param_dict["data_path"] = os.path.join(config["top_data_dir"], name, "")
        if path is not None:
            path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict


class StandardField(Field):
    a = 0.0


class FRBField(Field):
    def __init__(self,
                 name: str = None,
                 centre_coords: Union[SkyCoord, str] = None,
                 param_path: str = None,
                 data_path: str = None,
                 objs: List[objects.Object] = None,
                 frb: objects.FRB = None):
        if centre_coords is None:
            if frb is not None:
                centre_coords = frb.position

        # Input attributes

        super().__init__(name=name,
                         centre_coords=centre_coords,
                         param_path=param_path,
                         data_path=data_path,
                         objs=objs
                         )

        self.frb = frb
        self.epochs_imaging_old = {}

    @classmethod
    def default_params(cls):
        default_params = super().default_params()

        default_params.update({
            "type": "FRBField",
            "frb": objects.FRB.default_params(),
            "subtraction": {"template_epochs": {"des": None,
                                                "fors2": None,
                                                "xshooter": None,
                                                "sdss": None
                                                }
                            }
        })

        return default_params

    @classmethod
    def new_yaml(cls, name: str, path: str = None, quiet: bool = False):
        param_dict = super().new_yaml(name=name, path=None)
        param_dict["frb"]["name"] = name
        param_dict["frb"]["host_galaxy"]["name"] = name.replace("FRB", "HG")
        if path is not None:
            path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):
        name, param_file, param_dict = _params_init(param_file)

        # Check data_dir path for relevant .yamls (output_values, etc.)

        centre_ra, centre_dec = p.select_coords(param_dict["centre"])
        frb = objects.FRB.from_dict(param_dict["frb"])

        return cls(name=name,
                   centre_coords=f"{centre_ra} {centre_dec}",
                   param_path=param_file,
                   data_path=param_dict["data_path"],
                   objs=param_dict["objects"],
                   frb=frb
                   )

    @classmethod
    def convert_old_param(cls, frb: str):
        new_params = cls.new_yaml(name=frb, path=None)
        old_params = p.object_params_frb(frb)

        new_params["centre"]["dec"]["decimal"] = old_params["burst_dec"]
        new_params["centre"]["dec"]["dms"] = old_params["burst_dec_str"]

        new_params["centre"]["ra"]["decimal"] = old_params["burst_ra"]
        new_params["centre"]["ra"]["hms"] = old_params["burst_ra_str"]

        new_params["data_path"] = old_params["data_dir"]

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
                new_params["objects"][obj] = objects.position_dictionary
                new_params["objects"][obj]["dec"]["decimal"] = old_params["other_objects"][obj]["dec"]
                new_params["objects"][obj]["ra"]["decimal"] = old_params["other_objects"][obj]["ra"]
        del new_params["objects"]["<name>"]

        new_params["subtraction"]["template_epochs"]["des"] = old_params["template_epoch_des"]
        new_params["subtraction"]["template_epochs"]["fors2"] = old_params["template_epoch_fors2"]
        new_params["subtraction"]["template_epochs"]["sdss"] = old_params["template_epoch_sdss"]
        new_params["subtraction"]["template_epochs"]["xshooter"] = old_params["template_epoch_xshooter"]

        p.save_params(file=os.path.join(p.param_path, "fields", frb, f"{frb}.yaml"), dictionary=new_params, quiet=False)

    def gather_epochs_old(self):
        print("Searching for old-format imaging epoch param files...")
        epochs = {}
        param_dir = p.param_path
        for instrument_path in filter(lambda d: d.startswith("epochs_"), os.listdir(param_dir)):
            instrument = instrument_path.split("_")[-1]
            instrument_path = os.path.join(param_dir, instrument_path)
            for epoch_param in filter(lambda f: f.startswith(self.name) and f.endswith(".yaml"),
                                      os.listdir(instrument_path)):
                epoch_name = epoch_param[:epoch_param.find('.yaml')]
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
    def __init__(self,
                 param_path: str = None,
                 name: str = None,
                 field: Union[str, Field] = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 program_id: str = None,
                 target: str = None,
                 do: Union[list, str] = None
                 ):

        # Input attributes
        self.param_path = param_path
        self.name = name
        self.field = field
        self.data_path = data_path
        if data_path is not None:
            u.mkdir_check_nested(data_path)
        self.instrument = instrument
        self.date = date
        if type(self.date) is str:
            self.date = Time(date)
        self.program_id = program_id
        self.target = target

        self.do = do

        # Written attributes
        self.output_file = None
        self.stages_complete = self.stages()
        self.log = {}

        self.binning = None

        # Data reduction paths
        self.paths = {}
        self._path_0_raw()

        # Frames
        self.frames_raw = []
        self.frames_bias = []
        self.frames_standard = []
        self.frames_science = []
        self.frames_dark = []

        self.load_output_file()

    def pipeline(self, **kwargs):
        self._pipeline_init()

    def _pipeline_init(self, ):
        if self.data_path is not None:
            u.mkdir_check(self.data_path)
        else:
            raise ValueError(f"data_path has not been set for {self}")
        self.do = _check_do_list(self.do)

    def proc_1_initial_setup(self):
        do = self.query_stage("Do initial setup of files?", stage='1-initial_setup')
        return do

    def _path_0_raw(self):
        if self.data_path is not None and "raw_dir" not in self.paths:
            self.paths["raw_dir"] = os.path.join(self.data_path, epoch_stage_dirs["0-download"])

    def load_output_file(self):
        if self.output_file is None:
            if self.data_path is not None and self.name is not None:
                self.output_file = os.path.join(self.data_path, f"{self.name}_outputs.yaml")
                outputs = p.load_params(file=self.output_file)
                if outputs is not None:
                    self.stages_complete.update(outputs["stages"])
                    if "paths" in outputs:
                        self.paths.update(outputs["paths"])
                return outputs
            else:
                return False
        else:
            return True

    def _output_dict(self):
        return {
            "stages": self.stages_complete,
            "paths": self.paths,
        }

    def update_output_file(self):
        if self.output_file is not None:
            param_dict = p.load_params(self.output_file)
            if param_dict is None:
                param_dict = {}
            # For each of these, check if None first.
            param_dict.update(self._output_dict())
            p.save_params(dictionary=param_dict, file=self.output_file)
        else:
            raise ValueError("Output could not be saved to file due to lack of valid output path.")

    def check_done(self, stage: str):
        if stage not in self.stages_complete:
            raise ValueError(f"{stage} is not a valid stage for this Epoch.")
        return self.stages_complete[stage]

    def query_stage(self, message: str, stage: str):
        n = int(stage[0])
        if self.do is not None:
            if n in self.do:
                return True
        else:
            message = f"{n}. {message}"
            done = self.check_done(stage=stage)
            if done is not None:
                time_since = (Time.now() - done).sec * units.second
                time_since = u.relevant_timescale(time_since)
                message += f" (last performed at {done.isot}, {time_since.round(1)} ago)"
            options = ["No", "Yes", "Exit"]
            opt, _ = u.select_option(message=message, options=options)
            if opt == 0:
                return False
            if opt == 1:
                return True
            if opt == 2:
                exit(0)

    def set_program_id(self, program_id: str):
        self.program_id = program_id
        self.update_param_file("program_id")

    def set_date(self, date: Union[str, Time]):
        self.date = date
        self.update_param_file("date")

    def set_target(self, target: str):
        self.target = target
        self.update_param_file("target")

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

    def get_path(self, key):
        if key in self.paths:
            return self.paths[key]
        else:
            raise KeyError(f"{key} has not been set.")

    def add_raw_frame(self, raw_frame: image.Image):
        self.frames_raw.append(raw_frame)
        self.sort_frame(raw_frame)

    def sort_frame(self, frame: image.Image):
        if frame.frame_type == "bias":
            self.frames_bias.append(frame)
        elif frame.frame_type == "science":
            self.frames_science.append(frame)
        elif frame.frame_type == "standard":
            self.frames_standard.append(frame)
        elif frame.frame_type == "dark":
            self.frames_dark.append(frame)

    @classmethod
    def stages(cls):
        return {"1-initial_setup": None}

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "field": None,
            "data_path": None,
            "instrument": None,
            "date": None,
            "target": None,
            "program_id": None
        }
        return default_params

    @classmethod
    def new_yaml(cls, name: str, path: str = None, quiet: bool = False):
        param_dict = cls.default_params()
        param_dict["name"] = name
        if path is not None:
            path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
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


class ImagingEpoch(Epoch):
    def __init__(self,
                 name: str = None,
                 field: Union[str, Field] = None,
                 param_path: str = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 program_id: str = None,
                 target: str = None,
                 standard_epochs: list = None):
        super().__init__(name=name, field=field, param_path=param_path, data_path=data_path, instrument=instrument,
                         date=date, program_id=program_id, target=target)
        self.guess_data_path()

    def guess_data_path(self):
        if self.data_path is None and self.field is not None and self.field.data_path is not None and \
                self.instrument is not None and self.date is not None:
            self.data_path = os.path.join(self.field.data_path, "Imaging", self.instrument.upper(), self.date.isot)
        return self.data_path

    @classmethod
    def stages(cls):
        stages = super().stages()
        stages.update({})
        return stages

    @classmethod
    def from_params(cls, name: str, field: Union[Field, str] = None, instrument: str = None, old_format: bool = False):
        instrument = instrument.lower()
        field_name, field = cls._from_params_setup(name=name, field=field)
        if old_format:
            instrument = instrument.split("-")[-1]
            path = os.path.join(p.param_path, f"epochs_{instrument}", name)
        else:
            path = os.path.join(p.param_path, "fields", field_name, "imaging", instrument, name)
        return cls.from_file(param_file=path, field=field)

    @classmethod
    def from_file(cls, param_file: Union[str, dict], old_format: bool = False, field: Field = None):

        name, param_file, param_dict = _params_init(param_file)

        if old_format:
            instrument = "vlt-fors2"
        else:
            instrument = param_dict["instrument"].lower()

        if field is None:
            field = param_dict["field"]

        sub_cls = cls.select_child_class(instrument=instrument)
        if sub_cls is ImagingEpoch:
            return cls(name=name,
                       field=field,
                       param_path=param_file,
                       data_path=param_dict['data_path'],
                       instrument=instrument,
                       date=param_dict['date'],
                       program_id=param_dict["program_id"],
                       obj=param_dict["obj"]
                       )
        else:
            return sub_cls.from_file(param_dict, name=name, old_format=old_format, field=field)

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        default_params.update({
            "sextractor":
                {"aperture_diameters": [7.72],
                 "dual_mode": True,
                 "threshold": 1.5,
                 "kron_factor": 3.5,
                 "kron_radius_min": 1.0
                 },
            "background_subtraction":
                {"renormalise_centre": objects.position_dictionary.copy(),
                 "test_synths":
                     [{"position": objects.position_dictionary.copy(),
                       "mags": {}
                       }]

                 },
            "skip":
                {"esoreflex_copy": False,
                 "sextractor_individual": False,
                 "astrometry_net": False,
                 "sextractor": False,
                 "esorex": False,
                 },
        })
        return default_params

    @classmethod
    def select_child_class(cls, instrument: str):
        instrument = instrument.lower()
        if instrument == "vlt-fors2":
            return FORS2ImagingEpoch
        elif instrument in instruments_imaging:
            return ImagingEpoch
        else:
            raise ValueError(f"Unrecognised instrument {instrument}")


class ESOImagingEpoch(ImagingEpoch):

    def __init__(self,
                 name: str = None,
                 field: Field = None,
                 param_path: str = None,
                 data_path: str = None,
                 instrument: str = None,
                 program_id: str = None,
                 date: Union[str, Time] = None,
                 standard_epochs: list = None):
        super().__init__(name=name, field=field, param_path=param_path, data_path=data_path, instrument=instrument,
                         date=date, program_id=program_id,
                         standard_epochs=standard_epochs)

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)
        self.proc_0_download()
        self.proc_1_initial_setup()

    def proc_0_download(self, do: list = None):
        if self.query_stage("Download raw data from ESO archive?", stage='0-download'):
            r = self.retrieve()
            if r:
                self.stages_complete['0-download'] = Time.now()
                self.update_output_file()

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
        param_dict.update({"0-download": None,
                           })
        return param_dict


class FORS2ImagingEpoch(ESOImagingEpoch):

    @classmethod
    def from_file(cls, param_file: Union[str, dict], name: str = None, old_format: bool = False, field: Field = None):

        if old_format:
            if name is None:
                raise ValueError("name must be provided for old_format=True.")
            param_file = cls.convert_old_params(epoch_name=name)
        name, param_file, param_dict = _params_init(param_file)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")

        if field is None:
            field = param_dict["field"]

        return cls(name=name,
                   field=field,
                   param_path=param_file,
                   data_path=param_dict['data_path'],
                   instrument='vlt-fors2',
                   program_id=param_dict['program_id'],
                   date=param_dict['date'])

    @classmethod
    def convert_old_params(cls, epoch_name: str):
        new_params = cls.new_yaml(name=epoch_name, path=None)
        old_params = p.object_params_fors2(epoch_name)

        field = epoch_name[:epoch_name.find("_")]

        new_params["instrument"] = "vlt-fors2"
        new_params["data_path"] = old_params["data_dir"]
        new_params["field"] = field

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
                synth_dict = {}
                synth_dict["position"]["ra"] = old_params["test_synths"]["ra"][i]
                synth_dict["position"]["dec"] = old_params["test_synths"]["dec"][i]
                synth_dict["mags"]["g"] = old_params["test_synths"]["g_mag"][i]
                synth_dict["mags"]["I"] = old_params["test_synths"]["I_mag"][i]
                new_params["background_subtraction"]["test_synths"].append(synth_dict)

        new_params["skip"]["esoreflex_copy"] = old_params["skip_copy"]
        new_params["skip"]["sextractor_individual"] = not old_params["do_sextractor_individual"]
        new_params["skip"]["astrometry_net"] = old_params["skip_astrometry"]
        new_params["skip"]["sextractor"] = not old_params["do_sextractor"]
        new_params["skip"]["esorex"] = old_params["skip_esorex"]

        instrument_path = os.path.join(p.param_path, "fields", field, "imaging", "vlt-fors2")
        u.mkdir_check(instrument_path)
        output_path = os.path.join(instrument_path, epoch_name)
        p.save_params(file=output_path,
                      dictionary=new_params, quiet=False)

        return output_path


class SpectroscopyEpoch(Epoch):
    grisms = {}

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
        self.grism = grism
        if grism is None or grism not in self.grisms:
            warnings.warn("grism not configured.")

        self.obj = target

        self._path_2_pypeit()
        self.standards_raw = []
        self._instrument_pypeit = self.instrument.replace('-', '_')

        self._pypeit_file = None
        self._pypeit_sorted_file = None

    def read_pypeit_sorted_file(self):
        if "pypeit_setup_dir" in self.paths and self.paths["pypeit_setup_dir"] is not None:
            setup_files = self.paths["pypeit_setup_dir"]
            sorted_path = os.path.join(setup_files,
                                       filter(lambda f: f.endswith(".sorted"), os.listdir(setup_files)).__next__())
            with open(sorted_path) as sorted_file:
                self._pypeit_sorted_file = sorted_file.readlines()
        else:
            raise KeyError("pypeit_setup_dir has not been set.")

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
            raw = image.SpecRaw.from_pypeit_line(line=line, pypeit_raw_path=self.paths["raw_dir"])
            self.add_raw_frame(raw)
        return pypeit_lines

    def read_pypeit_file(self, setup: str):
        if "pypeit_dir" in self.paths and self.paths["pypeit_dir"] is not None:
            filename = f"{self._instrument_pypeit}_{setup}"
            self._read_pypeit_file(filename=filename)
            return self._pypeit_file
        else:
            raise KeyError("pypeit_run_dir has not been set.")

    def _set_pypeit_file(self, lines: list):
        self._pypeit_file = lines

    def _get_pypeit_file(self):
        return self._pypeit_file

    def write_pypeit_file(self):
        """
        Rewrites the stored .pypeit file to disk at its original path.
        :return: path of .pypeit file.
        """
        if self._pypeit_file is not None:
            pypeit_file_path = self.paths["pypeit_file"]
            # Delete .pypeit file, to be rewritten.
            os.remove(pypeit_file_path)
            # Write .pypeit file to disk.
            with open(pypeit_file_path, 'w') as pypeit_file:
                pypeit_file.writelines(self._pypeit_file)
            return pypeit_file_path
        else:
            raise ValueError("pypeit_file has not yet been read.")

    def add_pypeit_user_param(self, param: list, value: str):
        """
        Inserts a parameter for the PypeIt run at the correct point in the stored .pypeit file.
        :param param: For m
        :param value:
        :return:
        """
        pypeit_file = self._get_pypeit_file()
        if pypeit_file is not None:
            # Build the final line of the setting specially.
            setting = "\t" * (len(param) - 1) + f"{param.pop()} = {value}\n"
            # First, check if param sub-headings are already there:
            p_start = pypeit_file.index("[rdx]\n")
            insert_here = False
            for i, par in enumerate(param):
                # Encase each level of the parameter in the correct number of square brackets and tabs.
                par = "\t" * i + "[" * (i + 1) + par + "]" * (i + 1) + "\n"
                if par in pypeit_file and not insert_here:
                    p_start = pypeit_file.index(par) + 1
                else:
                    pypeit_file.insert(p_start, par)
                    p_start += 1
                    insert_here = True
            # Insert the final line.
            pypeit_file.insert(p_start, setting)

            self._set_pypeit_file(lines=pypeit_file)
        else:
            raise ValueError("pypeit_file has not yet been read.")

    def add_pypeit_file_lines(self, lines: list):
        if self._pypeit_file is not None:
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

    def proc_2_pypeit_setup(self):
        return self.query_stage("Do PypeIt setup?", stage='2-pypeit_setup')

    def proc_3_pypeit_run(self, do_not_reuse_masters=False):
        return self.query_stage("Run PypeIt?", stage='3-pypeit_run')

    def _path_2_pypeit(self):
        if self.data_path is not None and "pypeit_dir" not in self.paths:
            self.paths["pypeit_dir"] = os.path.join(self.data_path, epoch_stage_dirs["2-pypeit"])

    @classmethod
    def stages(cls):
        param_dict = super().stages()
        param_dict.update({"2-pypeit_setup": None,
                           "3-pypeit_run": None,
                           "4-pypeit_flux_calib": None})
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
        name, param_file, param_dict = _params_init(param_file)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")
        instrument = param_dict["instrument"].lower()
        if field is None:
            field = param_dict["field"]
        sub_cls = cls.select_child_class(instrument=instrument)
        # if sub_cls is SpectroscopyEpoch:
        return sub_cls(name=name,
                       field=field,
                       param_path=param_file,
                       data_path=param_dict["data_path"],
                       instrument=instrument,
                       date=param_dict["date"],
                       program_id=param_dict["program_id"],
                       )
        # else:
        # return sub_cls.from_file(param_file=param_file, field=field)

    @classmethod
    def from_params(cls, name, field: Union[Field, str] = None, instrument: str = None):
        instrument = instrument.lower()
        field_name, field = cls._from_params_setup(name=name, field=field)
        path = os.path.join(p.param_path, "fields", field_name, "spectroscopy", instrument, name)
        return cls.from_file(param_file=path, field=field)


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
            do_not_reuse_masters = True
        else:
            do_not_reuse_masters = False
        self.proc_0_raw()
        self.proc_1_initial_setup()
        self.proc_2_pypeit_setup()
        self.proc_3_pypeit_run(do_not_reuse_masters=do_not_reuse_masters)

    def proc_0_raw(self):
        if self.query_stage("Download raw data from ESO archive?", stage='0-download'):
            self._path_0_raw()
            r = self.retrieve()
            if r:
                self.stages_complete['0-download'] = Time.now()
                self.update_output_file()

    def proc_1_initial_setup(self):
        if self.query_stage("Do initial setup of files?", stage='1-initial_setup'):
            self._path_0_raw()
            m_path = os.path.join(self.paths["raw_dir"], "M")
            u.mkdir_check(m_path)
            os.system(f"mv {os.path.join(self.paths['raw_dir'], 'M.')}* {m_path}")
            ff.fits_table_all(input_path=self.paths["raw_dir"],
                              output_path=os.path.join(self.data_path, f"{self.name}_fits_table_science.csv"))
            ff.fits_table_all(input_path=self.paths["raw_dir"],
                              output_path=os.path.join(self.data_path, f"{self.name}_fits_table_all.csv"),
                              science_only=False)
            self.stages_complete['1-initial_setup'] = Time.now()
            self.update_output_file()

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
    _instrument_pypeit = "vlt_fors2"
    grisms = {
        "GRIS_300I": {
            "lambda_min": 6000 * units.angstrom,
            "lambda_max": 11000 * units.angstrom
        }}

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)
        self.proc_4_pypeit_flux()
        self.proc_5_pypeit_coadd()

    def proc_2_pypeit_setup(self):
        if self.query_stage("Do PypeIt setup?", stage='2-pypeit_setup'):
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
            std_raw = image.SpecRaw.from_pypeit_line(std_line, pypeit_raw_path=self.paths['raw_dir'])
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
            self.write_pypeit_file()

            self.stages_complete['2-pypeit_setup'] = Time.now()
            self.update_output_file()

    def proc_3_pypeit_run(self, do_not_reuse_masters=False):
        if self.query_stage("Run PypeIt?", stage='3-pypeit_run'):
            spec.run_pypeit(pypeit_file=self.paths['pypeit_file'],
                            redux_path=self.paths['pypeit_run_dir'],
                            do_not_reuse_masters=do_not_reuse_masters)
            self.stages_complete['3-pypeit_run'] = Time.now()
            self.update_output_file()

    def proc_4_pypeit_flux(self):
        if self.query_stage("Do fluxing with PypeIt?", stage='4-pypeit_flux_calib'):
            pypeit_science_dir = os.path.join(self.paths["pypeit_run_dir"], "Science")
            std_reduced_filename = filter(lambda f: "spec1d" in f and "STD" in f and f.endswith(".fits"),
                                          os.listdir(pypeit_science_dir)).__next__()
            std_reduced_path = os.path.join(pypeit_science_dir, std_reduced_filename)
            print(f"Using {std_reduced_path} for fluxing.")
            sensfunc_path = os.path.join(self.paths['pypeit_run_dir'], "sens.fits")
            # Generate sensitivity function from standard observation
            os.system(f"pypeit_sensfunc {std_reduced_path} -o {sensfunc_path}")
            # Generate flux setup file.
            cwd = os.getcwd()
            os.chdir(self.paths["pypeit_run_dir"])
            os.system(f"pypeit_flux_setup {pypeit_science_dir}")
            os.chdir(cwd)
            flux_setup_path = os.path.join(self.paths["pypeit_run_dir"], f"{self._instrument_pypeit}.flux")

            # Insert name of sensitivity file to flux setup file.
            with open(flux_setup_path, "r") as flux_setup:
                flux_lines = flux_setup.readlines()
            file_first = flux_lines.index("flux read\n") + 1
            flux_lines[file_first] = flux_lines[file_first][:-1] + " " + sensfunc_path + "\n"
            # Delete flux setup file for rewriting.
            os.remove(flux_setup_path)
            # Rewrite modified .flux file.
            with open(flux_setup_path, "w") as flux_setup:
                flux_setup.writelines(flux_lines)
            # Run pypeit_flux_calib
            os.system(f"pypeit_flux_calib {flux_setup_path}")

            self.paths["pypeit_sensitivity_file"] = sensfunc_path
            self.paths["pypeit_std_reduced"] = std_reduced_path
            self.paths["pypeit_science_dir"] = pypeit_science_dir
            self.paths["pypeit_flux_setup"] = flux_setup_path
            self.stages_complete['4-pypeit_flux_calib'] = Time.now()
            self.update_output_file()

    def proc_5_pypeit_coadd(self):
        if self.query_stage(
                "Do coaddition with PypeIt?\nYou should first inspect the 2D spectra to determine which objects to co-add.",
                stage='5-pypeit_coadd'):
            for file in filter(lambda f: "spec2d" in f, os.listdir(self.paths["pypeit_science_dir"])):
                path = os.path.join(self.paths["pypeit_science_dir"], file)
                os.system(f"pypeit_show_2dspec {path}")

    def proc_6_marz_format(self, do: list = None):
        return self.query_stage("Convert co-added 1D spectra to Marz format?", stage='6-marz-format')


class XShooterSpectroscopyEpoch(ESOSpectroscopyEpoch):
    _instrument_pypeit = "vlt_xshooter"
    _cfg_split_letters = {"uvb": "A",
                          "nir": "B",
                          "vis": "C"}
    arms = ['uvb', "nir", "vis"]

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
        self._pypeit_file = {"uvb": [],
                             "vis": [],
                             "nir": []}
        self._pypeit_user_param_start = {"uvb": None,
                                         "vis": None,
                                         "nir": None}
        self._pypeit_user_param_end = {"uvb": None,
                                       "vis": None,
                                       "nir": None}

        self.binning = {"uvb": None,
                        "vis": None,
                        "nir": None}
        self.decker = {"uvb": None,
                       "vis": None,
                       "nir": None}

        self.load_output_file()
        self._current_arm = None

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)
        # self.proc_4_pypeit_flux()
        # self.proc_5_pypeit_coadd()

    def proc_2_pypeit_setup(self):
        if self.query_stage("Do PypeIt setup?", stage='2-pypeit_setup'):
            self._path_2_pypeit()
            for arm in self.arms:
                self._current_arm = arm
                # arm = "vis"
                spec.pypeit_setup(root=self.paths['raw_dir'], output_path=self.paths['pypeit_dir'],
                                  spectrograph=f"{self._instrument_pypeit}_{arm}")
                setup_files = os.path.join(self.paths["pypeit_dir"], 'setup_files', '')
                os.system(f"rm {setup_files}*")
                setup = self._cfg_split_letters[arm]
                spec.pypeit_setup(root=self.paths['raw_dir'], output_path=self.paths['pypeit_dir'],
                                  spectrograph=f"{self._instrument_pypeit}_{arm}", cfg_split=setup)

                # Retrieve text from .pypeit file
                self.read_pypeit_file(setup=setup)
                if arm == "nir":
                    self.add_pypeit_user_param(param=["calibrations", "pixelflatframe", "process", "use_darkimage"],
                                               value="True")
                    self.add_pypeit_user_param(param=["calibrations", "illumflatframe", "process", "use_darkimage"],
                                               value="True")
                    self.add_pypeit_user_param(param=["calibrations", "traceframe", "process", "use_darkimage"],
                                               value="True")
                # Remove incompatible binnings and frametypes
                self.find_science_attributes(arm=arm)
                print(f"\nRemoving incompatible files for {arm} arm:")
                pypeit_file = self._get_pypeit_file()
                for raw_frame in self.frames_raw[arm]:
                    if raw_frame.binning != self.binning[arm] \
                            or raw_frame.frame_type == "None" \
                            or raw_frame.decker not in ["Pin_row", self.decker[arm]]:
                        pypeit_file.remove(raw_frame.pypeit_line)
                    if arm == "nir":
                        # For the NIR arm, PypeIt only works if you use the Science frames for the arc, tilt calib.
                        if raw_frame.frame_type in ["arc,tilt", "tilt,arc"]:
                            self._pypeit_file[arm].remove(raw_frame.pypeit_line)
                        elif raw_frame.frame_type == "science":
                            raw_frame.frame_type = "science,arc,tilt"
                            # Find original line in PypeIt file
                            to_replace = pypeit_file.index(raw_frame.pypeit_line)
                            # Rewrite pypeit line.
                            raw_frame.pypeit_line = raw_frame.pypeit_line.replace("science", "science,arc,tilt")
                            pypeit_file[to_replace] = raw_frame.pypeit_line
                self._set_pypeit_file(pypeit_file)
                self.set_path("pypeit_setup_dir", setup_files)
                self.write_pypeit_file()
            self._current_arm = None
            self.stages_complete['2-pypeit_setup'] = Time.now()
            self.update_output_file()

    def proc_3_pypeit_run(self, do_not_reuse_masters=False):
        for arm in self.arms:
            self._current_arm = arm
            if self.query_stage(f"Run PypeIt for {arm.upper()} arm?", stage='3-pypeit_run'):
                spec.run_pypeit(pypeit_file=self.get_path('pypeit_file'),
                                redux_path=self.get_path('pypeit_run_dir'),
                                do_not_reuse_masters=do_not_reuse_masters)
                self.stages_complete[f'3-pypeit_run_{arm}'] = Time.now()
                self.update_output_file()
        self._current_arm = None

    def add_raw_frame(self, raw_frame: image.Image):
        if self._current_arm is not None:
            arm = self._current_arm
            self.frames_raw[arm].append(raw_frame)
            self.sort_frame(raw_frame)
        else:
            raise ValueError("self._current_arm not set.")

    def sort_frame(self, frame: image.Image):
        arm = self._current_arm
        if frame.frame_type == "bias":
            self.frames_bias[arm].append(frame)
        elif frame.frame_type == "science":
            self.frames_science[arm].append(frame)
        elif frame.frame_type == "standard":
            self.frames_standard[arm].append(frame)
        elif frame.frame_type == "dark":
            self.frames_dark[arm].append(frame)

    def find_science_attributes(self, arm: str):
        if self.frames_science:
            frame = self.frames_science[arm][0]
            self.binning[arm] = frame.binning
            self.decker[arm] = frame.decker
        else:
            raise ValueError(f"Science frames list for {arm} is empty.")
        return self.binning[arm]

    def read_pypeit_file(self, setup: str):
        if "pypeit_dir" in self.paths and self.paths["pypeit_dir"] is not None:
            arm = self._current_arm
            filename = f"{self._instrument_pypeit}_{arm}_{setup}"
            self._read_pypeit_file(filename=filename)
            return self._pypeit_file[arm]
        else:
            raise KeyError("pypeit_run_dir has not been set.")

    def _set_pypeit_file(self, lines: list):
        if self._current_arm is not None:
            self._pypeit_file[self._current_arm] = lines
        else:
            raise ValueError("No arm currently active.")

    def _get_pypeit_file(self):
        if self._current_arm is not None:
            return self._pypeit_file[self._current_arm]
        else:
            raise ValueError("No arm currently active.")

    def _get_key_arm(self, key):
        if self._current_arm is not None:
            if self._current_arm is not None:
                arm = self._current_arm
                key = f"{arm}_{key}"
        return key

    def get_path(self, key):
        key = self._get_key_arm(key)
        return self.paths[key]

    def set_path(self, key: str, value: str):
        key = self._get_key_arm(key)
        self.paths[key] = value

    @classmethod
    def stages(cls):
        param_dict = super().stages()
        param_dict.update({"2-pypeit_setup": None,
                           "3-pypeit_run_uvb": None,
                           "3-pypeit_run_vis": None,
                           "3-pypeit_run_nir": None,
                           "4-pypeit_flux_calib": None})
        return param_dict

    def _output_dict(self):
        return {
            "stages": self.stages_complete,
            "paths": self.paths,
        }

    def load_output_file(self):
        outputs = super().load_output_file()
        if outputs not in [None, True, False]:
            self.stages_complete.update(outputs["stages"])
            if "paths" in outputs:
                self.paths.update(outputs[f"paths"])
        return outputs

    def write_pypeit_file(self):
        """
        Rewrites the stored .pypeit file to disk at its original path.
        :return: path of .pypeit file.
        """
        arm = self._current_arm
        pypeit_lines = self._get_pypeit_file()
        if pypeit_lines is not None:
            pypeit_file_path = self.get_path("pypeit_file")
            # Delete .pypeit file, to be rewritten.
            os.remove(pypeit_file_path)
            # Write .pypeit file to disk.
            print(f"Writing pypeit file to {pypeit_file_path}")
            with open(pypeit_file_path, 'w') as pypeit_file:
                pypeit_file.writelines(pypeit_lines)
        else:
            raise ValueError(f"pypeit_file has not yet been read for {arm}.")
        return pypeit_file_path

# def test_frbfield_from_params():
#     frb_field = FRBField.from_file("FRB181112")
#     assert frb_field.frb.position_err.a_stat ==
