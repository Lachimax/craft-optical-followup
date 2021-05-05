import os.path
import warnings
from typing import Union, List

from astropy.time import Time
from astropy.coordinates import SkyCoord

import craftutils.astrometry as a
import craftutils.utils as u
import craftutils.params as p
import craftutils.retrieve as retrieve
import craftutils.observation.objects as objects
import craftutils.observation.epoch.epoch as ep

config = p.config


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


def params_init(param_file: Union[str, dict]):
    if type(param_file) is str:
        # Load params from .yaml at path.
        param_file = u.sanitise_file_ext(filename=param_file, ext="yaml")
        param_dict = p.load_params(file=param_file)
        if param_dict is None:
            return None, None, None
        name = u.get_filename(path=param_file, include_ext=False)
        param_dict["param_path"] = param_file
    else:
        param_dict = param_file
        name = param_dict["name"]
        param_file = param_dict["param_path"]

    return name, param_file, param_dict


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
        self.data_path = data_path

        # Derived attributes

        self.epochs_spectroscopy = []
        self.epochs_imaging = []

    def __str__(self):
        return ""

    def __repr__(self):
        return self.__str__()

    def list_epochs(self):

        for epoch in self.epochs:
            a = 0

    def mkdir(self):
        u.mkdir_check(self.data_path)

    def mkdir_params(self):
        if self.param_dir is not None:
            u.mkdir_check(os.path.join(self.param_dir, "spectroscopy"))
            u.mkdir_check(os.path.join(self.param_dir, "imaging"))
        else:
            warnings.warn(f"param_dir is not set for this {type(self)}.")

    def gather_epochs(self, mode: str = "imaging"):
        print(f"Searching for {mode} epoch param files...")
        epochs = []
        if self.param_dir is not None:
            mode_path = os.path.join(self.param_dir, mode)
            print(f"Looking in {mode_path}")
            for instrument in filter(lambda d: os.path.isdir(os.path.join(mode_path, d)), os.listdir(mode_path)):
                instrument_path = os.path.join(mode_path, instrument)
                print(f"Looking in {instrument_path}")
                for epoch_param in filter(lambda f: f.endswith(".yaml"), os.listdir(instrument_path)):
                    epochs.append({
                        "name": epoch_param[:epoch_param.find(".yaml")],
                        "instrument": instrument,
                        "path": os.path.join(instrument_path, epoch_param),
                        "format": "current"
                    })
        return epochs

    def gather_epochs_spectroscopy(self):
        epochs = self.gather_epochs(mode="spectroscopy")
        self.epochs_spectroscopy += epochs
        return epochs

    def gather_epochs_imaging(self):
        epochs = self.gather_epochs(mode="imaging")
        self.epochs_imaging += epochs
        return epochs

    def select_epoch_imaging(self):
        self.epochs_imaging.sort(key=lambda e: e["name"])
        options = []
        for i, epoch in enumerate(self.epochs_imaging):
            old_string = ""
            if epoch["format"] == 'old':
                old_string = " (old format)"
            options.append(f'{epoch["name"]}{old_string} {epoch["instrument"]}')
        j, _ = u.select_option(message="Select epoch.", options=options)
        to_load = self.epochs_imaging.pop(j)
        old_format = False
        if to_load["format"] == "old":
            old_format = True
        epoch = ImagingEpoch.from_file(to_load['path'], old_format=old_format)
        self.epochs_imaging.append(epoch)
        return epoch

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
    def from_params(cls, name):
        path = os.path.join(p.param_path, "fields", name, name)
        return cls.from_file(param_file=path)

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):

        name, param_file, param_dict = params_init(param_file)
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
            return FRBField().from_file(param_file)

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
        name, param_file, param_dict = params_init(param_file)

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
        epochs = []
        param_dir = p.param_path
        for instrument_path in filter(lambda d: d.startswith("epochs_"), os.listdir(param_dir)):
            instrument = instrument_path.split("_")[-1]
            instrument_path = os.path.join(param_dir, instrument_path)
            for epoch_param in filter(lambda f: f.startswith(self.name) and f.endswith(".yaml"),
                                      os.listdir(instrument_path)):
                epochs.append({
                    "name": epoch_param[:epoch_param.find('.yaml')],
                    "instrument": instrument,
                    "path": os.path.join(instrument_path, epoch_param),
                    "format": "old"
                })
        self.epochs_imaging += epochs
        return epochs


epoch_stage_dirs = ["0-data_with_raw_calibs"]


class Epoch:
    def __init__(self,
                 name: str = None,
                 field: Union[str, Field] = None,
                 param_path: str = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 obj: str = None,
                 program_id: str = None
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
        self.obj = obj
        self.program_id = program_id

        # Written attributes
        self.output_file = None
        self.check_output_file_path()
        self.stages = {}

    def check_output_file_path(self):
        if self.output_file is None:
            if self.data_path is not None and self.name is not None:
                self.output_file = os.path.join(self.data_path, f"{self.name}_outputs.yaml")
                return True
            else:
                return False

    def query_stage(self, message, stage):
        done = self.check_done(stage=stage)
        if done:
            message += " (performed previously)"
        options = ["Yes", "Skip", "Exit"]
        opt, _ = u.select_option(message=message, options=options)
        if opt == 0:
            return True
        if opt == 1:
            return False
        if opt == 2:
            exit(0)

    def check_done(self, stage: str):
        if stage not in self.stages:
            raise ValueError(f"{stage} is not a valid stage for this Epoch.")
        return self.stages[stage]

    def set_program_id(self, program_id: str):
        self.program_id = program_id
        self.update_param_file("program_id")

    def set_date(self, date: Union[str, Time]):
        self.date = date
        self.update_param_file("date")

    def update_param_file(self, param: str):
        p_dict = {"program_id": self.program_id,
                  "date": self.date}
        if param not in p_dict:
            raise ValueError(f"Either {param} is not a valid parameter, or it has not been configured.")
        if self.param_path is None:
            raise ValueError("param_path has not been set.")
        else:
            params = p.load_params(self.param_path)
        params[param] = p_dict[param]
        p.save_params(file=self.param_path, dictionary=params)

    def update_output_file(self):
        proceed = self.check_output_file_path()
        if proceed:
            params = p.load_params(file=self.output_file)
            params.update({
                "stages": self.stages
            })
            p.save_params(dictionary=params, file=self.output_file)
        else:
            warnings.warn("Output could not be saved to file due to lack of valid output path.")

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "field": None,
            "data_path": None,
            "instrument": None,
            "date": None,
            "obj": None,
            "program_id": None
        }
        return default_params


class ImagingEpoch(Epoch):
    def __init__(self,
                 name: str = None,
                 field: Union[str, Field] = None,
                 param_path: str = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 program_id: str = None,
                 standard_epochs: list = None):
        super().__init__(name=name, field=field, param_path=param_path, data_path=data_path, instrument=instrument,
                         date=date, program_id=program_id)

        # Written attributes
        self.stages = {"download": False}

    @classmethod
    def from_params(cls, field: str, name: str):
        path = os.path.join(p.param_path, "fields", field, name)
        return cls.from_file(param_file=path)

    @classmethod
    def from_file(cls, param_file: Union[str, dict], old_format: bool = False):

        name, param_file, param_dict = params_init(param_file)

        if old_format:
            instrument = "vlt-fors2"
        else:
            instrument = param_dict["instrument"].lower()

        if instrument == "vlt-fors2":
            return FORS2ImagingEpoch().from_file(param_dict, name=name, old_format=old_format)
        else:
            return cls(name=name,
                       field=param_dict["field"],
                       param_path=param_file,
                       data_path=param_dict['data_path'],
                       instrument=instrument,
                       date=param_dict['date'],
                       program_id=param_dict["program_id"]
                       )

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
    def new_yaml(cls, name: str, path: str = None, quiet: bool = False):
        param_dict = cls.default_params()
        param_dict["name"] = name
        if path is not None:
            path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict


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

    def retrieve(self):
        """
        Check ESO archive for the epoch raw frames, and download those frames and associated files.
        :return:
        """
        if self.program_id is None:
            self.set_program_id(input("A program ID is required to retrieve ESO data. Enter here:\n"))
        if self.date is None:
            self.set_date(Time(
                input("An observation date is required to retrieve ESO data. Enter here, in iso or isot format:\n")))

        raw_path = os.path.join(self.data_path, epoch_stage_dirs[0])
        u.mkdir_check(raw_path)
        instrument = self.instrument.split('-')[-1]
        r = retrieve.save_eso_raw_data_and_calibs(output=raw_path, date_obs=self.date, obj=self.obj,
                                                  program_id=self.program_id, instrument=instrument)
        if r:
            os.system(f"uncompress {raw_path}/*.Z")
            self.stages['download'] = True
        return r


class FORS2ImagingEpoch(ESOImagingEpoch):

    @classmethod
    def from_file(cls, param_file: Union[str, dict], name: str = None, old_format: bool = False):
        if old_format:
            if name is None:
                raise ValueError("name must be provided for old_format=True.")
            param_file = cls.convert_old_params(epoch_name=name)
        name, param_file, param_dict = params_init(param_file)

        return cls(name=name,
                   field=param_dict["field"],
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


class SpectroscopyEpoch:
    def __init__(self,
                 name: str = None,
                 field: Field = None,
                 param_path: str = None,
                 data_path: str = None,
                 ):
        self.name = name
        self.field = field
        self.param_path = param_path
        self.data_path = data_path

# def test_frbfield_from_params():
#     frb_field = FRBField.from_file("FRB181112")
#     assert frb_field.frb.position_err.a_stat ==
