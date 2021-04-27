import os.path
from typing import Union, List

from astropy.coordinates import SkyCoord

from craftutils import astrometry as a
from craftutils import utils as u
from craftutils import params as p
from craftutils import astronobjects

config = p.config


class Field:
    def __init__(self,
                 name: str = None,
                 centre_coords: Union[SkyCoord, str] = None,
                 param_path: str = None,
                 data_path: str = None,
                 objects: Union[List[astronobjects.Object], dict] = None
                 ):
        """

        :param centre_coords:
        :param name:
        :param param_path:
        :param data_path:
        :param objects: a list of objects of interest in the field. The primary object of interest should be first in
        the list.
        """

        if objects is dict:
            obj_list = []
            for name in objects:
                obj = astronobjects.Object.from_dict(objects[name])
                obj_list.append(obj)
            objects = obj_list

        self.objects = objects

        if centre_coords is None:
            if objects is not None:
                centre_coords = objects[0].coords
        if centre_coords is not None:
            self.centre_coords = a.attempt_skycoord(centre_coords)

        self.name = name
        self.param_path = param_path
        self.output_path = data_path

    @classmethod
    def default_params(cls):
        default_params = {"name": None,
                          "type": "Field",
                          "centre": astronobjects.position_dictionary.copy(),
                          "objects": {"<name>": astronobjects.position_dictionary.copy()
                                      }
                          }
        return default_params

    @classmethod
    def from_params(cls, name):
        path = os.path.join(p.param_path, "fields", name)
        return cls.from_file(param_file=path)

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):

        if type(param_file) is str:
            u.sanitise_file_ext(filename=param_file, ext="yaml")
            param_dict = p.load_params(file=param_file)
            if param_dict is None:
                return None
            name = os.path.splitext(os.path.split(param_file)[-1])[0]
        else:
            param_dict = param_file
            name = param_dict["name"]
        # Check data_dir path for relevant .yamls (output_values, etc.)

        field_type = param_dict["type"]
        centre_ra, centre_dec = p.select_coords(param_dict["centre"])

        if field_type == "Field":
            return cls(name=name,
                       centre_coords=f"{centre_ra} {centre_dec}",
                       param_path=param_file,
                       data_path=param_dict["data_path"],
                       objects=param_file["objects"]
                       )
        elif field_type == "FRBField":
            return FRBField().from_file(param_dict)

    @classmethod
    def new_yaml(cls, name: str, path: str = None, quiet: bool = False):
        param_dict = cls.default_params()
        param_dict["data_path"] = os.path.join(config["top_data_dir"], name, "")
        param_dict["name"] = name
        if path is not None:
            path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict

    def __str__(self):
        return ""

    def __repr__(self):
        return self.__str__()


class FRBField(Field):
    def __init__(self,
                 name: str = None,
                 centre_coords: Union[SkyCoord, str] = None,
                 param_path: str = None,
                 data_path: str = None,
                 objects: List[astronobjects.Object] = None,
                 frb: astronobjects.FRB = None):
        if centre_coords is None:
            if frb is not None:
                centre_coords = frb.position

        super(FRBField, self).__init__(name=name,
                                       centre_coords=centre_coords,
                                       param_path=param_path,
                                       data_path=data_path,
                                       objects=objects
                                       )
        self.frb = frb

    @classmethod
    def new_yaml(cls, name: str, path: str = None, quiet: bool = False):
        param_dict = super(FRBField, cls).new_yaml(name=name, path=None)
        param_dict["frb"]["name"] = name
        param_dict["frb"]["host_galaxy"]["name"] = name.replace("FRB", "HG")
        if path is not None:
            path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict

    @classmethod
    def default_params(cls):
        default_params = super(FRBField, cls).default_params()
        default_params.update({
            "type": "FRBField",
            "frb": astronobjects.FRB.default_params(),
            "subtraction": {"template_epochs": {"des": None,
                                                "fors2": None,
                                                "xshooter": None,
                                                "sdss": None
                                                }
                            }
        })

        return default_params

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):
        if type(param_file) is str:
            u.sanitise_file_ext(filename=param_file, ext="yaml")
            param_dict = p.load_params(file=param_file)
            if param_dict is None:
                return None
            name = os.path.splitext(os.path.split(param_file)[-1])[0]
        else:
            param_dict = param_file
            name = param_dict["name"]
            param_file = None
        # Check data_dir path for relevant .yamls (output_values, etc.)

        centre_ra, centre_dec = p.select_coords(param_dict["centre"])
        frb = astronobjects.FRB.from_dict(param_dict["frb"])

        return cls(name=name,
                   centre_coords=f"{centre_ra} {centre_dec}",
                   param_path=param_file,
                   data_path=param_dict["data_path"],
                   objects=param_file["objects"],
                   frb=frb
                   )


class Epoch:
    mjd = 0.0


class Image:
    frame_type = "stacked"


def test_frbfield_from_params():
    frb_field = FRBField.from_file("FRB181112")
    assert frb_field.frb.position_err.a_stat
