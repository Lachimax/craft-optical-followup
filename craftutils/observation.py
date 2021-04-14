import os.path
from typing import Union, List

from astropy.coordinates import SkyCoord

from craftutils import astrometry as a
from craftutils import utils as u
from craftutils import params as p
from craftutils import astronobjects

config = p.config


class Field:
    default_params = {"centre": {"decimal_ra": 0.0,
                                 "decimal_dec": 0.0,
                                 "str_ra": "0h0m0s",
                                 "str_dec": "0d0m0s"},
                      "data_path": ""
                      }

    def from_file(self, path):

        u.sanitise_file_ext(filename=path, ext="yaml")
        param_dict = p.load_params(file=path)
        # Check data_dir path for relevant .yamls (output_values, etc.)

        name = os.path.splitext(os.path.split(path)[-1])[0]
        centre_ra, centre_dec = p.select_coords(param_dict["centre"])

        return self.__init__(name=name,
                             centre_coords=f"{centre_ra} {centre_dec}",
                             param_path=path,
                             data_path=param_dict["data_path"],
                             objects=None
                             )

    def __init__(self,
                 name: str = None,
                 centre_coords: Union[SkyCoord, str] = None,
                 param_path: str = None,
                 data_path: str = None,
                 objects: List[astronobjects.Object] = None
                 ):
        """

        :param centre_coords:
        :param name:
        :param param_path:
        :param data_path:
        :param objects: a list of objects of interest in the field. The primary object of interest should be first in
        the list.
        """

        if centre_coords is None:
            if objects is None:
                raise ValueError("Either centre_coords or objects must be given.")
            else:
                centre_coords = objects[0].coords

        self.centre_coords = a.attempt_skycoord(centre_coords)

        self.name = name
        self.param_path = param_path
        self.output_path = data_path

    @classmethod
    def new_yaml(cls, name: str, path: str = None, quiet: bool = False):
        param_dict = cls.default_params
        param_dict["data_path"] = os.path.join(config["top_data_dir"], name, "")
        if path is not None:
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict

    def __str__(self):
        return ""

    def __repr__(self):
        return self.__str__()


class FRBField(Field):
    def __init__(self,
                 burst_coords: Union[SkyCoord, str],
                 centre_coords: Union[SkyCoord, str] = None, ):
        if centre_coords is None:
            centre_coords = burst_coords

        super(FRBField, self).__init__(centre_coords=centre_coords)


class Epoch:
    mjd = 0.0


class Image:
    frame_type = "stacked"
