import os
from typing import Union

import craftutils.params as p
from craftutils.retrieve import save_svo_filter


class Instrument:
    def __init__(self, **kwargs):
        self.name = None
        self.svo_facility = None
        self.svo_instrument = None
        if "svo_service" in kwargs:
            svo_dict = kwargs["svo_service"]
            if "facility" in svo_dict:
                self.svo_facility = svo_dict["facility"]
            if "instrument" in svo_dict:
                self.svo_instrument = svo_dict["instrument"]

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "svo_service": {
                "facility": None,
                "instrument": None
            }
        }

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):
        name, param_file, param_dict = p.params_init(param_file)

        if param_dict is None:
            raise FileNotFoundError("Param file missing!")

        instrument = param_dict["instrument"]
        return cls(instrument=instrument, **param_dict)

    @classmethod
    def build_data_path(cls, instrument_name: str):
        return os.path.join(p.data_path, "instruments", instrument_name)

    @classmethod
    def build_param_path(cls, instrument_name: str):
        return os.path.join(p.param_path, "instruments", instrument_name, f"{instrument_name}.yaml")


class Filter:
    def __init__(self, **kwargs):

        self.name = None
        if "name" in kwargs:
            self.name = kwargs["name"]

        self.data_path = None
        self.votable = None

        self.instrument = None
        if "instrument" in kwargs:
            self.instrument = kwargs["instrument"]

        self.lambda_eff = None
        self.lambda_fwhm = None
        self.transmission_table = None
        self.transmission_table_path = None

    def retrieve_from_svo(self):
        save_svo_filter(filter_name=self.name,
                        )

    def load_transmission_table(self):
        pass

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):
        name, param_file, param_dict = p.params_init(param_file)

        if param_dict is None:
            raise FileNotFoundError("Param file missing!")

        instrument = param_dict["instrument"]
        sub_cls = cls.select_child_class(instrument=instrument)
        return sub_cls(**param_dict)

    @classmethod
    def from_params(cls, filter_name: str, instrument_name: str):
        path = cls.build_param_path(instrument_name=instrument_name, filter_name=filter_name)
        return cls.from_file(param_file=path)

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "instrument": None,
            "data_path": os.path.join(p.data_path),
        }
        return default_params

    @classmethod
    def select_child_class(cls, instrument: str):
        if instrument[:3] == "vlt":
            return ESOFilter
        else:
            return Filter

    @classmethod
    def new_yaml(cls, filter_name: str, path: str = None, instrument_name: str = None, quiet: bool = False):
        param_dict = cls.default_params()
        param_dict["name"] = filter_name
        if instrument_name is not None:
            param_dict["data_path"] = cls.build_data_path(instrument_name=instrument_name, filter_name=filter_name)
        if path is not None:
            path = os.path.join(path, filter_name)
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict

    @classmethod
    def build_data_path(cls, instrument_name: str, filter_name: str):
        return os.path.join(Instrument.build_data_path(instrument_name=instrument_name), "filters", filter_name)

    @classmethod
    def build_param_path(cls, instrument_name: str, filter_name: str):
        instrument_path = Instrument.build_param_path(instrument_name=instrument_name)
        instrument_path, _ = os.path.split(instrument_path)
        return os.path.join(
            instrument_path,
            "filters", f"{filter_name}.yaml"
        )


class ESOFilter(Filter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.
