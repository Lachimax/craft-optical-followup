import os
from typing import Union

import craftutils.params as p
import craftutils.utils as u
import craftutils.observation.filters as filters

active_instruments = {}

__all__ = []


@u.export
class Instrument:

    def __init__(self, **kwargs):
        self.name = None
        if "name" in kwargs:
            self.name = kwargs["name"]
        self.formatted_name = None
        if "formatted_name" in kwargs:
            self.formatted_name = kwargs["formatted_name"]
        self.svo_facility = None
        self.svo_instrument = None
        if "svo_service" in kwargs:
            svo_dict = kwargs["svo_service"]
            if "facility" in svo_dict:
                self.svo_facility = svo_dict["facility"]
            if "instrument" in svo_dict:
                self.svo_instrument = svo_dict["instrument"]

        self.cigale_name = None
        if "cigale_name" in kwargs:
            self.cigale_name = kwargs["cigale_name"]

        self.filters = {}
        self.bands = {}
        self.gather_filters()

        active_instruments[self.name] = self

    def __str__(self):
        return str(self.name)

    def gather_filters(self):
        filter_dir = self.guess_filter_dir()
        for file in filter(lambda f: f.endswith(".yaml"), os.listdir(filter_dir)):
            u.debug_print(1, "Instrument.gather_filters(): file == ", file)
            fil = filters.Filter.from_params(filter_name=file[:-5], instrument_name=self.name)
            fil.instrument = self
            self.filters[fil.name] = fil
            if fil.band_name is not None:
                self.bands[fil.band_name] = fil
            if fil.votable_path is None or not os.path.isfile(fil.votable_path):
                fil.retrieve_from_svo()

    def guess_param_dir(self):
        return self._build_param_dir(instrument_name=self.name)

    def guess_filter_dir(self):
        return self._build_filter_dir(instrument_name=self.name)

    def nice_name(self):
        if self.formatted_name is not None:
            name = self.formatted_name
        else:
            name = self.name
        return name

    def new_filter(
            self,
            filter_name: str
    ):
        """
        Wraps Filter.new_param()
        :return:
        """
        filters.Filter.new_param(
            filter_name=filter_name,
            instrument_name=self.name
        )

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "formatted_name": None,
            "svo_service": {
                "facility": None,
                "instrument": None
            },
            "cigale_name": None
        }
        return default_params

    @classmethod
    def new_yaml(cls, instrument_name: str = None, path: str = None, **kwargs):
        param_dict = cls.default_params()
        param_dict["name"] = instrument_name

        param_dict.update(kwargs)
        if instrument_name is not None:
            param_dict["data_path"] = cls._build_data_path(instrument_name=instrument_name)
        if path is not None:
            p.save_params(file=path, dictionary=param_dict)
        return param_dict

    @classmethod
    def new_param(cls, instrument_name: str = None, **kwargs):
        path = cls._build_param_path(instrument_name=instrument_name)
        cls.new_yaml(instrument_name=instrument_name, path=path, **kwargs)

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):
        u.debug_print(1, "Instrument.from_file(): param_file ==", param_file)
        name, param_file, param_dict = p.params_init(param_file)
        u.debug_print(1, "Instrument.from_file(): name ==", name)
        u.debug_print(1, "Instrument.from_file(): param_dict ==", param_dict)
        if param_dict is None:
            raise FileNotFoundError("Param file missing!")
        return cls(**param_dict)

    @classmethod
    def from_params(cls, instrument_name: str):
        if instrument_name in active_instruments:
            return active_instruments[instrument_name]
        path = cls._build_param_path(instrument_name=instrument_name)
        u.debug_print(1, "Instrument.from_params(): instrument_name ==", instrument_name)
        u.debug_print(1, "Instrument.from_params(): path ==", path)
        return cls.from_file(param_file=path)

    @classmethod
    def _build_data_path(cls, instrument_name: str):
        # path = os.path.join(p.data_dir, "instruments")
        # u.mkdir_check(path)
        path = os.path.join("gemini", instrument_name)
        # u.mkdir_check(path)
        return path

    @classmethod
    def _build_param_dir(cls, instrument_name: str):
        path = os.path.join(p.param_dir, "gemini")
        u.mkdir_check(path)
        u.debug_print(2, "Instrument._build_param_dir(): instrument_name ==", instrument_name)
        path = os.path.join(path, instrument_name)
        u.mkdir_check(path)
        return path

    @classmethod
    def _build_filter_dir(cls, instrument_name: str):
        path = cls._build_param_dir(instrument_name=instrument_name)
        path = os.path.join(path, "filters")
        u.mkdir_check(path)
        return path

    @classmethod
    def _build_param_path(cls, instrument_name: str):
        """
        Get default path to an instrument param .yaml file.
        :param instrument_name:
        :return:
        """
        path = cls._build_param_dir(instrument_name=instrument_name)
        return os.path.join(path, f"{instrument_name}.yaml")

    @classmethod
    def filter_class(cls):
        return filters.Filter

