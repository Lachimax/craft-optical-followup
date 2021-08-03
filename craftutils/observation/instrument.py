import os
from typing import Union

import astropy.table as table
import astropy.io.votable as votable

import craftutils.params as p
import craftutils.utils as u
from craftutils.retrieve import save_svo_filter


class Instrument:
    def __init__(self, **kwargs):
        self.name = None
        if "name" in kwargs:
            self.name = kwargs["name"]
        self.svo_facility = None
        self.svo_instrument = None
        if "svo_service" in kwargs:
            svo_dict = kwargs["svo_service"]
            if "facility" in svo_dict:
                self.svo_facility = svo_dict["facility"]
            if "instrument" in svo_dict:
                self.svo_instrument = svo_dict["instrument"]

    def __str__(self):
        return str(self.name)

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "svo_service": {
                "facility": None,
                "instrument": None
            }
        }
        return default_params

    @classmethod
    def new_yaml(cls, instrument_name: str = None, path: str = None, quiet: bool = False, **kwargs):
        param_dict = cls.default_params()
        param_dict["name"] = instrument_name
        param_dict.update(kwargs)
        if instrument_name is not None:
            param_dict["data_path"] = cls.build_data_path(instrument_name=instrument_name)
        if path is not None:
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict

    @classmethod
    def new_param(cls, instrument_name: str = None, quiet: bool = False, **kwargs):
        path = cls.build_param_path(instrument_name=instrument_name)
        cls.new_yaml(path=path, instrument_name=instrument_name, quiet=quiet, **kwargs)

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):
        name, param_file, param_dict = p.params_init(param_file)

        if param_dict is None:
            raise FileNotFoundError("Param file missing!")
        return cls(**param_dict)

    @classmethod
    def from_params(cls, instrument_name: str):
        path = cls.build_param_path(instrument_name=instrument_name)
        return cls.from_file(param_file=path)

    @classmethod
    def build_data_path(cls, instrument_name: str):
        path = os.path.join(p.data_path, "instruments")
        u.mkdir_check(path)
        path = os.path.join(path, instrument_name)
        u.mkdir_check(path)
        return path

    @classmethod
    def build_param_path(cls, instrument_name: str):
        path = os.path.join(p.param_path, "instruments")
        u.mkdir_check(path)
        path = os.path.join(path, instrument_name)
        u.mkdir_check(path)
        return os.path.join(path, f"{instrument_name}.yaml")


class Filter:
    def __init__(self, **kwargs):

        self.name = None
        if "name" in kwargs:
            self.name = kwargs["name"]

        self.svo_id = None
        if "svo_id" in kwargs:
            self.svo_id = kwargs["svo_id"]

        self.data_path = None
        if "data_path" in kwargs:
            self.data_path = kwargs["data_path"]
        self.votable = None
        self.votable_path = None

        self.instrument = None
        if "instrument" in kwargs:
            self.instrument = kwargs["instrument"]
            if isinstance(self.instrument, str):
                self.instrument = Instrument.from_params(self.instrument)

        self.lambda_eff = None
        self.lambda_fwhm = None
        self.transmission_table_filter = None
        self.transmission_table_filter_path = None
        self.transmission_table_filter_instrument = None
        self.transmission_table_filter_instrument_path = None
        self.transmission_table_filter_instrument_atmosphere = None
        self.transmission_table_filter_instrument_atmosphere_path = None
        self.transmission_table_filter_atmosphere = None
        self.transmission_table_filter_atmosphere_path = None

        self.load_output_file()

    def retrieve_from_svo(self):
        path = os.path.join(self.data_path, f"{self.instrument}_{self.name}_SVOTable.xml")
        save_svo_filter(
            facility_name=self.instrument.svo_facility,
            instrument_name=self.instrument.svo_instrument,
            filter_name=self.svo_id,
            output=path
        )
        self.votable = votable.parse(path)
        self.votable_path = path

        components = self.votable.get_field_by_id("components").value
        if components == "Filter":
            self.transmission_table_filter = self.votable.get_first_table().to_table()
        elif components == "Filter + Instrument":
            self.transmission_table_filter_instrument = self.votable.get_first_table().to_table()
        elif components == "Filter + Instrument + Atmosphere":
            self.transmission_table_filter_instrument_atmosphere = self.votable.get_first_table().to_table()
        elif components == "Filter + Atmosphere":
            self.transmission_table_filter_atmosphere = self.votable.get_first_table().to_table()

        self.lambda_eff = self.votable.get_field_by_id("WavelengthEff")

        self.write_transmission_tables()
        self.update_output_file()

    def write_transmission_tables(self):
        if self.transmission_table_filter_path is None:
            self.transmission_table_filter_path = os.path.join(
                self.data_path,
                f"{self.instrument}_{self.name}_transmission_filter.ecsv")
        if self.transmission_table_filter is not None:
            self.transmission_table_filter.write(
                self.transmission_table_filter_path, format="ascii.ecsv")

        if self.transmission_table_filter_instrument_path is None:
            self.transmission_table_filter_instrument_path = os.path.join(
                self.data_path,
                f"{self.instrument}_{self.name}_transmission_filter_instrument.ecsv")
        if self.transmission_table_filter_instrument is not None:
            self.transmission_table_filter_instrument.write(
                self.transmission_table_filter_instrument_path, format="ascii.ecsv")

        if self.transmission_table_filter_instrument_atmosphere_path is None:
            self.transmission_table_filter_instrument_atmosphere_path = os.path.join(
                self.data_path,
                f"{self.instrument}_{self.name}_transmission_filter_instrument_atmosphere.ecsv")
        if self.transmission_table_filter_instrument_atmosphere is not None:
            self.transmission_table_filter_instrument_atmosphere.write(
                self.transmission_table_filter_instrument_atmosphere_path, format="ascii.ecsv")

        if self.transmission_table_filter_atmosphere_path is None:
            self.transmission_table_filter_atmosphere_path = os.path.join(
                self.data_path,
                f"{self.instrument}_{self.name}_transmission_filter_atmosphere.ecsv")
        if self.transmission_table_filter_atmosphere is not None:
            self.transmission_table_filter_atmosphere.write(
                self.transmission_table_filter_atmosphere_path, format="ascii.ecsv")

    def load_transmission_tables(self, force: bool = False):
        if self.transmission_table_filter_path is not None:
            if force:
                self.transmission_table_filter = None
            if self.transmission_table_filter is None:
                self.transmission_table_filter = table.QTable.read(self.transmission_table_filter_path)
        if self.transmission_table_filter_instrument_path is not None:
            if force:
                self.transmission_table_filter_instrument = None
            if self.transmission_table_filter_instrument is None:
                self.transmission_table_filter_instrument = table.QTable.read(self.transmission_table_filter_instrument_path)
        if self.transmission_table_filter_instrument_atmosphere_path is not None:
            if force:
                self.transmission_table_filter_instrument_atmosphere = None
            if self.transmission_table_filter_instrument_atmosphere is None:
                self.transmission_table_filter_instrument_atmosphere = table.QTable.read(self.transmission_table_filter_instrument_atmosphere_path)
        if self.transmission_table_filter_atmosphere_path is not None:
            if force:
                self.transmission_table_filter_atmosphere = None
            if self.transmission_table_filter_atmosphere is None:
                self.transmission_table_filter_atmosphere = table.QTable.read(self.transmission_table_filter_atmosphere_path)

    def _output_dict(self):
        return {
            "votable_path": self.votable_path,
            "transmission_table_filter_path": self.transmission_table_filter_path,
            "transmission_table_filter_instrument_path": self.transmission_table_filter_path,
            "transmission_table_filter_instrument_atmosphere_path": self.transmission_table_filter_path,
            "transmission_table_filter_atmospherepath": self.transmission_table_filter_path,
        }

    def update_output_file(self):
        p.update_output_file(self)

    def load_output_file(self):
        outputs = p.load_output_file(self)

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
            "data_path": None,
            "svo_id": None,
        }
        return default_params

    @classmethod
    def select_child_class(cls, instrument: str):
        if instrument[:3] == "vlt":
            return ESOFilter
        else:
            return Filter

    @classmethod
    def new_yaml(cls, filter_name: str, instrument_name: str = None, path: str = None, quiet: bool = False, **kwargs):
        param_dict = cls.default_params()
        param_dict["name"] = filter_name
        param_dict["instrument"] = instrument_name
        param_dict.update(kwargs)
        if instrument_name is not None:
            param_dict["data_path"] = cls.build_data_path(instrument_name=instrument_name, filter_name=filter_name)
        if path is not None:
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict

    @classmethod
    def new_param(cls, filter_name: str, instrument_name: str = None, quiet: bool = False, **kwargs):
        path = cls.build_param_path(filter_name=filter_name, instrument_name=instrument_name)
        cls.new_yaml(filter_name=filter_name, path=path, instrument_name=instrument_name, quiet=quiet, **kwargs)

    @classmethod
    def build_data_path(cls, instrument_name: str, filter_name: str):
        return os.path.join(Instrument.build_data_path(instrument_name=instrument_name), "filters", filter_name)

    @classmethod
    def build_param_path(cls, instrument_name: str, filter_name: str):
        instrument_path = Instrument.build_param_path(instrument_name=instrument_name)
        instrument_path, _ = os.path.split(instrument_path)
        return os.path.join(instrument_path, "filters", f"{filter_name}.yaml")


class ESOFilter(Filter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.
