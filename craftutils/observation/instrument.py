import os
from datetime import date
from typing import Union

import astropy.table as table
import astropy.io.votable as votable
import astropy.units as units

import craftutils.params as p
import craftutils.utils as u
from craftutils.retrieve import save_svo_filter, save_fors2_calib


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

        self.filters = {}
        self.gather_filters()

    def __str__(self):
        return str(self.name)

    def gather_filters(self):
        filter_dir = self.guess_filter_dir()
        for file in filter(lambda f: f.endswith(".yaml"), os.listdir(filter_dir)):
            fil = Filter.from_params(filter_name=file[:-5], instrument_name=self.name)
            fil.instrument = self
            self.filters[fil.name] = fil
            if fil.votable_path is None:
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
            param_dict["data_path"] = cls._build_data_path(instrument_name=instrument_name)
        if path is not None:
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict

    @classmethod
    def new_param(cls, instrument_name: str = None, quiet: bool = False, **kwargs):
        path = cls._build_param_path(instrument_name=instrument_name)
        cls.new_yaml(path=path, instrument_name=instrument_name, quiet=quiet, **kwargs)

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):
        u.debug_print(1, "Instrument.from_file(): param_file ==", param_file)
        name, param_file, param_dict = p.params_init(param_file)
        u.debug_print(1, "Instrument.from_file(): name", name)
        if param_dict is None:
            raise FileNotFoundError("Param file missing!")
        return cls(**param_dict)

    @classmethod
    def from_params(cls, instrument_name: str):
        path = cls._build_param_path(instrument_name=instrument_name)
        u.debug_print(2, "Instrument.from_params(): instrument_name ==", instrument_name)
        u.debug_print(2, "Instrument.from_params(): path ==", path)
        return cls.from_file(param_file=path)

    @classmethod
    def _build_data_path(cls, instrument_name: str):
        path = os.path.join(p.data_path, "instruments")
        u.mkdir_check(path)
        path = os.path.join(path, instrument_name)
        u.mkdir_check(path)
        return path

    @classmethod
    def _build_param_dir(cls, instrument_name: str):
        path = os.path.join(p.param_dir, "instruments")
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
        return Filter


class ESOInstrument(Instrument):
    def filter_class(self):
        if self.name == "vlt-fors2":
            return FORS2Filter
        else:
            return Filter


class Filter:
    def __init__(self, **kwargs):

        self.name = None
        if "name" in kwargs:
            self.name = kwargs["name"]
        self.formatted_name = None
        if "formatted_name" in kwargs:
            self.formatted_name = kwargs["formatted_name"]

        self.svo_id = None
        self.svo_instrument = None
        if "svo_service" in kwargs:
            svo = kwargs["svo_service"]
            if "id" in svo:
                self.svo_id = svo["id"]
            if "instrument" in svo:
                self.svo_instrument = svo["instrument"]

        self.data_path = None
        if "data_path" in kwargs:
            self.data_path = kwargs["data_path"]
        u.mkdir_check(self.data_path)
        self.votable = None
        self.votable_path = None

        self.instrument = None
        if "instrument" in kwargs:
            self.instrument = kwargs["instrument"]

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

    def __str__(self):
        return f"{self.instrument}.{self.name}"

    def nice_name(self):
        if self.formatted_name is not None:
            name = self.formatted_name
        else:
            name = self.name
        return name

    def load_instrument(self):
        if isinstance(self.instrument, str):
            self.instrument = Instrument.from_params(self.instrument)
        elif not isinstance(self.instrument, Instrument):
            raise TypeError(f"instrument must be of type Instrument or str, not {type(self.instrument)}")

    def retrieve_from_svo(self):
        self.load_instrument()
        path = os.path.join(self.data_path, f"{self.instrument}_{self.name}_SVOTable.xml")
        if self.svo_instrument is None:
            instrument = self.instrument.svo_instrument
        else:
            instrument = self.svo_instrument
        save_svo_filter(
            facility_name=self.instrument.svo_facility,
            instrument_name=instrument,
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

        lambda_eff_vot = self.votable.get_field_by_id("WavelengthEff")
        self.lambda_eff = lambda_eff_vot.value * lambda_eff_vot.unit
        lambda_fwhm_vot = self.votable.get_field_by_id("FWHM")
        self.lambda_fwhm = lambda_fwhm_vot.value * lambda_fwhm_vot.unit

        self.write_transmission_tables()
        self.update_output_file()

    def write_transmission_tables(self):

        if self.transmission_table_filter is not None:
            if self.transmission_table_filter_path is None:
                self.transmission_table_filter_path = os.path.join(
                    self.data_path,
                    f"{self.instrument}_{self.name}_transmission_filter.ecsv")
            self.transmission_table_filter.write(
                self.transmission_table_filter_path, format="ascii.ecsv", overwrite=True)

        if self.transmission_table_filter_instrument is not None:
            if self.transmission_table_filter_instrument_path is None:
                self.transmission_table_filter_instrument_path = os.path.join(
                    self.data_path,
                    f"{self.instrument}_{self.name}_transmission_filter_instrument.ecsv")
            self.transmission_table_filter_instrument.write(
                self.transmission_table_filter_instrument_path, format="ascii.ecsv", overwrite=True)

        if self.transmission_table_filter_instrument_atmosphere is not None:
            if self.transmission_table_filter_instrument_atmosphere_path is None:
                self.transmission_table_filter_instrument_atmosphere_path = os.path.join(
                    self.data_path,
                    f"{self.instrument}_{self.name}_transmission_filter_instrument_atmosphere.ecsv")
            self.transmission_table_filter_instrument_atmosphere.write(
                self.transmission_table_filter_instrument_atmosphere_path, format="ascii.ecsv", overwrite=True)

        if self.transmission_table_filter_atmosphere is not None:
            if self.transmission_table_filter_atmosphere_path is None:
                self.transmission_table_filter_atmosphere_path = os.path.join(
                    self.data_path,
                    f"{self.instrument}_{self.name}_transmission_filter_atmosphere.ecsv")
            self.transmission_table_filter_atmosphere.write(
                self.transmission_table_filter_atmosphere_path, format="ascii.ecsv", overwrite=True)

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
                self.transmission_table_filter_instrument = table.QTable.read(
                    self.transmission_table_filter_instrument_path)
        if self.transmission_table_filter_instrument_atmosphere_path is not None:
            if force:
                self.transmission_table_filter_instrument_atmosphere = None
            if self.transmission_table_filter_instrument_atmosphere is None:
                self.transmission_table_filter_instrument_atmosphere = table.QTable.read(
                    self.transmission_table_filter_instrument_atmosphere_path)
        if self.transmission_table_filter_atmosphere_path is not None:
            if force:
                self.transmission_table_filter_atmosphere = None
            if self.transmission_table_filter_atmosphere is None:
                self.transmission_table_filter_atmosphere = table.QTable.read(
                    self.transmission_table_filter_atmosphere_path)

    def select_transmission_table(self):
        filter_tables = [
            self.transmission_table_filter_instrument_atmosphere,
            self.transmission_table_filter_instrument,
            self.transmission_table_filter_atmosphere,
            self.transmission_table_filter
        ]
        for tbl in filter_tables:
            if tbl is not None:
                return tbl
        return None

    def _output_dict(self):
        return {
            "lambda_eff": self.lambda_eff,
            "votable_path": self.votable_path,
            "transmission_table_paths": {
                "filter": self.transmission_table_filter_path,
                "filter_instrument": self.transmission_table_filter_instrument_path,
                "filter_instrument_atmosphere": self.transmission_table_filter_instrument_atmosphere_path,
                "filter_atmosphere": self.transmission_table_filter_atmosphere_path},
        }

    def update_output_file(self):
        p.update_output_file(self)

    def load_output_file(self):
        outputs = p.load_output_file(self)
        if outputs is None:
            return
        if "lambda_eff" in outputs:
            self.lambda_eff = outputs["lambda_eff"]
        if "votable_path" in outputs:
            self.votable_path = outputs["votable_path"]
        if "transmission_table_paths" in outputs:
            transmission_table_paths = outputs["transmission_table_paths"]
            if "filter" in transmission_table_paths:
                self.transmission_table_filter_path = transmission_table_paths["filter"]
            if "filter_instrument" in transmission_table_paths:
                self.transmission_table_filter_instrument_path = transmission_table_paths["filter_instrument"]
            if "filter_instrument_atmosphere" in transmission_table_paths:
                self.transmission_table_filter_instrument_atmosphere_path = transmission_table_paths[
                    "filter_instrument_atmosphere"]
            if "filter_atmosphere" in transmission_table_paths:
                self.transmission_table_filter_atmosphere_path = transmission_table_paths[
                    "filter_instrument_atmosphere"]
        return outputs

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):
        name, param_file, param_dict = p.params_init(param_file)

        if param_dict is None:
            raise FileNotFoundError("Param file missing!")

        instrument_name = param_dict["instrument"]
        sub_cls = cls.select_child_class(instrument_name=instrument_name)
        return sub_cls(**param_dict)

    @classmethod
    def from_params(cls, filter_name: str, instrument_name: str):
        path = cls._build_param_path(instrument_name=instrument_name, filter_name=filter_name)
        return cls.from_file(param_file=path)

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "instrument": None,
            "data_path": None,
            "svo_service": {
                "filter_id": None,
                "instrument": None,
            }
        }
        return default_params

    @classmethod
    def select_child_class(cls, instrument_name: str):
        if instrument_name[:3] == "vlt":
            return FORS2Filter
        else:
            return Filter

    @classmethod
    def new_yaml(cls, filter_name: str, instrument_name: str = None, path: str = None, quiet: bool = False,
                 **kwargs):
        param_dict = cls.default_params()
        param_dict["name"] = filter_name
        param_dict["instrument"] = instrument_name
        param_dict.update(kwargs)
        if instrument_name is not None:
            param_dict["data_path"] = cls._build_data_path(instrument_name=instrument_name, filter_name=filter_name)
        if path is not None:
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict

    @classmethod
    def new_param(cls, filter_name: str, instrument_name: str = None, quiet: bool = False, **kwargs):
        path = cls._build_param_path(filter_name=filter_name, instrument_name=instrument_name)
        cls.new_yaml(filter_name=filter_name, path=path, instrument_name=instrument_name, quiet=quiet, **kwargs)

    @classmethod
    def _build_data_path(cls, instrument_name: str, filter_name: str):
        return os.path.join(Instrument._build_data_path(instrument_name=instrument_name), "filters", filter_name)

    @classmethod
    def _build_param_path(cls, instrument_name: str, filter_name: str):
        path = Instrument._build_filter_dir(instrument_name=instrument_name)
        return os.path.join(path, f"{filter_name}.yaml")


class FORS2Filter(Filter):
    qc1_retrievable = ['b_HIGH', 'v_HIGH', 'R_SPECIAL', 'I_BESS']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calibration_table = None
        self.calibration_table_path = None
        self.calibration_table_last_updated = None

    def load_calibration_table(self, force: bool = False):
        if self.calibration_table_path is not None:
            if force:
                self.calibration_table = None
            if self.calibration_table is None:
                self.calibration_table = table.QTable.read(self.calibration_table_path)
        else:
            print("calibration_table could not be loaded because calibration_table_path has not been set.")

    def write_calibration_table(self):
        if self.calibration_table is None:
            u.debug_print(1, "calibration_table not yet loaded.")
        else:
            if self.calibration_table_path is None:
                self.calibration_table_path = os.path.join(
                    self.data_path,
                    f"{self.instrument.name}_{self.name}_calibration_table.ecsv")
            u.debug_print(1, "Writing calibration table to", self.calibration_table_path)
            self.calibration_table.write(self.calibration_table_path, format="ascii.ecsv", overwrite=True)

    def retrieve_calibration_table(self, force=False):

        if self.name in self.qc1_retrievable:
            if self.calibration_table_last_updated != date.today() or force:
                down_path = os.path.join(self.data_path, "fors2_qc.tbl")
                fil = self.name
                if fil == "R_SPECIAL":
                    fil = "R_SPEC"
                save_fors2_calib(
                    output=down_path,
                    fil=fil,
                )
                self.calibration_table = table.QTable.read(down_path, format="ascii")
                self.calibration_table["zeropoint"] *= units.mag
                self.calibration_table["zeropoint_err"] *= units.mag
                self.calibration_table["colour_term"] *= units.mag
                self.calibration_table["colour_term_err"] *= units.mag
                self.calibration_table["extinction"] *= units.mag
                self.calibration_table["extinction_err"] *= units.mag
                self.write_calibration_table()
                self.calibration_table_last_updated = date.today()
            else:
                u.debug_print(1, "Filter calibrations already updated today; skipping.")

        else:
            u.debug_print(1, f"Cannot retrieve calibration table for {self.name}.")

        self.update_output_file()

        return self.calibration_table

    def get_nearest_calib_row(self, mjd: float):
        self.load_calibration_table()
        i, nrst = u.find_nearest(self.calibration_table["mjd_obs"], mjd)
        return self.calibration_table[i]

    def _output_dict(self):
        output_dict = super()._output_dict()
        output_dict.update(
            {
                "calibration_table_path": self.calibration_table_path,
                "calibration_table_last_updated": self.calibration_table_last_updated
            }
        )
        return output_dict

    def load_output_file(self):
        outputs = super().load_output_file()
        if type(outputs) is dict:
            if "calibration_table_path" in outputs:
                self.calibration_table_path = outputs["calibration_table_path"]
            if "calibration_table_last_updated" in outputs:
                self.calibration_table_last_updated = outputs["calibration_table_last_updated"]
        return outputs
