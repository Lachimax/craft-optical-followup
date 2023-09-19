import os

import numpy as np
from typing import Union

import astropy.table as table
import astropy.io.votable as votable
import astropy.units as units
import astropy.constants as constants

import craftutils.utils as u
import craftutils.params as p
from craftutils.retrieve import save_svo_filter
import craftutils.observation.instrument as instrument
import craftutils.photometry as ph

active_filters = {}

__all__ = []


@u.export
class Filter:

    def __init__(self, **kwargs):

        self.name = None
        if "name" in kwargs:
            self.name = kwargs["name"]
        self.formatted_name = None
        if "formatted_name" in kwargs:
            self.formatted_name = kwargs["formatted_name"]
        self.band_name = None
        if "band_name" in kwargs and kwargs["band_name"] is not None:
            self.band_name = kwargs["band_name"]
        elif self.name is not None:
            self.band_name = self.name[0]

        self.frb_repo_name = None
        if "frb_repo_name" in kwargs:
            self.frb_repo_name = kwargs["frb_repo_name"]

        self.svo_id = []
        self.svo_instrument = None
        if "svo_service" in kwargs:
            svo = kwargs["svo_service"]
            if "filter_id" in svo:
                ids = svo["filter_id"]
            elif "id" in svo:
                ids = svo["id"]
            else:
                ids = []
            if isinstance(ids, list):
                self.svo_id += ids
            else:
                self.svo_id.append(ids)
            if "instrument" in svo:
                self.svo_instrument = svo["instrument"]

        self.data_path = None
        if "data_path" in kwargs:
            if u.is_path_absolute(kwargs["data_path"]):
                self.data_path = kwargs["data_path"]
            else:
                self.data_path = os.path.join(p.data_dir, kwargs["data_path"])
            # print(self.data_path)
            u.mkdir_check_nested(self.data_path, remove_last=False)
        self.votable = None
        self.votable_path = None

        self.instrument = None
        if "instrument" in kwargs:
            self.instrument = kwargs["instrument"]

        self.cmap = None
        if "cmap" in kwargs:
            self.cmap = kwargs["cmap"]

        self.lambda_eff = None
        self.lambda_fwhm = None
        self.vega_zeropoint = None
        self.transmission_tables = {}
        self.transmission_table_paths = {}

        self.photometry_table = None

        self.load_output_file()

        active_filters[f"{self.instrument}_{self.name}"] = self

    def __str__(self):
        return f"{self.instrument}.{self.name}"

    def vega_magnitude_offset(
            self,
            transmission: table.QTable = None
    ):
        if transmission is None:
            transmission, _ = self.select_transmission_table()
        delta_mag = 2.5 * np.log10(
            self.ab_flux(transmission=transmission) /
            self.vega_flux(transmission=transmission)
        )
        return delta_mag * units.mag

    def vega_flux(
            self,
            transmission: table.QTable = None
    ):
        """


        :param transmission:
        :return:
        """
        if transmission is None:
            transmission, _ = self.select_transmission_table()
        return ph.flux_from_band(
            flux=self.vega_zeropoint,
            transmission=transmission["Transmission"],
            frequency=transmission["Frequency"],
            use_quantum_factor=True
        )

    def ab_flux(self, transmission: table.QTable = None):
        """
        Calculates the total integrated flux of the flat AB source (3631 Jy) as seen through the filter.

        :param transmission: transmission table to use. If `None`, `select_transmission_table()` will be used to select
            the 'best' one.
        :return:
        """
        if transmission is None:
            transmission, _ = self.select_transmission_table()
        return ph.flux_ab(
            transmission=transmission["Transmission"],
            frequency=transmission["Frequency"],
        )

    def compare_transmissions(self, other: 'Filter'):
        tbl_self, tbl_other = self.find_comparable_table(other)
        self_wavelength = tbl_self["Wavelength"]
        other_wavelength = tbl_other["Wavelength"]
        self_transmission = tbl_self["Transmission"]
        other_transmission = tbl_other["Transmission"]

        # Pad out the arrays to the same extents
        delta = self_wavelength[-1] - self_wavelength[-2]
        while self_wavelength[-1] < other_wavelength[-1]:
            self_wavelength = np.append(self_wavelength, self_wavelength[-1] + delta)
        while self_wavelength[0] > other_wavelength[0]:
            self_wavelength = np.append(self_wavelength[0] - delta, self_wavelength)

        delta = other_wavelength[-1] - other_wavelength[-2]
        while other_wavelength[-1] < self_wavelength[-1]:
            other_wavelength = np.append(other_wavelength, other_wavelength[-1] + delta)
        while other_wavelength[0] > self_wavelength[0]:
            other_wavelength = np.append(other_wavelength[0] - delta, other_wavelength)

        other_transmission = np.interp(
            x=self_wavelength,
            xp=other_wavelength,
            fp=other_transmission
        ) * max(self_transmission) / max(other_transmission)

        difference = np.sum(np.abs(other_transmission - self_transmission))
        return difference

    def interp_to_wavelength(
            self,
            wavelengths: units.Quantity,
            table_name: str = None,
    ):
        if table_name is None:
            tbl, _ = self.select_transmission_table()
        else:
            tbl = self.transmission_tables[table_name]
        tbl = tbl.copy()
        tbl.sort("Wavelength")
        # Find the difference between wavelength entries in the table
        avg_delta = np.median(np.diff(tbl["Wavelength"]))
        # Pad the transmission table with "0" on either side so that the interpolation goes to zero.
        tbl.add_row(tbl[0])
        tbl[-1]["Wavelength"] = tbl["Wavelength"].min() - avg_delta
        tbl[-1]["Transmission"] = 0
        tbl.add_row(tbl[0])
        tbl[-1]["Wavelength"] = tbl["Wavelength"].max() + avg_delta
        tbl[-1]["Transmission"] = 0
        tbl.sort("Wavelength")
        # Interpolate the transmission table at the wavelength values given in the model table.
        # print(wavelengths, tbl["Wavelength"], tbl["Transmission"])
        # print(len(wavelengths), len(tbl["Wavelength"]), len(tbl["Transmission"]))
        wavelengths = wavelengths.to("AA")
        return np.interp(
            x=wavelengths.value,
            xp=tbl["Wavelength"].value,
            fp=tbl["Transmission"].value
        )

    def compare_wavelength_range(self, other: 'Filter'):
        tbl_self, tbl_other = self.find_comparable_table(other)
        self_wavelength = tbl_self["Wavelength"]
        other_wavelength = tbl_other["Wavelength"]
        self_transmission = tbl_self["Transmission"]
        other_transmission = tbl_other["Transmission"]

        u.debug_print(2, min(self_wavelength[self_transmission > 0]), min(other_wavelength[other_transmission > 0]))

        difference = np.abs(
            min(self_wavelength[self_transmission > 0]) - min(other_wavelength[other_transmission > 0])) + np.abs(
            max(self_wavelength[self_transmission > 0]) - max(other_wavelength[other_transmission > 0]))

        return difference

    def find_comparable_table(self, other: 'Filter'):

        tbls_self, _ = self.transmission_by_usefulness()
        tbls_other, _ = other.transmission_by_usefulness()

        for i, tbl in enumerate(tbls_self):
            if tbl is not None and tbls_other[i] is not None:
                return tbl, tbls_other[i]

        for i, tbl_self in enumerate(tbls_self):
            if tbl_self is not None:
                for j, tbl_other in enumerate(tbls_other):
                    if tbl_other is not None:
                        return tbl_self, tbl_other

        return None, None

    def nice_name(self):
        if self.formatted_name is not None:
            name = self.formatted_name
        else:
            name = self.name
        return name

    def machine_name(self):
        return f"{self.instrument.name}_{self.name.replace('_', '-')}"

    def load_instrument(self):
        if isinstance(self.instrument, str):
            self.instrument = instrument.Instrument.from_params(self.instrument)
        elif not isinstance(self.instrument, instrument.Instrument):
            raise TypeError(f"instrument must be of type Instrument or str, not {type(self.instrument)}")

    def retrieve_from_svo(self):
        """

        :return:
        """
        self.load_instrument()
        if self.svo_instrument is None:
            inst = self.instrument.svo_instrument
        else:
            inst = self.svo_instrument

        for svo_id in self.svo_id:
            path = os.path.join(self.data_path, f"{self.instrument}_{svo_id}_SVOTable.xml")
            save_svo_filter(
                facility_name=self.instrument.svo_facility,
                instrument_name=inst,
                filter_name=svo_id,
                output=path
            )
            self.votable = votable.parse(path)
            self.votable_path = path

            components = self.votable.get_field_by_id("components").value

            key = components.replace(" ", "").lower()
            self.transmission_tables[key] = table.QTable(self.votable.get_first_table().to_table())

            lambda_eff_vot = self.votable.get_field_by_id("WavelengthEff")
            self.lambda_eff = lambda_eff_vot.value * lambda_eff_vot.unit
            lambda_fwhm_vot = self.votable.get_field_by_id("FWHM")
            self.lambda_fwhm = lambda_fwhm_vot.value * lambda_fwhm_vot.unit
            zp_vega = self.votable.get_field_by_id("ZeroPoint")
            self.vega_zeropoint = zp_vega.value * zp_vega.unit

        self.write_transmission_tables()
        self.update_output_file()

    def write_transmission_tables(self):

        for table_name in self.transmission_tables:
            if self.transmission_tables[table_name] is not None:
                tbl = self.transmission_tables[table_name]
                if table_name not in self.transmission_table_paths or self.transmission_table_paths[table_name] is None:
                    self.transmission_table_paths[table_name] = os.path.join(
                        self.data_path,
                        f"{self.instrument}_{self.name}_transmission_{table_name}.ecsv"
                    )
                tbl.write(self.transmission_table_paths[table_name], overwrite=True)

    def load_transmission_tables(self, force: bool = False):

        for table_name in self.transmission_table_paths:
            path = self.transmission_table_paths[table_name]
            if path is None:
                continue
            elif not os.path.isfile(path):
                self.transmission_table_paths[table_name] = None
                continue
            if force:
                self.transmission_tables[table_name] = None
            if table_name not in self.transmission_tables or self.transmission_tables[table_name] is None:
                tbl = table.QTable.read(path)
                tbl["Frequency"] = (constants.c / tbl["Wavelength"]).to(units.Hz)
                self.transmission_tables[table_name] = tbl

        return self.transmission_tables

    def transmission_by_usefulness(self):

        self.load_transmission_tables()
        tbl_names = [
            "filter",
            "filter+atmosphere",
            "filter+instrument",
            "filter+ccd+instrument",
            "filter+instrument+atmosphere"
        ]
        tbl_names.reverse()
        for name in self.transmission_tables:
            if name not in tbl_names:
                tbl_names.append(name)

        tbls = []
        names = []
        for name in tbl_names:
            if name in self.transmission_tables:
                tbls.append(self.transmission_tables[name])
                names.append(name)
        return tbls, names

    def select_transmission_table(self):
        filter_tables, table_names = self.transmission_by_usefulness()
        for i, tbl in enumerate(filter_tables):
            if tbl is not None:
                return tbl, table_names[i]
        return None, None

    def _output_dict(self):
        return {
            "lambda_eff": self.lambda_eff,
            "lambda_fwhm": self.lambda_fwhm,
            "vega_zeropoint": self.vega_zeropoint,
            "votable_path": self.votable_path,
            "transmission_table_paths": self.transmission_table_paths
        }

    def update_output_file(self):
        p.update_output_file(self)

    def load_output_file(self):
        outputs = p.load_output_file(self)
        if outputs is None:
            return
        if "lambda_eff" in outputs:
            self.lambda_eff = outputs["lambda_eff"]
        if "lambda_fwhm" in outputs:
            self.lambda_fwhm = outputs["lambda_fwhm"]
        if "lambda_fwhm" in outputs:
            self.vega_zeropoint = outputs["vega_zeropoint"]
        if "votable_path" in outputs:
            self.votable_path = outputs["votable_path"]
        if "transmission_table_paths" in outputs:
            self.transmission_table_paths = outputs["transmission_table_paths"]
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
        band_str = f"{instrument_name}_{filter_name}"
        if band_str in active_filters:
            return active_filters[band_str]
        path = cls._build_param_path(instrument_name=instrument_name, filter_name=filter_name)
        return cls.from_file(param_file=path)

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "band_name": None,
            "formatted_name": None,
            "instrument": None,
            "data_path": None,
            "svo_service": {
                "filter_id": None,
                "instrument": None,
            },
        }
        return default_params

    @classmethod
    def select_child_class(cls, instrument_name: str):
        if instrument_name[:3] == "vlt":
            from craftutils.observation.filters import FORS2Filter
            return FORS2Filter
        else:
            return Filter

    @classmethod
    def new_yaml(cls, filter_name: str, instrument_name: str = None, path: str = None,
                 **kwargs):
        param_dict = cls.default_params()
        param_dict["name"] = filter_name
        param_dict["instrument"] = instrument_name
        param_dict.update(kwargs)
        if instrument_name is not None:
            param_dict["data_path"] = cls._build_data_path(
                instrument_name=instrument_name,
                filter_name=filter_name
            )
        if path is not None:
            p.save_params(file=path, dictionary=param_dict)
        return param_dict

    @classmethod
    def new_param(cls, filter_name: str, instrument_name: str = None, **kwargs):
        path = cls._build_param_path(filter_name=filter_name, instrument_name=instrument_name)
        cls.new_yaml(filter_name=filter_name, path=path, instrument_name=instrument_name, **kwargs)

    @classmethod
    def _build_data_path(cls, instrument_name: str, filter_name: str):
        return os.path.join(instrument.Instrument._build_data_path(instrument_name=instrument_name), "filters",
                            filter_name)

    @classmethod
    def _build_param_path(cls, instrument_name: str, filter_name: str):
        path = instrument.Instrument._build_filter_dir(instrument_name=instrument_name)
        return os.path.join(path, f"{filter_name}.yaml")

    @classmethod
    def _build_photometry_table_path(cls, instrument_name: str, filter_name: str):
        return os.path.join(
            cls._build_data_path(
                instrument_name=instrument_name,
                filter_name=filter_name,
            ),
            f"{filter_name}_photometry.ecsv"
        )
