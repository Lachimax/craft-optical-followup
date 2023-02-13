import numpy as np

import astropy.table as table
import astropy.units as units
import astropy.constants as constants
import astropy.cosmology as cosmology

import craftutils.observation.filters as filters


class SEDModel:
    default_cosmology = cosmology.Planck18

    def __init__(
            self,
            path: str,
            z: float,
            **kwargs
    ):
        self.path = str(path)
        self.z: float = z
        self.model_table: table.QTable = None
        self.obs_table: table.QTable = None
        self.luminosity_distance: units.Quantity = None
        self.rest_luminosity = None
        self.cosmology: cosmology.LambdaCDM = self.default_cosmology
        if "cosmology" in kwargs and isinstance(kwargs["cosmology"], cosmology.LambdaCDM):
            self.cosmology = kwargs["cosmology"]
        self.distances()
        self.load_data()
        self.shifted_tables = {}

    def __getitem__(self, *items):
        if len(items) == 1:
            return self.model_table[items[0]]
        else:
            return self.model_table[items]

    def __setitem__(self, key, value):
        self.model_table[key] = value

    def load_data(self):
        self.model_table = table.QTable.read(self.path)
        self.prep_columns()

    def prep_columns(self):
        self.sanitise_columns()
        self.frequency_from_wavelength()
        self.luminosity_per_frequency()
        self.flux_per_wavelength()

    def distances(
            self,
    ):
        if self.z is not None:
            self.luminosity_distance = self.cosmology.luminosity_distance(z=self.z)

    def sanitise_columns(self):
        columns_change = self.columns()
        for col_name in columns_change:
            old_col_name = columns_change[col_name]
            if old_col_name in self.model_table.colnames:
                self.model_table[col_name] = self.model_table[old_col_name]
                self.model_table.remove_column(old_col_name)

        self.model_table["flux_nu"] = self.model_table["flux_nu"].to("Jy")
        self.model_table["wavelength"] = self.model_table["wavelength"].to("AA")

    def frequency_from_wavelength(
            self
    ):
        self.model_table["frequency"] = (constants.c / self.model_table["wavelength"]).to("Hz")

    def luminosity_per_frequency(
            self
    ):
        tbl = self.model_table
        if "luminosity_lambda" in tbl.colnames:
            tbl["luminosity_nu"] = (tbl["luminosity_lambda"] * tbl["wavelength"] ** 2 / constants.c).to("W Hz^-1")
        elif "flux_nu" in tbl.colnames:
            tbl["luminosity_nu"] = (tbl["flux_nu"] * self._flux_area())
        return self.model_table["luminosity_nu"]

    def luminosity_per_wavelength(
            self
    ):
        tbl = self.model_table
        if "luminosity_nu" in tbl.colnames:
            tbl["luminosity_lambda"] = (tbl["luminosity_lambda"] * constants.c / tbl["wavelength"] ** 2).to("W Hz^-1")
        elif "flux_lambda" in tbl.colnames:
            tbl["luminosity_lambda"] = (tbl["flux_lambda"] * self._flux_area())

    def _flux_area(self):
        return 4 * np.pi * self.luminosity_distance ** 2

    def flux_per_wavelength(
            self,
            tbl=None
    ):
        if tbl is None:
            tbl = self.model_table
        if "luminosity_lambda" in self.model_table.colnames:
            tbl["flux_lambda"] = (tbl["luminosity_lambda"] / self._flux_area()).to("W nm-1 m-2")
        elif "flux_nu" in self.model_table.colnames:
            tbl["flux_lambda"] = tbl["flux_nu"] * tbl["frequency"] ** 2 / constants.c

    def calculate_rest_luminosity(self):
        self._d_nu_d_lambda()
        tbl = self.model_table["wavelength", "frequency", "luminosity_lambda", "luminosity_nu"].copy()
        tbl["wavelength"] = self.redshift_wavelength(z_shift=0)
        tbl["frequency"] = self.redshift_frequency(z_shift=0)
        self._d_nu_d_lambda(tbl=tbl)
        tbl["luminosity_lambda"] = tbl["luminosity_lambda_d_lambda"] / tbl["d_lambda"]
        tbl["luminosity_nu"] = tbl["luminosity_nu_d_nu"] / tbl["d_nu"]
        self.rest_luminosity = tbl
        return tbl

    def redshift_wavelength(self, z_shift: float):
        return self.model_table["wavelength"] / ((1 + self.z) / (1 + z_shift))

    def redshift_frequency(self, z_shift: float):
        return self.model_table["frequency"] / ((1 + z_shift) / (1 + self.z))

    def redshift_flux_nu(
            self,
            z_shift: float,
    ):
        d_l = self.luminosity_distance
        d_l_shift = self.cosmology.luminosity_distance(z_shift)
        tbl = self.model_table

        return tbl["flux_nu"] * ((1 + z_shift) * d_l ** 2) / ((1 + self.z) * d_l_shift ** 2)

    def redshift_flux_lambda(
            self,
            z_shift: float,
    ):
        d_l = self.luminosity_distance
        d_l_shift = self.cosmology.luminosity_distance(z_shift)
        tbl = self.model_table

        return tbl["flux_lambda"] * ((1 + self.z) * d_l ** 2) / ((1 + z_shift) * d_l_shift ** 2)

    def _d_nu_d_lambda(self, tbl: table.QTable = None):
        if tbl is None:
            tbl = self.model_table
        d_lambda = tbl["wavelength"][1:] - tbl["wavelength"][:-1]
        d_lambda = np.append(0 * units.AA, d_lambda)
        tbl["d_lambda"] = d_lambda

        d_nu = tbl["frequency"][1:] - tbl["frequency"][:-1]
        d_nu = np.append(0 * units.Hz, d_nu)
        tbl["d_nu"] = d_nu
        tbl["flux_nu_d_nu"] = self.model_table["flux_nu"] * d_nu
        tbl["luminosity_lambda_d_lambda"] = (tbl["luminosity_lambda"] * d_lambda).to(units.W)

    def move_to_redshift(self, z: float = 1.):
        self._d_nu_d_lambda()

        new_tbl = self.model_table.copy()
        new_tbl["flux_nu"] = self.redshift_flux_nu(z_shift=z)
        self._d_nu_d_lambda(new_tbl)
        new_tbl["wavelength"] = self.redshift_wavelength(z_shift=z)
        new_tbl["frequency"] = self.redshift_frequency(z_shift=z)
        new_tbl["flux_nu"] = new_tbl["flux_nu_d_nu"] / new_tbl["d_nu"]

        self.flux_per_wavelength(new_tbl)

        self.shifted_tables[z] = new_tbl
        return new_tbl

    def magnitude_AB(
            self,
            fil: filters.Filter
    ):
        pass

    @classmethod
    def columns(cls):
        return {}
