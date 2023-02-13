import numpy as np

import astropy.table as table
import astropy.units as units
import astropy.constants as constants
import astropy.cosmology as cosmology


class SEDModel:
    def __init__(
            self,
            path: str,
            z: float = None
    ):
        self.path = str(path)
        self.z: float = z
        self.data_table: table.QTable = None
        self.luminosity_distance: units.Quantity = None
        self.rest_luminosity = None
        self.distances()
        self.load_data()
        self.shifted_tables = {}

    def load_data(self):
        self.data_table = table.QTable.read(self.path)
        self.sanitise_columns()

    def distances(
            self,
            cosmo: cosmology.LambdaCDM = cosmology.Planck18
    ):
        if self.z is not None:
            self.luminosity_distance = cosmo.luminosity_distance(z=self.z)

    def sanitise_columns(self):
        columns_change = self.columns()
        for col_name in columns_change:
            old_col_name = columns_change[col_name]
            if old_col_name in self.data_table.colnames:
                self.data_table[col_name] = self.data_table[old_col_name]
                self.data_table.remove_column(old_col_name)

        self.data_table["flux_nu"] = self.data_table["flux_nu"].to("Jy")
        self.data_table["wavelength"] = self.data_table["wavelength"].to("AA")

    def frequency_from_wavelength(
            self
    ):
        self.data_table["frequency"] = (constants.c / self.data_table["wavelength"]).to("Hz")

    def luminosity_per_frequency(
            self
    ):
        if "luminosity_lambda" in self.data_table.colnames:
            self.data_table["luminosity_nu"] = (
                    self.data_table["luminosity_lambda"] * self.data_table["wavelength"] ** 2 / constants.c).to(
                "W Hz^-1")
        elif "flux_lambda" in self.data_table.colnames:
            self.data_table["luminosity_lambda"] = (
                    self.data_table["flux_lambda"] * self._flux_area())

    def _flux_area(self):
        return 4 * np.pi * self.luminosity_distance ** 2

    def flux_per_wavelength(
            self,
            tbl=None
    ):
        if tbl is None:
            tbl = self.data_table
        if "luminosity_lambda" in self.data_table.colnames:
            tbl["flux_lambda"] = (tbl["luminosity_lambda"] / self._flux_area()).to("W nm-1 m-2")
        elif "flux_nu" in self.data_table.colnames:
            tbl["flux_lambda"] = tbl["flux_nu"] * tbl["frequency"] ** 2 / constants.c

    def calculate_rest_luminosity(self):
        self._d_nu_d_lambda()
        tbl = self.data_table["wavelength", "frequency", "luminosity_lambda", "luminosity_nu"].copy()
        tbl["wavelength"] = self.redshift_wavelength(z_shift=0)
        tbl["frequency"] = self.redshift_frequency(z_shift=0)
        self._d_nu_d_lambda(tbl=tbl)
        tbl["luminosity_lambda_total"] = tbl["luminosity_lambda_d_lambda"] / tbl["d_lambda"]
        tbl["luminosity_nu_total"] = tbl["luminosity_nu_d_nu"] / tbl["d_nu"]
        self.rest_luminosity = tbl
        return tbl

    def redshift_wavelength(self, z_shift: float):
        return self.data_table["wavelength"] / ((1 + self.z) / (1 + z_shift))

    def redshift_frequency(self, z_shift: float):
        return self.data_table["frequency"] / ((1 + z_shift) / (1 + self.z))

    def _d_nu_d_lambda(self, tbl: table.QTable = None):
        if tbl is None:
            tbl = self.data_table
        d_lambda = tbl["wavelength"][1:] - tbl["wavelength"][:-1]
        d_lambda = np.append(0 * units.AA, d_lambda)
        tbl["d_lambda"] = d_lambda

        d_nu = tbl["frequency"][1:] - tbl["frequency"][:-1]
        d_nu = np.append(0 * units.Hz, d_nu)
        tbl["d_nu"] = d_nu
        tbl["flux_nu_d_nu"] = self.data_table["flux_nu"] * d_nu
        tbl["luminosity_lambda_d_lambda"] = (tbl["luminosity_lambda_total"] * d_lambda).to(units.W)

    def move_to_redshift(self, z: float = 0):
        self._d_nu_d_lambda()

        new_tbl = self.data_table.copy()
        new_tbl["wavelength"] = new_tbl["wavelength"] / (1 + z)

        self.shifted_tables[z] = new_tbl

    @classmethod
    def columns(cls):
        return {}
