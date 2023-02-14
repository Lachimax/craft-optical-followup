import numpy as np

import astropy.table as table
import astropy.units as units
import astropy.constants as constants
import astropy.cosmology as cosmology

import craftutils.observation.filters as filters
from craftutils.photometry import magnitude_AB
import craftutils.utils as u


@u.export
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
            self,
            z: float = None
    ):
        tbl = self.get_table(z=z)
        if "luminosity_lambda" in tbl.colnames:
            tbl["luminosity_nu"] = (tbl["luminosity_lambda"] * tbl["wavelength"] ** 2 / constants.c).to("W Hz-1")
        elif "flux_nu" in tbl.colnames:
            tbl["luminosity_nu"] = (tbl["flux_nu"] * self._flux_area())
        return self.model_table["luminosity_nu"]

    def luminosity_per_wavelength(
            self,
            z: float = None
    ):
        tbl = self.get_table(z=z)
        if "luminosity_nu" in tbl.colnames:
            tbl["luminosity_lambda"] = (tbl["luminosity_nu"] * constants.c / tbl["wavelength"] ** 2).to("W AA-1")
        elif "flux_lambda" in tbl.colnames:
            tbl["luminosity_lambda"] = (tbl["flux_lambda"] * self._flux_area())
        return tbl["luminosity_lambda"]

    def _flux_area(self):
        return 4 * np.pi * self.luminosity_distance ** 2

    def flux_per_wavelength(
            self,
            z: float = None
    ):
        tbl = self.get_table(z=z)
        if "luminosity_lambda" in self.model_table.colnames:
            tbl["flux_lambda"] = (tbl["luminosity_lambda"] / self._flux_area()).to("W nm-1 m-2")
        elif "flux_nu" in self.model_table.colnames:
            tbl["flux_lambda"] = tbl["flux_nu"] * tbl["frequency"] ** 2 / constants.c

    def calculate_rest_luminosity(self):
        self._d_nu_d_lambda()
        tbl = self.model_table["wavelength", "frequency", "luminosity_lambda", "luminosity_nu"].copy()
        tbl["wavelength"] = self.redshift_wavelength(z_shift=0)
        tbl["frequency"] = self.redshift_frequency(z_shift=0)
        self._d_nu_d_lambda(z=0)
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

    def _d_nu_d_lambda(self, z: float = None):
        tbl = self.get_table(z=z)
        d_lambda = tbl["wavelength"][1:] - tbl["wavelength"][:-1]
        d_lambda = np.append(0 * units.AA, d_lambda)
        tbl["d_lambda"] = d_lambda

        d_nu = tbl["frequency"][1:] - tbl["frequency"][:-1]
        d_nu = np.append(d_nu.min(), d_nu)
        tbl["d_nu"] = d_nu
        tbl["flux_nu_d_nu"] = self.model_table["flux_nu"] * d_nu
        print("Here 2")
        if "luminosity_lambda" in tbl.colnames:
            tbl["luminosity_lambda_d_lambda"] = (tbl["luminosity_lambda"] * d_lambda).to(units.W)

    def move_to_redshift(self, z: float = 1.):
        self._d_nu_d_lambda()
        new_tbl = self.model_table.copy()
        new_tbl["flux_nu"] = self.redshift_flux_nu(z_shift=z)
        print("Here 1")
        self.shifted_tables[z] = new_tbl
        self._d_nu_d_lambda(z=z)
        new_tbl["wavelength"] = self.redshift_wavelength(z_shift=z)
        new_tbl["frequency"] = self.redshift_frequency(z_shift=z)
        new_tbl["flux_nu"] = new_tbl["flux_nu_d_nu"] / new_tbl["d_nu"]

        self.flux_per_wavelength(z=z)

        return new_tbl

    # def flux_through_band(
    #         self
    # ):

    def get_table(self, z: float = None):
        if z is None or z == self.z:
            tbl = self.model_table
        elif z == 0.:
            if self.rest_luminosity is None:
                self.calculate_rest_luminosity()
            return self.rest_luminosity
        elif z in self.shifted_tables:
            tbl = self.shifted_tables[z]
        else:
            tbl = self.move_to_redshift(z)
        return tbl

    def magnitude_AB(
            self,
            band: filters.Filter,
            z: float = None,
            use_quantum_factor: bool = True
    ):
        tbl = self.get_table(z=z)
        transmission = self.add_band(band=band, z=z)
        return magnitude_AB(
            flux=tbl["flux_nu"],
            band_transmission=transmission,
            frequency=tbl["frequency"],
            use_quantum_factor=use_quantum_factor
        )

    def add_band(
            self,
            band: filters.Filter,
            z: float = None
    ):
        """
        Regrids a bandpass to the same wavelength coordinates as the data table.
        :param band:
        :return:
        """
        tbl = self.get_table(z=z)
        band_name = f"{band.name}_{band.instrument.name}"
        transmission_tbl, _ = band.select_transmission_table()
        transmission_tbl = transmission_tbl.copy()
        if band_name not in tbl.colnames:
            # Find the difference between wavelength entries in the table
            avg_delta = np.median(transmission_tbl["Wavelength"][1:] - transmission_tbl["Wavelength"][:-1])
            # Pad the transmission table with "0" on either side so that the interpolation goes to zero.
            transmission_tbl.add_row(transmission_tbl[0])
            transmission_tbl[-1]["Wavelength"] = transmission_tbl["Wavelength"].min() - avg_delta
            transmission_tbl[-1]["Transmission"] = 0
            transmission_tbl.add_row(transmission_tbl[0])
            transmission_tbl[-1]["Wavelength"] = transmission_tbl["Wavelength"].max() + avg_delta
            transmission_tbl[-1]["Transmission"] = 0
            transmission_tbl.sort("Wavelength")
            # Interpolate the transmission table at the wavelength values given in the model table.
            tbl[band_name] = np.interp(
                x=tbl[f"wavelength"].value,
                xp=transmission_tbl["Wavelength"].value,
                fp=transmission_tbl["Transmission"].value
            )
            # Get the flux as seen through this band.
            tbl[f"flux_nu_{band_name}"] = tbl[band_name] * tbl["flux_nu"]
            self.shifted_tables[z] = tbl
        return tbl[band_name]

    @classmethod
    def columns(cls):
        return {}
