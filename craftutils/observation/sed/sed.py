import numpy as np

import astropy.table as table
import astropy.units as units
import astropy.constants as constants
import astropy.cosmology as cosmology

import craftutils.observation.filters as filters
import craftutils.photometry as ph
import craftutils.utils as u


@u.export
class SEDModel:
    default_cosmology = cosmology.Planck18

    def __init__(
            self,
            z: float = None,
            path: str = None,
            model_table: table.QTable = None,
            **kwargs
    ):
        self.path: str = None
        self.model_table: table.QTable = None
        self.z: float = z
        self.model_table: table.QTable = None
        self.obs_table: table.QTable = None
        self.luminosity_distance: units.Quantity = None
        self.rest_luminosity = None
        self.cosmology: cosmology.LambdaCDM = self.default_cosmology
        if "cosmology" in kwargs and isinstance(kwargs["cosmology"], cosmology.LambdaCDM):
            self.cosmology = kwargs["cosmology"]
        self.shifted_models = {}
        if path is not None:
            self.path = str(path)
            self.load_data()
        elif model_table is not None:
            self.model_table = model_table

    def prep_data(self):
        self.distances()
        self.prep_columns()

    def __getitem__(self, *items):
        if len(items) == 1:
            return self.model_table[items[0]]
        else:
            return self.model_table[items]

    def __setitem__(self, key, value):
        self.model_table[key] = value

    def load_data(self):
        self.model_table = table.QTable.read(self.path)
        self.prep_data()

    def prep_columns(self):
        self.sanitise_columns()
        self.frequency_from_wavelength()
        self.luminosity_per_frequency()
        self.flux_per_wavelength()

    def distances(
            self,
    ):
        print(self.z)
        if self.z is not None:
            self.luminosity_distance = self.cosmology.luminosity_distance(z=self.z)

    def sanitise_columns(
            self,
    ):
        tbl = self.model_table
        columns_change = self.columns()
        for col_name in columns_change:
            old_col_name = columns_change[col_name]
            if old_col_name in tbl.colnames:
                tbl[col_name] = tbl[old_col_name]
                tbl.remove_column(old_col_name)

        tbl["flux_nu"] = tbl["flux_nu"].to("Jy")
        tbl["wavelength"] = tbl["wavelength"].to("AA")

    def frequency_from_wavelength(
            self,
    ):
        tbl = self.model_table
        tbl["frequency"] = (constants.c / tbl["wavelength"]).to("Hz")

    def luminosity_per_frequency(
            self,
    ):
        tbl = self.model_table
        if "luminosity_lambda" in tbl.colnames:
            tbl["luminosity_nu"] = (tbl["luminosity_lambda"] * tbl["wavelength"] ** 2 / constants.c).to("W Hz-1")
        elif "flux_nu" in tbl.colnames:
            tbl["luminosity_nu"] = (tbl["flux_nu"] * self._flux_area())
        self.model_table = tbl
        return tbl["luminosity_nu"]

    def luminosity_per_wavelength(
            self,
    ):
        tbl = self.model_table
        if "luminosity_nu" in tbl.colnames:
            tbl["luminosity_lambda"] = (tbl["luminosity_nu"] * constants.c / tbl["wavelength"] ** 2).to("W AA-1")
        elif "flux_lambda" in tbl.colnames:
            tbl["luminosity_lambda"] = (tbl["flux_lambda"] * self._flux_area())
        return tbl["luminosity_lambda"]

    def _flux_area(self):
        return 4 * np.pi * self.luminosity_distance ** 2

    def flux_per_wavelength(
            self,
    ):
        tbl = self.model_table
        if "luminosity_lambda" in self.model_table.colnames:
            tbl["flux_lambda"] = (tbl["luminosity_lambda"] / self._flux_area()).to("W nm-1 m-2")
        elif "flux_nu" in self.model_table.colnames:
            tbl["flux_lambda"] = tbl["flux_nu"] * tbl["frequency"] ** 2 / constants.c

    def calculate_rest_luminosity(self):
        self.model_table = _d_nu_d_lambda(self.model_table)
        tbl = self.model_table[
            "wavelength", "frequency", "luminosity_lambda", "luminosity_nu", "d_nu", "d_lambda"].copy()
        tbl["luminosity_lambda_d_lambda"] = (tbl["luminosity_lambda"] * tbl["d_lambda"]).to(units.W)
        tbl["luminosity_nu_d_nu"] = (tbl["luminosity_nu"] * tbl["d_nu"]).to(units.W)
        tbl["wavelength"] = self.redshift_wavelength(z_shift=0)
        tbl["frequency"] = self.redshift_frequency(z_shift=0)
        tbl = _d_nu_d_lambda(tbl)
        tbl["luminosity_lambda"] = tbl["luminosity_lambda_d_lambda"] / tbl["d_lambda"]
        tbl["luminosity_nu"] = tbl["luminosity_nu_d_nu"] / tbl["d_nu"]
        self.rest_luminosity = tbl
        return tbl

    def redshift_wavelength(self, z_shift: float):
        return ph.redshift_frequency(nu=self.model_table["wavelength"], z=self.z, z_new=z_shift)

    def redshift_frequency(self, z_shift: float):
        return ph.redshift_frequency(nu=self.model_table["frequency"], z=self.z, z_new=z_shift)

    def redshift_flux_nu(
            self,
            z_new: float,
    ):
        d_l = self.luminosity_distance
        d_l_shift = self.cosmology.luminosity_distance(z_new)
        tbl = self.model_table

        return tbl["flux_nu"] * ((1 + z_new) * d_l ** 2) / ((1 + self.z) * d_l_shift ** 2)

    def redshift_flux_lambda(
            self,
            z_new: float,
    ):
        d_l = self.luminosity_distance
        d_l_shift = self.cosmology.luminosity_distance(z_new)
        tbl = self.model_table

        return tbl["flux_lambda"] * ((1 + self.z) * d_l ** 2) / ((1 + z_new) * d_l_shift ** 2)

    def move_to_redshift(self, z_new: float = 1.):
        if z_new != self.z:
            # self.model_table = _d_nu_d_lambda(self.model_table)
            new_tbl = self.model_table["wavelength", "frequency", "flux_nu"].copy()
            # new_tbl["flux_nu_d_nu"] = new_tbl["flux_nu"] * new_tbl["d_nu"]
            new_tbl["wavelength"] = ph.redshift_wavelength(wavelength=new_tbl["wavelength"], z=self.z, z_new=z_new)
            new_tbl["frequency"] = ph.redshift_frequency(nu=new_tbl["frequency"], z=self.z, z_new=z_new)
            # new_tbl = _d_nu_d_lambda(new_tbl)
            # new_tbl["flux_nu"] = new_tbl["flux_nu_d_nu"] / new_tbl["d_nu"]
            new_tbl["flux_nu"] = ph.redshift_flux_nu(
                flux=new_tbl["flux_nu"],
                z=self.z,
                z_new=z_new,
                cosmo=self.cosmology
            )
            new_model = self.__class__(z=z_new, model_table=new_tbl, cosmology=self.cosmology)
            new_model.model_table = new_tbl
            self.shifted_models[z_new] = new_model
        else:
            new_model = self
        return new_model

    # def flux_through_band(
    #         self
    # ):

    def magnitude_AB(
            self,
            band: filters.Filter,
            use_quantum_factor: bool = True
    ):
        tbl = self.model_table
        tbl.sort("frequency")
        transmission = self.add_band(band=band)
        return ph.magnitude_AB(
            flux=tbl["flux_nu"],
            band_transmission=transmission,
            frequency=tbl["frequency"],
            use_quantum_factor=use_quantum_factor
        )

    #
    # def k_correction(
    #         self,
    #         band_obs: filters.Filter,
    #         band_rest: filters.Filter = None,
    # ):
    #     if band_rest is None:
    #         band_rest = band_obs
    #
    # def luminosity_bolometric(self):
    #

    def add_band(
            self,
            band: filters.Filter,
    ):
        """
        Regrids a bandpass to the same wavelength coordinates as the data table.
        :param band:
        :return:
        """
        tbl = self.model_table
        band_name = f"{band.name}_{band.instrument.name}"
        transmission_tbl, _ = band.select_transmission_table()
        transmission_tbl = transmission_tbl.copy()
        if band_name not in tbl.colnames:
            # Find the difference between wavelength entries in the table
            avg_delta = np.median(np.diff(transmission_tbl["Wavelength"]))
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
        return tbl[band_name]

    @classmethod
    def columns(cls):
        return {}


def _d_nu_d_lambda(tbl):
    d_lambda = np.diff(tbl["wavelength"])
    d_lambda = np.append(d_lambda[d_lambda != 0].min(), d_lambda)
    tbl["d_lambda"] = d_lambda
    tbl = tbl[tbl["d_lambda"] != 0]
    d_nu = np.diff(tbl["frequency"])
    d_nu = np.append(d_nu[d_nu != 0].min(), d_nu)
    tbl["d_nu"] = d_nu
    tbl = tbl[tbl["d_nu"] != 0]
    return tbl
