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
        self.distance_modulus: units.Quantity = None
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
        if self.z is not None:
            self.luminosity_distance = self.cosmology.luminosity_distance(z=self.z)
            self.distance_modulus = ph.distance_modulus(self.luminosity_distance).value

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

    def calculate_rest_luminosity(self, force: bool = False):
        if self.rest_luminosity is None or force:
            self.model_table = _d_nu_d_lambda(self.model_table)
            tbl = self.model_table[
                "wavelength", "frequency", "luminosity_lambda", "luminosity_nu", "d_nu", "d_lambda"].copy()
            tbl["luminosity_lambda_d_lambda"] = (tbl["luminosity_lambda"] * tbl["d_lambda"]).to(units.W)
            tbl["luminosity_nu_d_nu"] = (tbl["luminosity_nu"] * tbl["d_nu"]).to(units.W)
            tbl["wavelength"] = self.redshift_wavelength(z_shift=0)
            tbl["frequency"] = self.redshift_frequency(z_shift=0)
            tbl = _d_nu_d_lambda(tbl)
            tbl["luminosity_lambda"] = tbl["luminosity_lambda_d_lambda"] / tbl["d_lambda"]
            tbl["luminosity_nu"] = (tbl["luminosity_nu_d_nu"] / tbl["d_nu"]).to("solLum Hz-1")
            self.rest_luminosity = tbl
        return self.rest_luminosity

    def redshift_wavelength(self, z_shift: float):
        return ph.redshift_wavelength(wavelength=self.model_table["wavelength"], z=self.z, z_new=z_shift)

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

    def flux_in_band(
            self,
            band: filters.Filter,
    ):
        tbl = self.model_table
        transmission = self.add_band(band=band)
        return ph.flux_from_band(
            flux=tbl["flux_nu"],
            transmission=transmission,
            frequency=tbl["frequency"],
            use_quantum_factor=True
        )

    def luminosity_in_band(
            self,
            band: filters.Filter,
    ):
        self.calculate_rest_luminosity()
        tbl = self.rest_luminosity
        transmission = self.add_band(band=band)
        return ph.luminosity_in_band(
            luminosity_nu=tbl["luminosity_nu"],
            transmission=transmission,
            frequency=tbl["frequency"],
            use_quantum_factor=False
        )

    def magnitude_AB(
            self,
            band: filters.Filter,
    ):
        tbl = self.model_table
        transmission = self.add_band(band=band)
        return ph.magnitude_AB(
            flux=tbl["flux_nu"],
            transmission=transmission,
            frequency=tbl["frequency"],
            use_quantum_factor=True
        )

    def magnitude_absolute(
            self,
            band: filters.Filter,
    ):
        self.luminosity_per_frequency()
        self.calculate_rest_luminosity()
        transmission = self.add_band(band=band)
        tbl = self.rest_luminosity
        return ph.magnitude_absolute_from_luminosity(
            luminosity_nu=tbl["luminosity_nu"],
            transmission=transmission,
            frequency=tbl["frequency"],
        )

    def k_correction(
            self,
            band: filters.Filter,
    ):
        """
        Implementing a less-general form of Equation (9) of Hogg et al 2002 (https://arxiv.org/abs/astro-ph/0210394v1)
        :param band:
        :return:
        """

        self.luminosity_per_frequency()
        transmission_obs = self.add_band(band)
        # transmission_emm = self.add_band_rest(band)
        tbl_obs = self.model_table
        tbl_emm = self.rest_luminosity
        # lum_obs_mod = self._luminosity(frequency=(1 + self.z) * tbl_obs["frequency"], tbl=tbl_emm)
        lum_obs = np.trapz(
            x=tbl_obs["luminosity_nu"] * transmission_obs / tbl_obs["frequency"],
            y=tbl_obs["frequency"]
        )
        lum_emm = np.trapz(
            x=tbl_emm["luminosity_nu"] * transmission_obs / tbl_obs["frequency"],
            y=tbl_emm["frequency"]
        )

        return -2.5 * np.log10((1 + self.z) * lum_obs / lum_emm)

    def _luminosity(
            self,
            frequency,
            tbl=None,
    ):
        if tbl is None:
            tbl = self.model_table
        luminosity_interp = np.interp(
            x=frequency,
            xp=tbl["frequency"],
            fp=tbl["luminosity_nu"]
        )
        return luminosity_interp

    def luminosity_bolometric(self):
        self.luminosity_per_frequency()
        self.model_table.sort("frequency")
        return np.trapz(
            y=self.model_table["luminosity_nu"],
            x=self.model_table["frequency"]
        )

    def add_band_rest(
            self,
            band: filters.Filter
    ):
        """

        :param band:
        :return:
        """
        bn = band.machine_name()
        self.calculate_rest_luminosity()
        self.rest_luminosity[bn] = _add_band(self.rest_luminosity, band)
        self.model_table[f"luminosity_nu_{bn}"] = self.rest_luminosity["luminosity_nu"] * self.rest_luminosity[bn]
        return self.rest_luminosity[bn]

    def add_band(
            self,
            band: filters.Filter,
    ):
        """
        Regrids a bandpass to the same wavelength coordinates as the data table, and adds the transmission as a column
        in the model table.
        :param band:
        :return:
        """
        bn = band.machine_name()
        self.model_table[bn] = _add_band(self.model_table, band)
        self.model_table[f"flux_nu_{bn}"] = self.model_table["flux_nu"] * self.model_table[bn]
        return self.model_table[bn]

    def write(self, path: str, fmt="ascii.ecsv", **kwargs):
        self.model_table.write(path, format=fmt, overwrite=True, **kwargs)

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


def _add_band(tbl: table.QTable, band: filters.Filter):
    band_name = band.machine_name()
    if band_name not in tbl.colnames:
        tbl.sort("wavelength")
        tbl[band_name] = band.interp_to_wavelength(tbl[f"wavelength"])
    return tbl[band_name]
