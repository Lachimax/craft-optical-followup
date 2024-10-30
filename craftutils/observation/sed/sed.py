import os.path
from typing import List, Union, Dict

import numpy as np

import astropy.table as table
import astropy.units as units
import astropy.constants as constants
import astropy.cosmology as cosmology

import craftutils.observation.filters as filters
import craftutils.photometry as ph
import craftutils.utils as u
import craftutils.params as p


@u.export
class SEDModel:
    default_cosmology = cosmology.Planck18

    def __init__(
            self,
            z: float = None,
            input_path: str = None,
            model_table: table.QTable = None,
            **kwargs
    ):
        self.input_path: str = None
        self.output_dir: str = None
        self.output_file: str = None
        self.model_table: table.QTable = None
        self.model_table_path: str = None
        self.name: str = None
        self.z: float = z
        self.obs_table: table.QTable = None
        self.luminosity_distance: units.Quantity = None
        self.distance_modulus: units.Quantity = None
        self.rest_luminosity = None
        self.cosmology: cosmology.LambdaCDM = self.default_cosmology
        self.z_mag_tbl: table.QTable = None
        self.z_flux_tbl: table.QTable = None
        self.sfh = None

        self.shifted_models: Dict[str, 'SEDModel'] = {}
        if input_path is not None:
            self.input_path = str(input_path)
            self.load_data()
        elif model_table is not None:
            self.model_table = model_table
        for key, value in kwargs.items():
            setattr(self, key, value)

        if isinstance(self.output_dir, str):
            u.mkdir_check(self.output_dir)
            if self.output_file is None:
                self.set_output_file()

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
        self.model_table = table.QTable.read(self.input_path)
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
            tbl["flux_lambda"] = (tbl["flux_nu"] * tbl["frequency"] ** 2 / constants.c).to("erg cm-2 s-1 angstrom-1")

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

    def move_to_redshift(
            self,
            z_new: float = 1.,
            force: bool = False,
            write: bool = True,
            output: str = None
    ) -> 'SEDModel':
        """
        Displaces this model in redshift, attenuating its flux appropriately.

        :param z_new: cosmological redshift to move the model to.
        :param force: recalculates the model; otherwise, if the same z has been calculated before, re-uses that.
        :param write: attempts to write the new table to disk.
        :return: New SEDModel object with the redshifted values.
        """
        if force or z_new not in self.shifted_models:
            if z_new != self.z:
                new_tbl = self.model_table["wavelength", "frequency", "flux_nu"].copy()
                new_tbl["wavelength"] = ph.redshift_wavelength(wavelength=new_tbl["wavelength"], z=self.z, z_new=z_new)
                new_tbl["frequency"] = ph.redshift_frequency(nu=new_tbl["frequency"], z=self.z, z_new=z_new)
                new_tbl["flux_nu"] = ph.redshift_flux_nu(
                    flux=new_tbl["flux_nu"],
                    z=self.z,
                    z_new=z_new,
                    cosmo=self.cosmology
                )

                if self.name:
                    name = self.name + f"_z_{z_new}"
                else:
                    name = None
                if write:
                    if isinstance(output, str):
                        new_dir = output
                    elif self.output_dir:
                        new_dir = os.path.join(self.output_dir, name)
                    else:
                        raise ValueError("If write is True, self.output_dir must be set or output must be provided")
                else:
                    new_dir = None
                new_model = self.__class__(
                    z=z_new,
                    model_table=new_tbl,
                    cosmology=self.cosmology,
                    name=name,
                    output_dir=new_dir
                )
                if write:
                    new_model.write_model_table()
                    new_model.update_output_file()
                self.shifted_models[z_new] = new_model
            else:
                new_model = self
        else:
            new_model = self.shifted_models[z_new]
        return new_model

    def set_output_file(self, path: str = None):
        if path is None and self.output_dir is not None:
            self.output_file = os.path.join(self.output_dir, self._output_file_name())
        else:
            self.output_file = path

    def mag_from_model_flux(
            self,
            bands: List[filters.Filter],
            z_shift: float = 1.0,
            force: bool = False,
            write: bool = True,
    ):
        """
        Uses a set of bandpasses to derive observed magnitudes for this model when placed at a counterfactual redshift.

        :param z_shift: Redshift at which to place the model.
        :param bands: List or other iterable of `craftutils.observation.filters.Filter` objects through which to observe
            the redshifted galaxy.
                :param force: recalculates the model; otherwise, if the same z has been calculated before, re-uses that.
        :param write: attempts to write the new table to disk.
        :return: dict of AB magnitudes in the given bands, with keys being the band names and values being the magnitudes.
        """
        mags = {}
        shifted_model = self.move_to_redshift(z_new=z_shift, force=force, write=write)

        fluxes = {}

        for band in bands:
            band_name = band.machine_name()
            m_ab_shift = shifted_model.magnitude_AB(band=band)
            mags[band_name] = m_ab_shift
            fluxes[band_name] = shifted_model.flux_in_band(band=band)

        return mags, fluxes

    def z_mag_table(
            self,
            bands: List[filters.Filter],
            z_min: float = 0.01,
            z_max: float = 3.0,
            n_z: int = 20,
            recalculate_all: bool = False,
            write_all: bool = False,
            include_actual: bool = True
    ):
        """
        Uses `move_to_redshift()` to construct a table of magnitudes vs displaced redshift for this model.

        :param bands: List or other iterable of `craftutils.observation.filters.Filter` objects through which to observe
            the redshifted galaxy.
        :param z_min: Minimum redshift to take measurements at.
        :param z_max: Maximum redshift to take measurements at.
        :param n_z: number of redshifts to take measurements at.
        :param bands: List or other iterable of `craftutils.observation.filters.Filter` objects through which to observe
            the redshifted galaxy.
        :param recalculate_all: Forces models to be recalculated; otherwise, those that were stored already are used.
        :param write_all: Write all of the model tables to disk?
        :param include_actual: Whether to insert actual redshift in the z list.
        :return: `astropy.table.QTable` of the derived magnitudes, with columns corresponding to bands
            and redshift.
        """
        mags = {}
        fluxes = {}
        for band in bands:
            band_name = band.machine_name()
            mags[band_name] = []
            fluxes[band_name] = []
        # Construct array of redshifts.
        zs = list(np.linspace(z_min, z_max, n_z))
        if include_actual:
            zs.append(self.z)
            zs.sort()
        for z in zs:
            # Get magnitudes at this redshift.
            mags_this, flux_this = self.mag_from_model_flux(
                bands=bands,
                z_shift=z,
                force=recalculate_all,
                write=write_all
            )
            for band in bands:
                band_name = band.machine_name()
                mags[band_name].append(
                    mags_this[band_name]
                )
                fluxes[band_name].append(
                    flux_this[band_name]
                )
        mags["z"] = zs
        fluxes["z"] = zs
        self.z_mag_tbl = table.QTable(mags)
        self.z_flux_tbl = table.QTable(fluxes)
        return self.z_mag_tbl

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
        """
        Generates an AB magnitude from the SED using the given bandpass.

        :param band: `craftutils.observation.filters.Filter` object through which to observe the galaxy.
        :return:
        """
        tbl = self.model_table
        transmission = self.add_band(band=band)
        return ph.magnitude_AB(
            flux=tbl["flux_nu"],
            transmission=transmission,
            frequency=tbl["frequency"],
            use_quantum_factor=True,
            mag_unit=False
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

    def write_model_table(
            self,
            path: str = None,
            **kwargs
    ):
        if path is None:
            path = self._model_table_path()
        self.model_table_path = path
        self.model_table.write(path, overwrite=True, **kwargs)

    def _model_table_path(self):
        if self.output_dir:
            return os.path.join(self.output_dir, f"{self.name}_model_table.ecsv")

    def load_model_table(
            self,
            path: str = None,
    ):
        if path is None:
            if self.model_table_path:
                path = self.model_table_path
            elif self.output_dir:
                path = self._model_table_path()
            else:
                raise ValueError("No valid path for model table found.")
        self.model_table = table.QTable.read(path)
        return self.model_table

    def expunge_shifted_models(self):
        """
        Clears `self.shifted_models` and removes every z-displaced version of this model from memory.
        Useful for keeping memory usage low; the model tables can occupy memory quickly at scale.
        """
        for name, model in self.shifted_models.items():
            self.shifted_models[name] = None
            del model
        self.shifted_models.clear()

    def load_z_mag_table(
            self,
            path: str = None,
    ):
        """
        Loads a z-magnitude table from disk. If no path is given, uses either `self.z_mag_tbl_path` or, if this is not
        set, tries to guess.

        :param path: the path to the table.
        :return: the z-magnitude table.
        """
        if path is None:
            if self.z_mag_tbl_path:
                path = self.z_mag_tbl_path
            elif self.output_dir:
                path = self._z_mag_tbl_path()
            else:
                raise ValueError("No valid path for model table found.")
        self.z_mag_tbl = table.QTable.read(path)
        return self.z_mag_tbl

    def _z_mag_tbl_path(self):
        return os.path.join(self.output_dir, f"{self.name}_z_mag_table.ecsv")

    def write_z_mag_table(
            self,
            path: str = None,
            **kwargs
    ):
        if path is None:
            path = self._z_mag_tbl_path()
        self.z_mag_tbl_path = path
        self.z_mag_tbl.write(path, overwrite=True, **kwargs)

    def write_z_flux_table(
            self,
            path: str = None,
            **kwargs
    ):
        if path is None:
            path = os.path.join(self.output_dir, f"{self.name}_z_mag_table.ecsv")
        self.z_flux_tbl_path = path
        self.z_flux_tbl.write(path, overwrite=True, **kwargs)

    def _output_file_name(self):
        if self.name:
            name = f"{self.name}_outputs.yaml"
        else:
            name = f"{type(self)}"
        return name

    def update_output_file(self):
        if self.output_dir is not None:
            p.update_output_file(self)

    def _output_dict(self):
        output_dict = self.__dict__.copy()
        output_dict.pop("z_flux_tbl")
        output_dict.pop("z_mag_tbl")
        output_dict.pop("model_table")
        for key, value in output_dict.items():
            output_dict[key] = value
        return output_dict

    @classmethod
    def from_dict(cls, dictionary: dict):
        sed_class = SEDModel.select_child_class(dictionary["type"])
        model = sed_class(
            **dictionary
        )
        return model

    @classmethod
    def select_child_class(cls, type_str: str):
        from .__init__ import CIGALEModel, GordonProspectorModel
        type_dict = {
            "CIGALEModel": CIGALEModel,
            "GordonProspectorModel": GordonProspectorModel,
            "SEDModel": SEDModel
        }
        if type_str in type_dict:
            return type_dict[type_str]
        else:
            raise ValueError(f"SEDModel type {type_str} not recognised")

    @classmethod
    def columns(cls):
        return {}

@u.export
class NormalisedTemplate(SEDModel):
    pass

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
