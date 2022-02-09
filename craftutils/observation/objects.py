from typing import Union, Tuple
import os

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as units
import astropy.table as table
import astropy.cosmology as cosmo
from astropy.modeling import models, fitting
from astropy.visualization import quantity_support

ne2001_installed = True
try:
    from ne2001.density import NEobject
except ImportError:
    print("ne2001 is not installed; DM_ISM estimates will not be available.")
    ne2001_installed = False

frb_installed = True
try:
    import frb.halos.models as halos
    import frb.halos.hmf as hmf
    import frb.dm.igm as igm
except ImportError:
    print("FRB is not installed; DM_ISM estimates will not be available.")
    frb_installed = False

import craftutils.params as p
import craftutils.astrometry as a
import craftutils.utils as u
import craftutils.observation.instrument as inst
import craftutils.retrieve as r

try:
    cosmology = cosmo.Planck18
except AttributeError:
    cosmology = cosmo.Planck15

quantity_support()

position_dictionary = {
    "ra": {
        "decimal": 0.0,
        "hms": None},
    "dec": {
        "decimal": 0.0,
        "dms": None
    },
}

uncertainty_dict = {
    "sys": 0.0,
    "stat": 0.0
}


class PositionUncertainty:
    def __init__(
            self,
            uncertainty: Union[float, units.Quantity, dict, tuple] = None,
            position: SkyCoord = None,
            ra_err_sys: Union[float, units.Quantity] = None,
            ra_err_stat: Union[float, units.Quantity] = None,
            dec_err_sys: Union[float, units.Quantity] = None,
            dec_err_stat: Union[float, units.Quantity] = None,
            a_stat: Union[float, units.Quantity] = None,
            a_sys: Union[float, units.Quantity] = None,
            b_stat: Union[float, units.Quantity] = None,
            b_sys: Union[float, units.Quantity] = None,
            theta: Union[float, units.Quantity] = None,
            sigma: float = None
    ):
        """
        If a single value is provided for uncertainty, the uncertainty ellipse will be assumed to be circular.
        Values in dictionary, if provided, override values given as arguments.
        Position values provided without units are assumed to be in degrees.
        On the other hand, uncertainty values provided without units are assumed to be in arcseconds;
        except for uncertainties in RA, which are assumed in RA seconds.
        :param uncertainty:
        :param position:
        :param ra_err_sys:
        :param ra_err_stat:
        :param dec_err_sys:
        :param dec_err_stat:
        :param a_stat:
        :param a_sys:
        :param b_stat:
        :param b_sys:
        :param theta:
        :param sigma: The confidence interval (expressed in multiples of sigma) of the uncertainty ellipse.
        """
        self.sigma = sigma
        # Assign values from dictionary, if provided.
        if type(uncertainty) is dict:
            if "ra" in uncertainty and uncertainty["ra"] is not None:
                if "sys" in uncertainty["ra"] and uncertainty["ra"]["sys"] is not None:
                    ra_err_sys = uncertainty["ra"]["sys"]
                if "stat" in uncertainty["ra"] and uncertainty["ra"]["stat"] is not None:
                    ra_err_stat = uncertainty["ra"]["stat"]
            if "dec" in uncertainty and uncertainty["dec"] is not None:
                if "sys" in uncertainty["dec"] and uncertainty["dec"]["sys"] is not None:
                    dec_err_sys = uncertainty["dec"]["sys"]
                if "stat" in uncertainty["dec"] and uncertainty["dec"]["stat"] is not None:
                    dec_err_stat = uncertainty["dec"]["stat"]
            if "a" in uncertainty and uncertainty["a"] is not None:
                if "sys" in uncertainty["a"] and uncertainty["a"]["sys"] is not None:
                    a_sys = uncertainty["a"]["sys"]
                if "stat" in uncertainty["a"] and uncertainty["a"]["stat"] is not None:
                    a_stat = uncertainty["a"]["stat"]
            if "b" in uncertainty and uncertainty["b"] is not None:
                if "sys" in uncertainty["b"] and uncertainty["b"]["sys"] is not None:
                    b_sys = uncertainty["b"]["sys"]
                if "stat" in uncertainty["b"] and uncertainty["a"]["stat"] is not None:
                    b_stat = uncertainty["b"]["stat"]
            if "theta" in uncertainty and uncertainty["theta"] is not None:
                theta = uncertainty["theta"]

        # If uncertainty is a single value, assume a circular uncertainty region without distinction between systematic
        # and statistical.
        elif uncertainty is not None:
            a_stat = uncertainty
            a_sys = 0.0
            b_stat = uncertainty
            b_sys = 0.0
            theta = 0.0

        # Check whether we're specifying uncertainty using equatorial coordinates or ellipse parameters.
        u.debug_print(2, "PositionUncertainty.__init__(): a_stat, a_sys, b_stat, b_sys, theta, position ==", a_stat,
                      a_sys, b_stat, b_sys, theta, position)
        u.debug_print(2, "PositionUncertainty.__init__(): ra_err_sys, ra_err_stat, dec_err_sys, dec_err_stat ==",
                      ra_err_sys, ra_err_stat, dec_err_sys, dec_err_stat)
        if a_stat is not None and a_sys is not None and b_stat is not None and b_sys is not None and theta is not None and position is not None:
            ellipse = True
        elif ra_err_sys is not None and ra_err_stat is not None and dec_err_sys is not None and dec_err_stat is not None:
            ellipse = False
        else:
            raise ValueError(
                "Either all ellipse values (a, b, theta) or all equatorial values (ra, dec, position) must be provided.")

        # Convert equatorial uncertainty to ellipse with theta=0
        if not ellipse:
            ra_err_sys = u.check_quantity(number=ra_err_sys, unit=units.hourangle / 3600)
            ra_err_stat = u.check_quantity(number=ra_err_stat, unit=units.hourangle / 3600)
            dec_err_sys = u.check_quantity(number=dec_err_sys, unit=units.arcsec)
            dec_err_stat = u.check_quantity(number=dec_err_stat, unit=units.arcsec)

            ra = position.ra
            dec = position.dec
            a_sys = SkyCoord(0.0 * units.degree, dec).separation(SkyCoord(ra_err_sys, dec))
            a_stat = SkyCoord(0.0 * units.degree, dec).separation(SkyCoord(ra_err_stat, dec))
            b_sys = SkyCoord(ra, dec).separation(SkyCoord(ra, dec + dec_err_sys))
            b_stat = SkyCoord(ra, dec).separation(SkyCoord(ra, dec + dec_err_stat))
            a_sys, b_sys = max(a_sys, b_sys), min(a_sys, b_sys)
            a_stat, b_stat = max(a_stat, b_stat), min(a_stat, b_stat)
            theta = 0.0 * units.degree
        # Or use ellipse parameters as given.
        else:
            a_sys = u.check_quantity(number=a_sys, unit=units.arcsec)
            a_stat = u.check_quantity(number=a_stat, unit=units.arcsec)
            b_sys = u.check_quantity(number=b_sys, unit=units.arcsec)
            b_stat = u.check_quantity(number=b_stat, unit=units.arcsec)
            theta = u.check_quantity(number=theta, unit=units.arcsec)

        self.a_sys = a_sys
        self.a_stat = a_stat
        self.b_sys = b_sys
        self.b_stat = b_stat
        self.theta = theta

    def uncertainty_quadrature(self):
        return np.sqrt(self.a_sys ** 2 + self.a_stat ** 2), np.sqrt(self.b_sys ** 2 + self.b_stat ** 2)

    @classmethod
    def default_params(cls):
        return {
            "ra": uncertainty_dict.copy(),
            "dec": uncertainty_dict.copy(),
            "a": uncertainty_dict.copy(),
            "b": uncertainty_dict.copy(),
            "theta": 0.0,
            "sigma": None,
            "healpix_path": None
        }


class Object:
    def __init__(
            self,
            name: str = None,
            position: Union[SkyCoord, str] = None,
            position_err: Union[float, units.Quantity, dict, PositionUncertainty, tuple] = 0.0 * units.arcsec,
            field=None,
            plotting: dict = None
    ):
        self.name = name
        self.position = a.attempt_skycoord(position)
        if type(position_err) is not PositionUncertainty:
            self.position_err = PositionUncertainty(uncertainty=position_err, position=self.position)
        self.position_galactic = None
        if isinstance(self.position, SkyCoord):
            self.position_galactic = self.position.transform_to("galactic")

        self.cat_row = None
        self.photometry = {}
        self.photometry_tbl = None
        self.data_path = None
        self.output_file = None
        self.field = field
        self.irsa_extinction_path = None
        self.irsa_extinction = None
        self.ebv_sandf = None
        self.extinction_power_law = None
        self.paths = {}
        self.load_output_file()
        if isinstance(plotting, dict):
            self.plotting_params = plotting
            if "frame" in self.plotting_params and self.plotting_params["frame"] is not None:
                self.plotting_params["frame"] = u.check_quantity(self.plotting_params["frame"], units.arcsec)
        else:
            self.plotting_params = {}

    def position_from_cat_row(self):
        self.position = SkyCoord(self.cat_row["RA"], self.cat_row["DEC"])
        return self.position

    def get_photometry(self):
        for cat in self.field.cats:
            pass

    def find_in_cat(self, cat_name: str):
        cat = self.field.load_catalogue(cat_name=cat_name)
        pass

    def _output_dict(self):
        return {
            "photometry": self.photometry,
            "irsa_extinction_path": self.irsa_extinction_path,
            "extinction_law": self.extinction_power_law,
        }

    def load_output_file(self):
        self.check_data_path()
        if self.data_path is not None:
            outputs = p.load_output_file(self)
            if outputs is not None:
                if "photometry" in outputs and outputs["photometry"] is not None:
                    self.photometry = outputs["photometry"]
                if "irsa_extinction_path" in outputs and outputs["irsa_extinction_path"] is not None:
                    self.irsa_extinction_path = outputs["irsa_extinction_path"]
            return outputs

    def check_data_path(self):
        if self.field is not None:
            u.debug_print(2, "", self.name)
            self.data_path = os.path.join(self.field.data_path, "objects", self.name)
            u.mkdir_check(self.data_path)
            self.output_file = os.path.join(self.data_path, f"{self.name}_outputs.yaml")
            return True
        else:
            return False

    def update_output_file(self):
        if self.check_data_path():
            p.update_output_file(self)

    def write_plot_photometry(self, output: str = None, **kwargs):
        """
        Plots available photometry (mag v lambda_eff) and writes to disk.
        :param output: Path to write plot.
        :return: matplotlib ax object containing plot info
        """
        if output is None:
            output = os.path.join(self.data_path, f"{self.name}_photometry.pdf")

        ax = self.plot_photometry(**kwargs)
        ax.legend()
        plt.savefig(output)
        return ax

    def plot_photometry(self, ax=None, **kwargs):
        """
        Plots available photometry (mag v lambda_eff).
        :param ax: matplotlib ax object to plot with. A new object is generated if none is provided.
        :param kwargs:
        :return: matplotlib ax object containing plot info
        """
        if ax is None:
            fig, ax = plt.subplots()
        if "ls" not in kwargs:
            kwargs["ls"] = ""
        if "marker" not in kwargs:
            kwargs["marker"] = "x"
        if "ecolor" not in kwargs:
            kwargs["ecolor"] = "black"

        with quantity_support():
            for instrument_name in self.photometry:
                instrument = inst.Instrument.from_params(instrument_name)
                lambda_eff = units.Quantity(list(map(
                    lambda f: instrument.filters[f].lambda_eff,
                    self.photometry[instrument_name]
                ))).to(units.Angstrom)
                mag = units.Quantity(list(map(
                    lambda f: self.photometry[instrument_name][f]["mag"],
                    self.photometry[instrument_name]
                )))
                mag_err = units.Quantity(list(map(
                    lambda f: self.photometry[instrument_name][f]["mag_err"],
                    self.photometry[instrument_name]
                )))

                mag_corrected = units.Quantity(list(map(
                    lambda f: self.photometry[instrument_name][f]["mag_ext_corrected"],
                    self.photometry[instrument_name]
                )))

                ax.errorbar(
                    lambda_eff,
                    mag,
                    yerr=mag_err,
                    label=instrument_name,
                    **kwargs,
                )
                ax.scatter(
                    lambda_eff,
                    mag_corrected,
                    color="violet",
                    label="Corrected for Galactic extinction"
                )
                ax.set_ylabel("Apparent magnitude")
                ax.set_xlabel("$\lambda_\mathrm{eff}$ (\AA)")
            ax.invert_yaxis()
        return ax

    def build_photometry_table_path(self):
        return os.path.join(self.data_path, f"{self.name}_photometry.ecsv")

    # TODO: Refactor photometry to use table instead of dict (not sure why I even did it that way to start with)

    def photometry_to_table(self, output: str = None, fmt: str = "ascii.ecsv"):
        """
        Converts the photometry information, which is stored internally as a dictionary, into an astropy QTable.
        :param output: Where to write table.
        :return:
        """

        if output is None:
            output = self.build_photometry_table_path()

        # if self.photometry_tbl is None:

        tbls = []
        for instrument_name in self.photometry:
            instrument = inst.Instrument.from_params(instrument_name)
            for filter_name in self.photometry[instrument_name]:
                fil = instrument.filters[filter_name]
                phot_dict = self.photometry[instrument_name][filter_name].copy()
                phot_dict["band"] = filter_name
                phot_dict["instrument"] = instrument_name
                phot_dict["lambda_eff"] = u.check_quantity(
                    number=fil.lambda_eff,
                    unit=units.Angstrom)
                tbl = table.QTable([phot_dict])
                tbls.append(tbl)
        self.photometry_tbl = table.vstack(tbls)

        self.photometry_tbl.write(output, format=fmt, overwrite=True)
        return self.photometry_tbl

    def estimate_galactic_extinction(self, ax=None, r_v: float = 3.1, **kwargs):
        import extinction
        if ax is None:
            fig, ax = plt.subplots()
        if "marker" not in kwargs:
            kwargs["marker"] = "x"

        self.retrieve_extinction_table()
        lambda_eff_tbl = self.irsa_extinction["LamEff"].to(
            units.Angstrom)
        power_law = models.PowerLaw1D()
        fitter = fitting.LevMarLSQFitter()
        fitted = fitter(power_law, lambda_eff_tbl, self.irsa_extinction["A_SandF"].value)

        tbl = self.photometry_to_table()

        x = np.linspace(0, 80000, 1000) * units.Angstrom

        a_v = r_v * self.ebv_sandf

        tbl["ext_gal_sandf"] = extinction.fitzpatrick99(tbl["lambda_eff"], a_v, r_v) * units.mag
        tbl["ext_gal_pl"] = fitted(tbl["lambda_eff"]) * units.mag
        tbl["ext_gal_interp"] = np.interp(
            tbl["lambda_eff"],
            lambda_eff_tbl,
            self.irsa_extinction["A_SandF"].value
        ) * units.mag

        ax.plot(
            x, extinction.fitzpatrick99(x, a_v, r_v),
            label="S\&F + F99 extinction law",
            c="red"
        )
        ax.plot(
            x, fitted(x),
            label=f"power law fit to IRSA",
            # , \\alpha={fitted.alpha.value}; $x_0$={fitted.x_0.value}; A={fitted.amplitude.value}",
            c="blue"
        )
        ax.scatter(
            lambda_eff_tbl, self.irsa_extinction["A_SandF"].value,
            label="from IRSA",
            c="green",
            **kwargs)
        ax.scatter(
            tbl["lambda_eff"], tbl["ext_gal_pl"].value,
            label="power law interpolation of IRSA",
            c="blue",
            **kwargs
        )
        ax.scatter(
            tbl["lambda_eff"], tbl["ext_gal_interp"].value,
            label="numpy interpolation from IRSA",
            c="violet",
            **kwargs
        )
        ax.scatter(
            tbl["lambda_eff"], tbl["ext_gal_sandf"].value,
            label="S\&F + F99 extinction law",
            c="red",
            **kwargs
        )
        ax.set_ylim(0, 0.6)
        ax.legend()
        plt.savefig(os.path.join(self.data_path, f"{self.name}_irsa_extinction.pdf"))
        plt.close()
        self.extinction_power_law = {
            "amplitude": fitted.amplitude.value * fitted.amplitude.unit,
            "x_0": fitted.x_0.value,
            "alpha": fitted.alpha.value
        }

        for row in tbl:
            instrument = row["instrument"]
            band = row["band"]

            # if row["lambda_eff"] > max(lambda_eff_tbl) or row["lambda_eff"] < min(lambda_eff_tbl):
            #     key = "ext_gal_pl"
            #     self.photometry[instrument][band]["ext_gal_type"] = "power_law_fit"
            # else:
            #     key = "ext_gal_interp"
            #     self.photometry[instrument][band]["ext_gal_type"] = "interpolated"
            key = "ext_gal_sandf"
            self.photometry[instrument][band]["ext_gal_type"] = "s_and_f"
            self.photometry[instrument][band]["ext_gal"] = row[key]
            u.debug_print(1, key, row['mag'], row[key])
            self.photometry[instrument][band]["mag_ext_corrected"] = row["mag"] - row[key]

        tbl_2 = self.photometry_to_table()
        # tbl_2.update(tbl)
        # tbl_2.write(self.build_photometry_table_path().replace("photometry", "photemetry_extended"))
        return ax

    def retrieve_extinction_table(self, force: bool = False):
        self.load_extinction_table()
        if force or self.irsa_extinction is None:
            raw_path = os.path.join(self.data_path, f"{self.name}_irsa_extinction.ecsv")
            r.save_irsa_extinction(
                ra=self.position.ra.value,
                dec=self.position.dec.value,
                output=raw_path
            )
            ext_tbl = table.QTable.read(raw_path, format="ascii")
            for colname in ext_tbl.colnames:
                if str(ext_tbl[colname].unit) == "mags":
                    ext_tbl[colname]._set_unit(units.mag)
            tbl_path = os.path.join(self.data_path, f"{self.name}_galactic_extinction.ecsv")
            ext_tbl.write(tbl_path, overwrite=True, format="ascii.ecsv")
            self.irsa_extinction = ext_tbl
            self.irsa_extinction_path = tbl_path

        if force or self.ebv_sandf is None:
            # Get E(B-V) at this coordinate.
            tbl = r.retrieve_irsa_details(coord=self.position)
            self.ebv_sandf = tbl["ext SandF ref"]

    def load_extinction_table(self, force: bool = False):
        if force or self.irsa_extinction is None:
            if self.irsa_extinction_path is not None:
                u.debug_print(1, "Loading irsa_extinction from", self.irsa_extinction_path)
                self.irsa_extinction = table.QTable.read(self.irsa_extinction_path, format="ascii.ecsv")

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "position": position_dictionary.copy(),
            "position_err": PositionUncertainty.default_params(),
            "type": None,
            "photometry_args_manual":
                {
                    "a": 0.0 * units.arcsec,
                    "b": 0.0 * units.arcsec,
                    "theta": 0.0 * units.arcsec,
                    "kron_radius": 3.5
                },
            "plotting": {
                "frame": None
            }
        }
        return default_params

    @classmethod
    def from_dict(cls, dictionary: dict, field=None) -> 'Object':
        """
        Construct an Object or appropriate child class (FRB, Galaxy...) from a passed dict.
        :param dictionary: dict with keys:
            'position': position dictionary as given by position_dictionary
            'position_err':
        :return: Object reflecting dictionary.
        """
        ra, dec = p.select_coords(dictionary["position"])
        if "position_err" in dictionary:
            position_err = dictionary["position_err"]
        else:
            position_err = PositionUncertainty.default_params()

        if "type" in dictionary and dictionary["type"] is not None:
            selected = cls.select_child_class(obj_type=dictionary["type"])
        else:
            selected = cls

        if "plotting" in dictionary:
            plotting = dictionary["plotting"]
        else:
            plotting = None

        if selected in (Object, FRB):
            return selected(
                name=dictionary["name"],
                position=f"{ra} {dec}",
                position_err=position_err,
                field=field,
                plotting=plotting
            )
        else:
            return selected.from_dict(dictionary=dictionary, field=field)

    @classmethod
    def select_child_class(cls, obj_type: str):
        obj_type = obj_type.lower()
        if obj_type == "galaxy":
            return Galaxy
        elif obj_type == "frb":
            return FRB
        else:
            raise ValueError(f"Didn't recognise obj_type '{obj_type}'")

    @classmethod
    def from_source_extractor_row(cls, row: table.Row, use_psf_params: bool = False):
        if use_psf_params:
            ra_key = "ALPHAPSF_SKY"
            dec_key = "DELTAPSF_SKY"
            ra_err_key = "ERRX2_WORLD"
            dec_err_key = "ERRY2_WORLD"
        else:
            ra_key = "ALPHA_SKY"
            dec_key = "DELTA_SKY"
            ra_err_key = "ERRX2PSF_WORLD"
            dec_err_key = "ERRY2PSF_WORLD"
        ra_err = np.sqrt(row[ra_err_key])
        dec_err = np.sqrt(row[dec_err_key])
        obj = cls(name=str(row["NUMBER"]),
                  position=SkyCoord(row[ra_key], row[dec_key]),
                  position_err=PositionUncertainty(
                      ra_err_stat=ra_err,
                      dec_err_stat=dec_err),
                  )
        obj.cat_row = row
        return obj


class Galaxy(Object):
    def __init__(
            self,
            name: str = None,
            position: Union[SkyCoord, str] = None,
            position_err: Union[float, units.Quantity, dict, PositionUncertainty, tuple] = 0.0 * units.arcsec,
            z: float = 0.0,
            field=None,
            plotting: dict = None,
            **kwargs
    ):
        super().__init__(
            name=name,
            position=position,
            position_err=position_err,
            field=field,
            plotting=plotting
        )
        self.z = z
        self.D_A = self.angular_size_distance()
        self.D_L = self.luminosity_distance()
        self.mu = self.distance_modulus()

        if "mass" in kwargs:
            self.mass = kwargs["mass"]
        else:
            self.mass = None
        if "mass_stellar" in kwargs:
            self.mass_stellar = kwargs["mass_stellar"]
        else:
            self.mass_stellar = None

    def angular_size_distance(self):
        return cosmology.angular_diameter_distance(z=self.z)

    def luminosity_distance(self):
        return cosmology.luminosity_distance(z=self.z)

    def distance_modulus(self):
        d = self.luminosity_distance()
        mu = (5 * np.log10(d / units.pc) - 5) * units.mag
        return mu

    def absolute_magnitude(
            self,
            apparent_magnitude: units.Quantity,
            internal_extinction: units.Quantity = 0 * units.mag,
            galactic_extinction: units.Quantity = 0 * units.mag
    ):
        mu = self.distance_modulus()
        return apparent_magnitude - mu - internal_extinction - galactic_extinction

    def absolute_photometry(self, internal_extinction: units.Quantity = 0.0 * units.mag):
        for instrument in self.photometry:
            for fil in self.photometry[instrument]:
                abs_mag = self.absolute_magnitude(
                    apparent_magnitude=self.photometry[instrument][fil]["mag"],
                    internal_extinction=internal_extinction
                )
                self.photometry[instrument][fil]["abs_mag"] = abs_mag
        self.update_output_file()

    def projected_distance(self, angle: units.Quantity):
        angle = angle.to(units.rad).value
        dist = angle * self.D_A
        return dist

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        default_params.update({
            "z": 0.0,
            "type": "galaxy"
        })
        return default_params

    @classmethod
    def from_dict(cls, dictionary: dict, field=None):
        ra, dec = p.select_coords(dictionary.pop("position"))
        if "position_err" in dictionary:
            position_err = dictionary.pop("position_err")
        else:
            position_err = PositionUncertainty.default_params()
        return cls(name=dictionary.pop("name"),
                   position=f"{ra} {dec}",
                   position_err=position_err,
                   z=dictionary.pop("z"),
                   field=field,
                   **dictionary)


dm_units = units.parsec * units.cm ** -3


class FRB(Object):
    def __init__(
            self,
            name: str = None,
            position: Union[SkyCoord, str] = None,
            position_err: Union[float, units.Quantity, dict, PositionUncertainty, tuple] = 0.0 * units.arcsec,
            host_galaxy: Galaxy = None,
            dm: Union[float, units.Quantity] = None,
            field=None,
            plotting: dict = None,
    ):
        super().__init__(
            name=name,
            position=position,
            position_err=position_err,
            field=field,
            plotting=plotting
        )
        self.host_galaxy = host_galaxy
        self.dm = dm
        if self.dm is not None:
            self.dm = u.check_quantity(self.dm, unit=dm_units)

    def estimate_dm_mw_ism(self):
        if not frb_installed:
            raise ImportError("FRB is not installed.")
        if not ne2001_installed:
            raise ImportError("ne2001 is not installed.")
        mw = halos.MilkyWay()
        # Declare MW parameters
        params = dict(F=1., e_density=1.)
        model_ne = NEobject(mw.ne, **params)
        l = self.position_galactic.l.value
        b = self.position_galactic.b.value
        return model_ne.DM(l, b, mw.r200.value)

    def estimate_dm_cosmic(self):
        if not frb_installed:
            raise ImportError("FRB is not installed.")
        return igm.average_DM(self.host_galaxy.z, cosmo=cosmo.Planck18)

    def estimate_dm_halos(self):
        if not frb_installed:
            raise ImportError("FRB is not installed.")
        hmf.init_hmf()
        return igm.average_DMhalos(self.host_galaxy.z, cosmo=cosmo.Planck18)

    def estimate_dm_excess(self):
        dm_ism = self.estimate_dm_mw_ism()
        dm_cosmic = self.estimate_dm_cosmic()
        dm_halo = 60 * dm_units
        return self.dm - dm_ism - dm_cosmic - dm_halo

    def estimate_dm_exgal(self):
        dm_ism = self.estimate_dm_mw_ism()
        dm_halo = 60 * dm_units
        return self.dm - dm_ism - dm_halo

    @classmethod
    def from_dict(cls, dictionary: dict, name: str = None, field=None):
        frb = super().from_dict(dictionary=dictionary)
        if "dm" in dictionary:
            frb.dm = dictionary["dm"] * dm_units
        frb.host_galaxy = Galaxy.from_dict(dictionary=dictionary["host_galaxy"], field=field)
        return frb

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        default_params.update({
            "host_galaxy": Galaxy.default_params(),
            "mjd": 58000,
            "dm": 0.0 * dm_units,
            "snr": 0.0
        })
        return default_params
