from typing import Union, Tuple
import os

import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as units
import astropy.table as table
import astropy.cosmology as cosmo

ne2001_installed = True
try:
    from ne2001.density import NEobject
except ImportError:
    print("ne2001 is not installed; DM_ISM estimates will not be available.")
    ne2001_installed = False

frb_installed = True
try:
    import frb.halos as halos
    import frb.dm.igm as igm
except ImportError:
    print("FRB is not installed; DM_ISM estimates will not be available.")
    frb_installed = False

import craftutils.params as p
import craftutils.astrometry as a
import craftutils.utils as u

position_dictionary = {"ra": {"decimal": 0.0,
                              "hms": "00h00m00s"},
                       "dec": {"decimal": 0.0,
                               "dms": "00d00m00s"},
                       }

uncertainty_dict = {"sys": 0.0,
                    "stat": 0.0}


class PositionUncertainty:
    def __init__(self,
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

        # If uncertainty is a single value, assume a circular uncertainty region without distinction between systematic
        # and statistical.
        elif uncertainty is not None:
            a_stat = uncertainty
            a_sys = 0.0
            b_stat = uncertainty
            b_sys = 0.0
            theta = 0.0

        # Check whether we're specifying uncertainty using equatorial coordinates or ellipse parameters.
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

    @classmethod
    def default_params(cls):
        return {"ra": uncertainty_dict.copy(),
                "dec": uncertainty_dict.copy(),
                "a": uncertainty_dict.copy(),
                "b": uncertainty_dict.copy(),
                "theta": 0.0,
                "sigma": None}


class Object:
    def __init__(
            self,
            name: str = None,
            position: Union[SkyCoord, str] = None,
            position_err: Union[float, units.Quantity, dict, PositionUncertainty, tuple] = 0.0 * units.arcsec,
            field=None):
        self.name = name
        self.position = a.attempt_skycoord(position)
        if type(position_err) is not PositionUncertainty:
            self.position_err = PositionUncertainty(uncertainty=position_err, position=self.position)
        self.position_galactic = None
        if isinstance(self.position, SkyCoord):
            self.position_galactic = self.position.transform_to("galactic")

        self.cat_row = None
        self.photometry = {}
        self.data_path = None
        self.output_file = None
        self.field = field
        self.load_output_file()

    def get_photometry(self):
        for cat in self.field.cats:
            pass

    def find_in_cat(self, cat_name: str):
        cat = self.field.load_catalogue(cat_name=cat_name)
        pass

    def _output_dict(self):
        return {"photometry": self.photometry}

    def load_output_file(self):
        self.check_data_path()
        if self.data_path is not None:
            outputs = p.load_output_file(self)
            if outputs is not None:
                if "photometry" in outputs and outputs["photometry"] is not None:
                    self.photometry = outputs["photometry"]
            return outputs

    def check_data_path(self):
        if self.field is not None:
            self.data_path = os.path.join(self.field.data_path, "objects", self.name)
            u.mkdir_check(self.data_path)
            self.output_file = os.path.join(self.data_path, f"{self.name}_outputs.yaml")
            return True
        else:
            return False

    def update_output_file(self):
        if self.check_data_path():
            p.update_output_file(self)

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "position": position_dictionary.copy(),
            "position_err": PositionUncertainty.default_params(),
            "type": None,
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

        if selected in (Object, FRB):
            return selected(name=dictionary["name"],
                            position=f"{ra} {dec}",
                            position_err=position_err,
                            field=field
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
            field=None
    ):
        super().__init__(
            name=name,
            position=position,
            position_err=position_err,
            field=field)
        self.z = z
        self.D_A = self.angular_size_distance()

    def angular_size_distance(self, cosmology: cosmo.core.FlatLambdaCDM = None):
        if cosmology is None:
            try:
                cosmology = cosmo.Planck18
            except AttributeError:
                cosmology = cosmo.Planck15
        return cosmology.angular_diameter_distance(z=self.z)

    def projected_distance(self, angle: units.Quantity):
        angle = angle.to(units.rad).value
        dist = angle * self.D_A
        return dist

    @classmethod
    def default_params(cls):
        default_params = super(Galaxy, cls).default_params()
        default_params.update({
            "z": 0.0
        })
        return default_params

    @classmethod
    def from_dict(cls, dictionary: dict, field=None):
        ra, dec = p.select_coords(dictionary["position"])
        if "position_err" in dictionary:
            position_err = dictionary["position_err"]
        else:
            position_err = PositionUncertainty.default_params()
        return cls(name=dictionary["name"],
                   position=f"{ra} {dec}",
                   position_err=position_err,
                   z=dictionary['z'],
                   field=field)


dm_units = units.parsec * units.cm ** -3


class FRB(Object):
    def __init__(
            self,
            name: str = None,
            position: Union[SkyCoord, str] = None,
            position_err: Union[float, units.Quantity, dict, PositionUncertainty, tuple] = 0.0 * units.arcsec,
            host_galaxy: Galaxy = None,
            dm: Union[float, units.Quantity] = None,
            field=None
    ):
        super().__init__(name=name,
                         position=position,
                         position_err=position_err,
                         field=field)
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
        })
        return default_params
