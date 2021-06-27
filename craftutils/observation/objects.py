from typing import Union, Tuple
import os

from astropy.coordinates import SkyCoord
import astropy.units as un

from craftutils import params as p
from craftutils import astrometry as a
from craftutils import utils as u

position_dictionary = {"ra": {"decimal": 0.0,
                              "hms": "00h00m00s"},
                       "dec": {"decimal": 0.0,
                               "dms": "00d00m00s"},
                       }

uncertainty_dict = {"sys": 0.0,
                    "stat": 0.0}


class PositionUncertainty:
    def __init__(self,
                 uncertainty: Union[float, un.Quantity, dict, tuple] = None,
                 position: SkyCoord = None,
                 ra_err_sys: Union[float, un.Quantity] = None,
                 ra_err_stat: Union[float, un.Quantity] = None,
                 dec_err_sys: Union[float, un.Quantity] = None,
                 dec_err_stat: Union[float, un.Quantity] = None,
                 a_stat: Union[float, un.Quantity] = None,
                 a_sys: Union[float, un.Quantity] = None,
                 b_stat: Union[float, un.Quantity] = None,
                 b_sys: Union[float, un.Quantity] = None,
                 theta: Union[float, un.Quantity] = None,
                 sigma: float = None
                 ):
        """
        If a single value is provided for uncertainty, the uncertainty ellipse will be assumed to be circular.
        Any angular values provided without units are assumed to be in degrees.
        Values in dictionary, if provided, override values given as arguments.
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
            ra_err_sys = u.check_quantity(number=ra_err_sys, unit=un.degree)
            ra_err_stat = u.check_quantity(number=ra_err_stat, unit=un.degree)
            dec_err_sys = u.check_quantity(number=dec_err_sys, unit=un.degree)
            dec_err_stat = u.check_quantity(number=dec_err_stat, unit=un.degree)

            ra = position.ra
            dec = position.dec
            a_sys = SkyCoord(0.0 * un.degree, dec).separation(SkyCoord(ra_err_sys, dec))
            a_stat = SkyCoord(0.0 * un.degree, dec).separation(SkyCoord(ra_err_stat, dec))
            b_sys = SkyCoord(ra, dec).separation(SkyCoord(ra, dec + dec_err_sys))
            b_stat = SkyCoord(ra, dec).separation(SkyCoord(ra, dec + dec_err_stat))
            a_sys, b_sys = max(a_sys, b_sys), min(a_sys, b_sys)
            a_stat, b_stat = max(a_stat, b_stat), min(a_stat, b_stat)
            theta = 0.0 * un.degree
        # Or use ellipse parameters as given.
        else:
            a_sys = u.check_quantity(number=a_sys, unit=un.degree)
            a_stat = u.check_quantity(number=a_stat, unit=un.degree)
            b_sys = u.check_quantity(number=b_sys, unit=un.degree)
            b_stat = u.check_quantity(number=b_stat, unit=un.degree)
            theta = u.check_quantity(number=theta, unit=un.degree)

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
    def __init__(self,
                 name: str = None,
                 position: Union[SkyCoord, str] = None,
                 position_err: Union[float, un.Quantity, dict, PositionUncertainty, tuple] = 0.0 * un.arcsec,
                 field=None):
        self.name = name
        self.position = a.attempt_skycoord(position)
        if type(position_err) is not PositionUncertainty:
            self.position_err = PositionUncertainty(uncertainty=position_err, position=self.position)

        self.cat_row = None
        self.photometry = {}
        self.data_path = None
        self.output_file = None
        self.field = field
        self.load_output_file()

    def _output_dict(self):
        return {"photometry": self.photometry}

    def load_output_file(self):
        if self.data_path is not None:
            outputs = p.load_output_file(self)
            if outputs is not None:
                if "photometry" in outputs and outputs["photometry"] is not None:
                    self.photometry = outputs["frame_type"]
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
        }
        return default_params

    @classmethod
    def from_dict(cls, dictionary: dict):
        ra, dec = p.select_coords(dictionary["position"])
        if "position_err" in dictionary:
            position_err = dictionary["position_err"]
        else:
            position_err = PositionUncertainty.default_params()
        return cls(name=dictionary["name"],
                   position=f"{ra} {dec}",
                   position_err=position_err)


class Galaxy(Object):
    def __init__(self,
                 name: str = None,
                 position: Union[SkyCoord, str] = None,
                 position_err: Union[float, un.Quantity, dict, PositionUncertainty, tuple] = 0.0 * un.arcsec,
                 z: float = 0.0):
        super().__init__(name=name,
                         position=position,
                         position_err=position_err)
        self.z = z

    @classmethod
    def default_params(cls):
        default_params = super(Galaxy, cls).default_params()
        default_params.update({
            "z": 0.0
        })
        return default_params


class FRB(Object):
    def __init__(self,
                 name: str = None,
                 position: Union[SkyCoord, str] = None,
                 position_err: Union[float, un.Quantity, dict, PositionUncertainty, tuple] = 0.0 * un.arcsec,
                 host_galaxy: Galaxy = None):
        super().__init__(name=name,
                         position=position,
                         position_err=position_err)
        self.host_galaxy = host_galaxy

    @classmethod
    def from_dict(cls, dictionary: dict, name: str = None):
        frb = super().from_dict(dictionary=dictionary)
        frb.host_galaxy = Galaxy.from_dict(dictionary=dictionary["host_galaxy"])
        return frb

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        default_params.update({
            "host_galaxy": Galaxy.default_params(),
            "mjd": 58000
        })
        return default_params
