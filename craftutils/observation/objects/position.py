from typing import Union, List
import copy

import numpy as np

from astropy.coordinates import SkyCoord, Longitude
import astropy.units as units

import craftutils.utils as u


position_dictionary = {
    "alpha": {
        "decimal": 0.0,
        "hms": None},
    "delta": {
        "decimal": 0.0,
        "dms": None
    },
}

uncertainty_dict = {
    "sys": 0.0,
    "stat": 0.0
}

@u.export
def skycoord_to_position_dict(skycoord: SkyCoord):
    ra_float = skycoord.ra.value
    dec_float = skycoord.dec.value

    s = skycoord.to_string("hmsdms")
    ra = s[:s.find(" ")]
    dec = s[s.find(" ") + 1:]

    position = {
        "alpha": {"decimal": ra_float, "hms": ra},
        "delta": {"decimal": dec_float, "dms": dec},
    }

    return position


@u.export
class PositionUncertainty:
    def __init__(
            self,
            uncertainty: Union[float, units.Quantity, dict, tuple] = None,
            position: SkyCoord = None,
            ra_err_total: Union[float, units.Quantity] = None,
            ra_err_sys: Union[float, units.Quantity] = None,
            ra_err_stat: Union[float, units.Quantity] = None,
            dec_err_total: Union[float, units.Quantity] = None,
            dec_err_sys: Union[float, units.Quantity] = None,
            dec_err_stat: Union[float, units.Quantity] = None,
            a_stat: Union[float, units.Quantity] = None,
            a_sys: Union[float, units.Quantity] = None,
            b_stat: Union[float, units.Quantity] = None,
            b_sys: Union[float, units.Quantity] = None,
            theta: Union[float, units.Quantity] = None,
            sigma: float = None,
            **kwargs
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
            ra_key = None
            if "ra" in uncertainty and uncertainty["ra"] is not None:
                ra_key = "ra"
            elif "alpha" in uncertainty and uncertainty["alpha"] is not None:
                ra_key = "alpha"

            if ra_key is not None:
                if "sys" in uncertainty[ra_key] and uncertainty[ra_key]["sys"] is not None:
                    ra_err_sys = uncertainty[ra_key]["sys"]
                    if isinstance(ra_err_sys, str):
                        ra_err_sys = Longitude(ra_err_sys).to("arcsec")
                if "stat" in uncertainty[ra_key] and uncertainty[ra_key]["stat"] is not None:
                    ra_err_stat = uncertainty[ra_key]["stat"]
                    if isinstance(ra_err_stat, str):
                        ra_err_stat = Longitude(ra_err_stat).to("arcsec")

            dec_key = None
            if "dec" in uncertainty and uncertainty["dec"] is not None:
                dec_key = "dec"
            if "delta" in uncertainty and uncertainty["delta"] is not None:
                dec_key = "delta"

            if dec_key is not None:
                if "sys" in uncertainty[dec_key] and uncertainty[dec_key]["sys"] is not None:
                    dec_err_sys = uncertainty[dec_key]["sys"]
                if "stat" in uncertainty[dec_key] and uncertainty[dec_key]["stat"] is not None:
                    dec_err_stat = uncertainty[dec_key]["stat"]

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
            a_sys = 0.0 * units.arcsec
            b_stat = uncertainty
            b_sys = 0.0 * units.arcsec
            theta = 0.0 * units.deg

        if ra_err_stat is None and "alpha_err_stat" in kwargs and kwargs["alpha_err_stat"] is not None:
            ra_err_stat = (kwargs["alpha_err_stat"] / np.cos(position.dec)).to("arcsec")
        if ra_err_sys is None and "alpha_err_sys" in kwargs and kwargs["alpha_err_sys"] is not None:
            ra_err_sys = (kwargs["alpha_err_sys"] / np.cos(position.dec)).to("arcsec")
        if dec_err_stat is None and "delta_err_stat" in kwargs and kwargs["delta_err_stat"] is not None:
            dec_err_stat = kwargs["delta_err_stat"]
        if dec_err_sys is None and "delta_err_sys" in kwargs and kwargs["delta_err_sys"] is not None:
            dec_err_sys = kwargs["delta_err_sys"]

        # Check whether we're specifying uncertainty using equatorial coordinates or ellipse parameters.
        u.debug_print(2, "PositionUncertainty.__init__(): a_stat, a_sys, b_stat, b_sys, theta, position ==", a_stat,
                      a_sys, b_stat, b_sys, theta, position)
        u.debug_print(2, "PositionUncertainty.__init__(): ra_err_sys, ra_err_stat, dec_err_sys, dec_err_stat ==",
                      ra_err_sys, ra_err_stat, dec_err_sys, dec_err_stat)
        if a_stat is not None and a_sys is not None and b_stat is not None and b_sys is not None and theta is not None:
            ellipse = True
        elif ra_err_sys is not None and ra_err_stat is not None and dec_err_sys is not None and dec_err_stat is not None and position is not None:
            ellipse = False
        else:
            raise ValueError(
                "Either all ellipse values (a, b, theta) or all equatorial values (ra, dec, position) must be provided.")

        ra_err_sys = u.check_quantity(number=ra_err_sys, unit=units.arcsec)
        ra_err_stat = u.check_quantity(number=ra_err_stat, unit=units.arcsec)
        dec_err_sys = u.check_quantity(number=dec_err_sys, unit=units.arcsec)
        dec_err_stat = u.check_quantity(number=dec_err_stat, unit=units.arcsec)
        # Convert equatorial uncertainty to ellipse with theta=0
        if not ellipse:
            ra = position.ra
            dec = position.dec
            a_sys = ra_err_sys * np.cos(dec)
            a_stat = ra_err_stat * np.cos(dec)
            b_sys = dec_err_sys
            b_stat = dec_err_stat
            if b_sys > a_sys:
                theta = 90. * units.deg
            else:
                theta = 0. * units.degree
            a_sys, b_sys = max(a_sys, b_sys), min(a_sys, b_sys)
            a_stat, b_stat = max(a_stat, b_stat), min(a_stat, b_stat)
        # Or use ellipse parameters as given.
        else:
            a_sys = u.check_quantity(number=a_sys, unit=units.arcsec)
            a_stat = u.check_quantity(number=a_stat, unit=units.arcsec)
            b_sys = u.check_quantity(number=b_sys, unit=units.arcsec)
            b_stat = u.check_quantity(number=b_stat, unit=units.arcsec)
            theta = u.check_quantity(number=theta, unit=units.deg)

        self.a_sys = a_sys
        self.a_stat = a_stat
        self.b_sys = b_sys
        self.b_stat = b_stat
        self.theta = theta
        print(f"Position: {theta=}")

        self.ra_sys = ra_err_sys
        self.dec_sys = dec_err_sys
        self.ra_stat = ra_err_stat
        self.dec_stat = dec_err_stat

    def __str__(self):
        return f"PositionUncertainty: a_stat={self.a_stat}, b_stat={self.b_stat}; a_sys={self.a_sys}, b_sys={self.b_sys}"

    def uncertainty_quadrature(self):
        a_quad = np.sqrt(self.a_sys ** 2 + self.a_stat ** 2)
        b_quad = np.sqrt(self.b_sys ** 2 + self.b_stat ** 2)
        return max(a_quad, b_quad), min(a_quad, b_quad)

    def uncertainty_quadrature_equ(self):
        return np.sqrt(self.ra_sys ** 2 + self.ra_stat ** 2), np.sqrt(self.dec_sys ** 2 + self.dec_stat ** 2)

    # TODO: Finish this

    def to_dict(self):
        return {
            "a_sys": self.a_sys,
            "a_stat": self.a_stat,
            "b_sys": self.b_sys,
            "b_stat": self.b_stat,
            "theta": self.theta,
            "alpha_err_sys": self.ra_sys,
            "delta_err_sys": self.dec_sys,
            "alpha_err_stat": self.ra_stat,
            "delta_err_stat": self.dec_stat
        }

    @classmethod
    def default_params(cls):
        return {
            "alpha": copy.deepcopy(uncertainty_dict),
            "delta": copy.deepcopy(uncertainty_dict),
            "a": copy.deepcopy(uncertainty_dict),
            "b": copy.deepcopy(uncertainty_dict),
            "theta": 0.0,
            "sigma": None,
            "healpix_path": None
        }
