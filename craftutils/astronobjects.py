from typing import Union, Tuple

from astropy.coordinates import SkyCoord
from astropy import units as un

from craftutils import astrometry as a
from craftutils import utils as u

uncertainty_dict = {"sys": 0.0,
                    "stat": 0.0}

position_uncertainty_dict = {"ra": uncertainty_dict.copy(),
                             "dec": uncertainty_dict.copy(),
                             "a": uncertainty_dict.copy(),
                             "b": uncertainty_dict.copy(),
                             "theta": 0.0}


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
                 theta: Union[float, un.Quantity] = None
                 ):
        """
        If a single value is provided for uncertainty, the uncertainty ellipse will be assumed to be circular.
        Any angular values provided without units are assumed to be in degrees.
        Values in dictionary, if provided, override values given as arguments.
        :param uncertainty:
        :param ra_err_sys:
        :param ra_err_stat:
        :param dec_err_sys:
        :param dec_err_stat:
        """
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
            b_sys = SkyCoord(f'{ra}d {dec}d').separation(SkyCoord(f'{ra}d {dec + dec_err_sys}d')).value
            b_stat = SkyCoord(f'{ra}d {dec}d').separation(SkyCoord(f'{ra}d {dec + dec_err_stat}d')).value
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


class Object:
    def __init__(self,
                 position: Union[SkyCoord, str] = None):
        self.position = a.attempt_skycoord(position)


class FRB(Object):
    def __init__(self,
                 position: Union[SkyCoord, str] = None,
                 position_err: Union[float, un.Quantity, dict, PositionUncertainty, tuple] = 0.0 * un.arcsec,
                 ):
        """
        Any angular values provided without units will be assumed to be in degrees.
        :param position:
        :param position_err:
        """

        if type(position_err) is not PositionUncertainty:
            self.position_err = PositionUncertainty(uncertainty=position_err)

        super(FRB, self).__init__(position=position
                                  )
