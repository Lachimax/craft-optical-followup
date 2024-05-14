from typing import Union

import astropy.units as units
import astropy.cosmology as cosmo

import craftutils.utils as u
from craftutils.photometry import distance_modulus

from .objects import Object

cosmology = cosmo.Planck18


@u.export
def set_cosmology(cos: Union[str, cosmo.Cosmology]):
    global cosmology
    if isinstance(cos, str):
        if cos in cosmo.available:
            cosmology = getattr(cosmo, cos)
        else:
            raise ValueError(f"Cosmology {cos} not found in `astropy.cosmology`. Available are: {cosmo.available}")
    elif isinstance(cos, cosmo.Cosmology):
        cosmology = cos
    else:
        raise TypeError("cos must be string or `astropy.cosmology.Cosmology`.")


@u.export
class Extragalactic(Object):
    optical = True

    def __init__(
            self,
            z: float = None,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.z = None
        self.z_err = None
        self.D_A = None
        self.D_L = None
        self.D_comoving = None
        self.mu = None
        self.set_z(z, **kwargs)

    def set_z(self, z: float, **kwargs):
        self.z = z
        if "z_err" in kwargs:
            self.z_err = kwargs["z_err"]
        self.D_A = self.angular_size_distance()
        self.D_L = self.luminosity_distance()
        self.D_comoving = self.comoving_distance()
        self.mu = self.distance_modulus()

    def angular_size_distance(self):
        if self.z is not None:
            return cosmology.angular_diameter_distance(z=self.z)

    def luminosity_distance(self):
        if self.z is not None:
            return cosmology.luminosity_distance(z=self.z)

    def comoving_distance(self):
        if self.z is not None:
            return cosmology.comoving_distance(z=self.z)

    def distance_modulus(self):
        d = self.luminosity_distance()
        if d is not None:
            mu = distance_modulus(d)
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
                for epoch in self.photometry[instrument][fil]:
                    abs_mag = self.absolute_magnitude(
                        apparent_magnitude=self.photometry[instrument][fil][epoch]["mag"],
                        internal_extinction=internal_extinction
                    )
                    self.photometry[instrument][fil][epoch]["abs_mag"] = abs_mag
        self.update_output_file()

    def projected_size(self, angle: Union[units.Quantity, float]) -> Union[units.Quantity, None]:
        """
        When given an angular size, calculates the projected physical size at the redshift of the galaxy.
        :param angle: Angular size. If not provided as a quantity, must be in arcseconds.
        :return: Projected physical size, with units kpc
        """
        if self.D_A is None:
            return None
        angle = u.check_quantity(angle, unit=units.arcsec).to(units.rad).value
        dist = angle * self.D_A
        return dist.to(units.kpc)

    def angular_size(self, distance: Union[units.Quantity, float]):
        """
        Given a physical projected size at the redshift of the galaxy, calculates the angular size as seen from Earth.
        :param distance: Physical projected size. If not provided as a quantity, must be in kiloparsecs.
        :return: Angular size, in arcseconds.
        """
        distance = u.check_quantity(distance, unit=units.kpc)
        theta = (distance * units.rad / self.D_A).to(units.arcsec)
        return theta

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        default_params.update({
            "z": None,
            "z_err": None,
        })
        return default_params
