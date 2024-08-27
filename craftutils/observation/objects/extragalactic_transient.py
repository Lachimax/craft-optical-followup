import astropy.units as units

import craftutils.utils as u

from .transient import Transient
from .extragalactic import Extragalactic, cosmology
from .transient_host import TransientHostCandidate


class ExtragalacticTransient(Transient, Extragalactic):

    def __init__(
            self,
            host_galaxy: TransientHostCandidate = None,
            **kwargs
    ):

        super().__init__(
            **kwargs
        )
        if isinstance(host_galaxy, str):
            hg = self._get_object(host_galaxy)
            if hg:
                host_galaxy = hg
        self.z = None
        self.z_err = None
        if "z" in kwargs:
            z = kwargs["z"]
            z_err = None
            if "z_err" in kwargs:
                z_err = kwargs["z_err"]
            self.set_z(z=z, z_err=z_err)

        self.host_galaxy = host_galaxy
        self.host_candidate_tables = {}
        self.host_candidates = []

    def get_host(self, name: str = None) -> TransientHostCandidate:
        """If `self.host_galaxy` is a string, checks for a host galaxy with that in the transient's field and sets
        `self.host_galaxy` to that object.
        If `self.host_galaxy` is `None`, sets it to an empty `TransientHostCandidate` with the same `z` and `z_err`.

        :return: The Galaxy or TransientHostCandidate object.
        """
        from craftutils.observation.field import Field
        if self.host_galaxy is None:
            self.host_galaxy = TransientHostCandidate(
                transient=self,
                z=self.z,
                z_err=self.z_err,
                name=name
            )
        elif isinstance(self.host_galaxy, str) and isinstance(self.field, Field):
            self.host_galaxy = self._get_object(self.host_galaxy)

        return self.host_galaxy

    def deprojected_host_offset(
            self
    ):
        from .galaxy import Galaxy
        if (not isinstance(self.host_galaxy, Galaxy)
                or not isinstance(self.host_galaxy.inclination, units.Quantity)
                or not isinstance(self.host_galaxy.position_angle, units.Quantity)
        ):
            return None
        return u.deprojected_offset(
            object_coord=self.position,
            galaxy_coord=self.host_galaxy.position,
            position_angle=self.host_galaxy.position_angle,
            inc=self.host_galaxy.inclination
        )

    def _updateable(self):
        p_dict = super()._updateable()
        p_dict.update({
            "host_galaxy": self.host_galaxy.name,
        })
        return p_dict
