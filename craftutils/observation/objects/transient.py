import astropy.time as time

import craftutils.params as p
import craftutils.utils as u

from .objects import Object
from .transient_host import TransientHostCandidate


@u.export
class Transient(Object):
    optical = False

    def __init__(
            self,
            host_galaxy: TransientHostCandidate = None,
            date: time.Time = None,
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
            self.z = kwargs["z"]
        if "z_err" in kwargs:
            self.z_err = kwargs["z"]
        self.host_galaxy = host_galaxy
        self.host_candidate_tables = {}
        self.host_candidates = []
        if not isinstance(date, time.Time) and date is not None:
            date = time.Time(date)
        self.date = date
        self.tns_name = None
        if "tns_name" in kwargs:
            self.tns_name = kwargs["tns_name"]

    def get_host(self, name: str = None) -> TransientHostCandidate:
        """If `self.host_galaxy` is a string, checks for a host galaxy with that in the FRB's field and sets
        `self.host_galaxy` to that object.
        If `self.host_galaxy` is `None`, sets it to an empty `TransientHostCandidate` with the same `z` and `z_err`.

        :return: The Galaxy or TransientHostCandidate object.
        """
        if self.host_galaxy is None:
            self.host_galaxy = TransientHostCandidate(
                transient=self,
                z=self.z,
                z_err=self.z_err,
                name=name
            )
        elif isinstance(self.host_galaxy, str) and self.field:
            self.host_galaxy = self._get_object(self.host_galaxy)

        return self.host_galaxy

    def update_param_file(self, param: str):
        p_dict = {
            "host_galaxy": self.host_galaxy.name,
        }
        if param not in p_dict:
            raise ValueError(f"Either {param} is not a valid parameter, or it has not been configured.")
        if self.param_path is None:
            raise ValueError("param_path has not been set.")
        else:
            params = p.load_params(self.param_path)
        params[param] = p_dict[param]
        p.save_params(file=self.param_path, dictionary=params)
