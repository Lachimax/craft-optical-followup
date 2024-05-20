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
            date: time.Time = None,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
        if not isinstance(date, time.Time) and date is not None:
            date = time.Time(date)
        self.date = date
        self.tns_name = None
        if "tns_name" in kwargs:
            self.tns_name = kwargs["tns_name"]
