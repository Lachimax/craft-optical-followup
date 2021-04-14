from typing import Union

from astropy.coordinates import SkyCoord

from craftutils import astrometry as a


class Object:
    def __init__(self,
                 coords: Union[SkyCoord, str] = None):
        self.coords = a.attempt_skycoord(coords)


class FRB(Object):
    def __init__(self):
        super(FRB, self).__init__()
