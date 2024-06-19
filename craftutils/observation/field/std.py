import os
from typing import Union

from astropy.coordinates import SkyCoord

import craftutils.params as p
import craftutils.utils as u
import craftutils.astrometry as astm
import craftutils.observation.objects as objects

from .field import Field

class StandardField(Field):
    def __init__(
            self,
            centre_coords: Union[SkyCoord, str] = None,
            **kwargs
    ):
        jname = astm.jname(
            coord=centre_coords,
            ra_precision=0,
            dec_precision=0
        )
        name = f"STD-{jname}"

        param_path = os.path.join(p.param_dir, "fields", name, f"{name}.yaml")
        if not os.path.isfile(param_path):
            u.mkdir_check_nested(param_path)
            self.new_yaml(
                name=name,
                path=param_path,
                centre=objects.skycoord_to_position_dict(centre_coords)
            )

        super().__init__(
            name=name,
            centre_coords=centre_coords,
            data_path=os.path.join(p.data_dir, name),
            param_path=param_path
        )

        self.retrieve_catalogues()
