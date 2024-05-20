# Code by Lachlan Marnoch, 2021 - 2024

import os
from typing import Union, List, Dict

from astropy.time import Time
from astropy.coordinates import SkyCoord

import craftutils.params as p
import craftutils.observation.field as fld
import craftutils.observation.objects as objects
import craftutils.observation.image as image
from craftutils.observation.instrument import Instrument

from .epoch import Epoch


class StandardEpoch(Epoch):
    instrument_name = "dummy-instrument"

    def __init__(
            self,
            centre_coords: SkyCoord,
            instrument: str,
            frames_standard: Dict[str, List[image.ImagingImage]] = {},
            frames_flat: Dict[str, List[image.ImagingImage]] = {},
            frames_bias: List[image.ImagingImage] = [],
            date: Union[str, Time] = None,
            **kwargs
    ):
        field = fld.StandardField(centre_coords=centre_coords)
        name = f"{field.name}_{date.strftime('%Y-%m-%d')}"
        param_path = os.path.join(p.param_dir, "fields", field.name, "imaging", f"{name}.yaml")

        if not os.path.isfile(param_path):
            self.new_yaml(
                name=name,
                path=param_path,
                centre=objects.skycoord_to_position_dict(centre_coords)
            )

        super().__init__(
            param_path=param_path,
            name=name,
            field=field,
            data_path=os.path.join(field.data_path, "imaging", str(instrument), name),
            instrument=str(instrument),
            date=date,
            **kwargs
        )

        self.frames_standard = frames_standard
        self.frames_bias = frames_bias
        self.frames_flat = frames_flat

        self.load_output_file()

    @classmethod
    def select_child_class(cls, instrument: Union[str, Instrument]):
        if isinstance(instrument, Instrument):
            instrument = instrument.name
        if instrument == "vlt-fors2":
            from craftutils.observation.epoch.imaging.eso.vlt_fors2.std import FORS2StandardEpoch
            return FORS2StandardEpoch
        else:
            return StandardEpoch
