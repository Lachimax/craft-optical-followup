import astropy.time as time
from typing import Union

import craftutils.observation.field as fld
import craftutils.utils as u


class Epoch:
    def __init__(self,
                 name: str = None,
                 field: fld.Field = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, time.Time] = None,
                 obj: str = None,
                 program_id: str = None
                 ):
        self.name = name
        self.field = field
        self.data_path = data_path
        u.mkdir_check(data_path)
        self.instrument = instrument
        self.date = date
        if type(self.date) is str:
            self.date = time.Time(date)
        self.obj = obj
        self.program_id = program_id

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "field": None,
            "data_path": None,
            "instrument": None,
            "date": None,
            "obj": None,
            "program_id": None
        }
        return default_params
