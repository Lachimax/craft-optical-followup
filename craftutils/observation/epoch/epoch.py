from craftutils.observation import field as fld


class Epoch:
    def __init__(self,
                 name: str = None,
                 field: fld.Field = None,
                 data_path: str = None,
                 ):
        self.name = name
        self.field = field
        self.data_path = data_path

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "instrument": None,
            "data_path": None,
        }
        return default_params
