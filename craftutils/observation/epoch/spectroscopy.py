from craftutils.observation import field as fld


class SpectroscopyEpoch:
    def __init__(self,
                 name: str = None,
                 field: fld.Field = None,
                 data_path: str = None,
                 ):
        self.name = name
        self.field = field
        self.data_path = data_path
