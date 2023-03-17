from craftutils import utils as u
from craftutils.observation.instrument import Instrument
import craftutils.observation.filters as filters

@u.export
class ESOInstrument(Instrument):
    def filter_class(self):
        if self.name == "vlt-fors2":
            return filters.FORS2Filter
        else:
            return filters.Filter
