from craftutils import utils as u
from craftutils.observation.instrument import Instrument
import craftutils.observation.filters as filters
from craftutils.observation.filters import FORS2Filter


@u.export
class ESOInstrument(Instrument):
    def filter_class(self):
        if self.name == "vlt-fors2":
            return FORS2Filter
        else:
            return filters.Filter
