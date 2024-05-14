import astropy.units as units

from craftutils.observation.objects.frb import dm_units

from .objects import ObjectCatalogue

class FRBCatalogue(ObjectCatalogue):
    @classmethod
    def column_names(cls):
        columns = super().column_names()
        columns.update({
            "dm": dm_units,
            "date": str,
            "instrument": str,
            "tns_name": str,
            "survey": str,
            "snr": float,
            "z": float
        })