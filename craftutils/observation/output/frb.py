import astropy.units as units

from .objects import ObjectCatalogue


class FRBCatalogue(ObjectCatalogue):
    @classmethod
    def column_names(cls):
        columns = super().column_names()
        columns.update({

        })