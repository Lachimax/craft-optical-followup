from typing import Union

import astropy.units as units

from .output import OutputCatalogue

object_tables = {}


class ObjectCatalogue(OutputCatalogue):

    def __init__(self, object_type, **kwargs):
        self.object_type = object_type
        super().__init__(
            name=f"master_table_{self.object_type}",
            **kwargs
        )

    @classmethod
    def column_names(cls):
        columns = super().column_names()
        columns.update({
            "transient_tns_name": str,
            "object_name": str,
            "jname": str,
            "ra": units.deg,
            "ra_err": units.deg,
            "dec": units.deg,
            "dec_err": units.deg,
            "a": units.arcsec,
            "b": units.arcsec,
            "theta": units.deg,
        })

    @classmethod
    def required(cls):
        required = super().required()
        required += [
            "object_name",
            "ra",
            "dec"
        ]
        return required


def load_objects_table(
        object_type: Union[type, str],
        force: bool = False,
) -> ObjectCatalogue:
    if isinstance(object_type, type):
        object_type = object_type.__name__

    if object_type == "optical":
        from .optical import OpticalCatalogue
        cls = OpticalCatalogue
    elif object_type == "frb":
        from .frb import FRBCatalogue
        cls = FRBCatalogue
    else:
        cls = ObjectCatalogue

    if force or object_type not in object_tables:
        object_tables[object_type] = cls(object_type=object_type)

    object_tables[object_type].load_table(force=force)

    return object_tables[object_type]
