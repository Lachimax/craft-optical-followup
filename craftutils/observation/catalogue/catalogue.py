import os.path

import astropy.table as table
from astropy.coordinates import SkyCoord

import craftutils.utils as u
import craftutils.params as p

__all__ = []

active_catalogues = {}

@u.export
class Catalogue:
    ra_key = "ra"
    dec_key = "dec"

    def __init__(
            self,
            path: str,
            **kwargs
    ):
        self.name = None
        self.output_file: str = None
        self.data_path = None
        # if "path" in kwargs and kwargs["path"] is not None:
        # elif "output_file" in kwargs and kwargs["output_file"] is not None:
        #     self.set_path(kwargs["output_file"])
        self.path: str = None
        self.set_table_path(path, load=False)
        self.table: table.QTable = None
        self.load_output_file()
        if self.path is not None:
            self.load_table()

    def __len__(self):
        return len(self.table)

    def __getitem__(self, *items):
        print(self.path)
        self.load_table()
        if len(items) == 1:
            return self.table[items[0]]
        else:
            return self.table[items]

    def __setitem__(self, key, value):
        self.table[key] = value

    def load_table(self, force: bool = False, table_path: str = None):
        if table_path is not None:
            self.set_table_path(path=table_path)
        u.debug_print(2, f"Catalogue.load_source_cat(): {self}.source_cat_path ==", self.path)
        if force or self.table is None:
            if self.path is not None:
                u.debug_print(1, "Loading source_table from", self.path)
                self.table = table.QTable.read(self.path, format="ascii.ecsv")
            else:
                raise ValueError(f"For {self.output_file}, table_path has not been set.")

    def set_table_path(self, path: str, load: bool = True):
        path = p.check_abs_path(path)
        if path.endswith("_outputs.yaml"):
            path = path.replace("_outputs.yaml", ".ecsv")
        self.path = path
        self.data_path, self.name = os.path.split(self.path[:-5])
        if load:
            self.load_table(force=True)
        return self.path

    def write(self):
        if self.table is None:
            u.debug_print(1, "table not yet loaded.")
        else:
            u.debug_print(1, "Writing source catalogue to", self.path)
            self.table.write(self.path, format="ascii.ecsv", overwrite=True)

    def load_output_file(self):
        outputs = p.load_output_file(self)
        if outputs is not None:
            self.__dict__.update(outputs)

    def to_skycoord(self):
        return SkyCoord(self.table[self.ra_key], self.table[self.dec_key])

    @classmethod
    def _do_not_include_in_output(cls):
        return ["table", "do_not_include_in_output", "output_file"]

    def _output_dict(self):
        outputs = self.__dict__.copy()
        dni = self._do_not_include_in_output()
        for key in dni:
            if key in outputs:
                outputs.pop(key)
        return outputs

    def update_output_file(self):
        self.write()
        p.update_output_file(self)

    def sort(self, **kwargs):
        self.table.sort(**kwargs)
