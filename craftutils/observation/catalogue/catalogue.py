import astropy.table as table
from astropy.coordinates import SkyCoord

import craftutils.utils as u
import craftutils.params as p

__all__ = []


@u.export
class Catalogue:
    ra_key = "ra"
    dec_key = "dec"
    def __init__(
            self,
            path: str,
            **kwargs
    ):
        self.output_file: str = None
        # if "path" in kwargs and kwargs["path"] is not None:
        self.set_path(path=path)
        # elif "output_file" in kwargs and kwargs["output_file"] is not None:
        #     self.set_path(kwargs["output_file"])
        self.table_path: str = None
        if "table_path" in kwargs:
            self.set_table_path(kwargs["table_path"])
        self.table: table.QTable = None
        if "table" in kwargs:
            self.table = table.QTable(kwargs["table"])

        if self.table_path is not None:
            self.load_table()
            self.load_output_file()

    def __len__(self):
        return len(self.table)

    def __getitem__(self, *items):
        if len(items) == 1:
            return self.table[items[0]]
        else:
            return self.table[items]

    def __setitem__(self, key, value):
        self.table[key] = value

    def load_table(self, force: bool = False, table_path: str = None):
        if table_path is not None:
            self.set_table_path(path=table_path)
        u.debug_print(2, f"Catalogue.load_source_cat(): {self}.source_cat_path ==", self.table_path)
        if force or self.table is None:
            if self.table_path is not None:
                u.debug_print(1, "Loading source_table from", self.table_path)
                self.table = table.QTable.read(self.table_path, format="ascii.ecsv")
            else:
                raise ValueError(f"For {self.output_file}, table_path has not been set.")

    def set_path(self, path: str):
        if path.endswith(".ecsv"):
            self.table_path = path
            path = path.replace(".ecsv", ".yaml")
        self.output_file = u.sanitise_file_ext(
            p.check_abs_path(path),
            ".yaml"
        )
        return self.output_file

    def set_table_path(self, path: str, load: bool = True):
        self.table_path = p.check_abs_path(path)
        if load:
            self.load_table(force=True)
        return self.table_path

    def write(self, path: str = None):
        if path is not None:
            self.set_table_path(path=path, load=False)
        if self.table is None:
            u.debug_print(1, "table not yet loaded.")
        else:
            if self.table_path is None:
                self.table_path = self.output_file.replace(".yaml", "_source_cat.ecsv")
            u.debug_print(1, "Writing source catalogue to", self.table_path)
            self.table.write(self.table_path, format="ascii.ecsv", overwrite=True)

    def load_output_file(self):
        outputs = p.load_output_file(self)
        if outputs is not None:
            self.__dict__.update(outputs)

    def to_skycoord(self):
        return SkyCoord(self.table[self.ra_key], self.table[self.dec_key])

    @classmethod
    def _do_not_include_in_output(cls):
        return ["path", "table", "do_not_include_in_output"]

    def _output_dict(self):
        outputs = self.__dict__.copy()
        dni = self._do_not_include_in_output()
        for key in dni:
            if key in outputs:
                outputs.pop(key)
        return outputs

    def update_output_file(self):
        p.update_output_file(self)
        self.write()

    def sort(self, **kwargs):
        self.table.sort(**kwargs)