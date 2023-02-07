import astropy.table as table

from .survey import *
from .se import *
import craftutils.utils as u
import craftutils.params as p


class Catalogue:

    def __init__(
            self,
            **kwargs
    ):
        self.path: str = None
        if "path" in kwargs:
            self.set_path(kwargs["path"])
        self.table_path: str = None
        if "table_path" in kwargs:
            self.set_table_path(kwargs["table_path"])
        self.table: table.QTable = None
        if "table" in kwargs:
            self.table = table.QTable(kwargs["table"])

        self.load_table()
        self.load_output_file()

    def __len__(self):
        return len(self.table)

    def __getitem__(self, *items):
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
                raise ValueError(f"For {self.path}, table_path has not been set.")

    def set_path(self, path: str):
        self.path = u.sanitise_file_ext(
            p.check_abs_path(path),
            ".yaml"
        )
        return self.path

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
                self.table_path = self.path.replace(".yaml", "_source_cat.ecsv")
            u.debug_print(1, "Writing source catalogue to", self.table_path)
            self.table.write(self.table_path, format="ascii.ecsv", overwrite=True)

    def load_output_file(self):
        outputs = p.load_output_file(self)
        if outputs is not None:
            self.__dict__.update(outputs)

    @classmethod
    def _do_not_include_in_output(cls):
        return ["path", "table", "do_not_include_in_output"]

    def _output_dict(self):
        outputs = self.__dict__.copy()
        dni = self._do_not_include_in_output()
        for key in outputs:
            if key in dni:
                outputs[key].pop()
        return outputs

    def update_output_file(self):
        p.update_output_file(self)
        self.write()
