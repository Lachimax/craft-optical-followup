import astropy.table as table

from .survey import *
from .image import *
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
        self.path = p.check_abs_path(path)
        return self.path

    def set_table_path(self, path: str):
        self.table_path = p.check_abs_path(path)
        return self.table_path

    def write(self):
        self.table.write()
