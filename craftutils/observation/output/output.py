import os
import copy
from typing import Union

import astropy.units as units
import astropy.table as table

import craftutils.params as p
import craftutils.utils as u

from ..generic import Generic

config = p.config
if config["table_dir"] is not None:
    u.mkdir_check_nested(config["table_dir"])


@u.export
def split_dtype(val, replace_str: bool = True):
    if isinstance(val, units.Unit) or isinstance(val, units.IrreducibleUnit):
        dtype = units.Quantity
        unit = val
    else:
        dtype = val
        unit = None
    if dtype is str and replace_str:
        dtype = "U32"

    return dtype, unit


class OutputCatalogue(Generic):
    output_format = "yaml"

    def __init__(
            self,
            table_dir: str = None,
            data_path: str = None,
            **kwargs
    ):
        if data_path is None:
            data_dir = os.path.dirname(config["table_dir"])
            data_path = os.path.join(data_dir, "table_outputs")

        super().__init__(
            data_path=data_path,
            **kwargs
        )
        self.table = None
        self.astropy_table = None
        self.filename = self.name + "." + self.output_format
        self.table_dir = table_dir
        if self.table_dir is None:
            self.table_dir = config["table_dir"]
        self.table_path = self.build_table_path()
        os.makedirs(self.table_dir, exist_ok=True)
        self.colnames = None
        self.dtypes = None
        self.units = None
        self.template = {}


    def build_table_path(self):
        return os.path.join(self.table_dir, self.filename)

    def load_table(
            self,
            force: bool = False,
    ):
        self.load_output_file()
        if self.template is None:
            self.template = self.build_default()

        if force or self.table is None:
            self.table = {}
            if os.path.isfile(self.table_path):
                tbl_dict = p.load_params(self.table_path)
                for name, entry in tbl_dict.items():
                    self.add_entry(key=name, entry=entry)
        if self.table is None:
            self.table = {}
        return self.table

    def to_astropy(
            self,
            sort_by: Union[str, list] = "field_name",
    ):
        if not isinstance(sort_by, list):
            sort_by = [sort_by]
        tbl = copy.deepcopy(self.table)
        if "template" in tbl:
            tbl.pop("template")
        tbl_list = list(tbl.values())
        tbl_astropy = table.QTable(tbl_list)

        # For FRB fields, try to make the field names match the TNS name (if it exists)
        if "transient_tns_name" in tbl_astropy.colnames:
            change_dict = {}
            for row in tbl_astropy:
                if row["transient_tns_name"] != "N/A" \
                        and row["field_name"].startswith("FRB") \
                        and row["transient_tns_name"].startswith("FRB"):
                    change_dict[row["field_name"]] = row["transient_tns_name"]
            # Some objects in an FRB field will not have an associated TNS name (non-host objects of interest)
            # so we loop again
            for row in tbl_astropy:
                if row["field_name"] in change_dict:
                    row["field_name"] = change_dict[row["field_name"]]

        tbl_names = tbl_astropy.colnames
        names = sort_by.copy()
        names.reverse()
        for name in names:
            if name in tbl_names:
                tbl_names.remove(name)
            tbl_names.insert(0, name)
        tbl_astropy = tbl_astropy[tbl_names]
        tbl_astropy.sort(sort_by)
        for i, row in enumerate(tbl_astropy):
            if None in row:
                print("Removing row", i, "from table", self.name, "because it had a None in it")
                tbl_astropy.remove_row(i)
        # colname, column = u.detect_problem_column(tbl_astropy)
        self.astropy_table = tbl_astropy
        return tbl_astropy

    def write_table(
            self,
            sort_by: Union[str, list] = None,
    ):
        self.load_output_file()
        self.update_entries()
        tbl = self.table
        tbl_path = self.table_path
        if sort_by is None:
            sort_by = self.required()
        if tbl is None:
            raise ValueError(f"Table {self.name} not loaded.")
        p.save_params(tbl_path, tbl)
        tbl_astropy = self.to_astropy(sort_by=sort_by)
        tbl_astropy.write(
            tbl_path.replace(".yaml", ".ecsv"),
            format="ascii.ecsv",
            overwrite=True
        )
        # u.detect_problem_row(tbl_astropy)
        tbl_astropy.write(
            tbl_path.replace(".yaml", ".csv"),
            format="ascii.csv",
            overwrite=True
        )
        self.update_output_file()

    def update_entries(self):
        for name, row in self.table.items():
            for colname in self.template:
                if name == "HG20190608B":
                if colname not in row or row[colname] == 0.0:
                    row[colname] = self.template[colname]

    def add_entry(
            self,
            key: str,
            entry: dict,
    ):
        for item in self.required():
            if item not in entry:
                raise ValueError(f"Required key {item} not found in entry.")

        for colname in entry:
            if colname not in self.template:
                if isinstance(entry[colname], units.Quantity):
                    self.template[colname] = -999 * entry[colname].unit
                elif isinstance(entry[colname], str):
                    self.template[colname] = "N/A"
                else:
                    self.template[colname] = type(entry[colname])(-999)

        self.table[key] = entry

        for name, other_entry in self.table.items():
            for colname in self.template:
                if colname not in other_entry:
                    other_entry[colname] = self.template[colname]

        return entry

    def construct_column_lists(
            self,
            replace_str: bool = True
    ):
        columns = self.column_names()
        dtypes = []
        un = []
        colnames = []
        columns_revised = {}
        for colname in columns:
            # if "{:s}" in colname:
            #     for fil in filters:
            #         colname_fil = colname.format(fil)
            #         colnames.append(colname_fil)
            #         columns_revised[colname_fil] = columns[colname]

            if "{:s}" not in colname:
                colnames.append(colname)
                columns_revised[colname] = columns[colname]

        for colname in colnames:
            val = columns_revised[colname]

            dtype, unit = split_dtype(val, replace_str=replace_str)
            dtypes.append(dtype)
            un.append(unit)

        self.colnames = colnames
        self.dtypes = dtypes
        self.units = units
        return colnames, dtypes, un

    def get_entry(self, key: str):
        return self.table[key]

    def get_row(self, colname: str, colval: str):
        if self.astropy_table is None:
            self.to_astropy()
        is_name = self.astropy_table[colname] == colval
        if sum(is_name) == 0:
            return None, None
        else:
            return self.astropy_table[is_name][0], is_name.argmax()

    def build_default(
            self
    ):
        """Constructs a default 'table' entry.

        :return:
        """
        colnames, dtypes, un = self.construct_column_lists()
        default = {}
        for i in range(len(colnames)):
            colname = colnames[i]
            dtype = dtypes[i]
            unit = un[i]
            if unit is not None:
                default_val = -999 * unit
            elif dtype is str:
                default_val = "N/A"
            else:
                default_val = dtype(-999)
            default[colname] = default_val
        return default

    def _output_dict(self):
        output_dict = super()._output_dict()
        output_dict.update({
            "template": self.template,
        })
        return output_dict

    def load_output_file(self, **kwargs):
        output = super().load_output_file(**kwargs)
        if "template" in output:
            self.template = output["template"]

    @classmethod
    def column_names(cls):
        return {"field_name": str}

    @classmethod
    def required(cls):
        return ["field_name"]
