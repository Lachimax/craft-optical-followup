import os

import astropy.table as table

import craftutils.params as p

from .output import OutputCatalogue


class FurbyCatalogue(OutputCatalogue):
    output_format = "csv"

    def __init__(self, **kwargs):
        super().__init__(
            data_path=os.path.join(p.furby_path, "craco_fu", "data"),
            name="craco_fu_db",
            **kwargs
        )

    def add_entry(
            self,
            key: str,
            entry: dict,
    ):
        pass

    def load_table(
            self,
            force: bool = False,
            table_path: str = None
    ):
        if force or self.table is None:
            path = self.table_path
            if path is not None:
                _, dtypes, _ = self.construct_column_lists()
                self.table = table.Table.read(path, format="ascii.csv", dtype=dtypes)
            else:
                self.table = None

        return self.table

    @classmethod
    def column_names(cls):
        return {
            "Name": str,
            "RA_FRB": float,
            "DEC_FRB": float,
            "JNAME": str,
            "DM_FRB": float,
            "sig_ra": float,
            "sig_dec": float,
            "CRACO": bool,
            "pass_loc": bool,
            "DM_ISM": float,
            "pass_MW": bool,
            "EBV": float,
            "pass_star": bool,
            "pass_img": bool,
            "R_OB": bool,
            "K_OB": bool,
            "R_obs": bool,
            "K_obs": bool,
            "R_rdx": bool,
            "K_rdx": bool,
            "R_UT": str,
            "K_UT": str,
            "PATH": bool,
            "max_POx": float,
            "RA_Host": float,
            "DEC_Host": float
        }


furby_cat = FurbyCatalogue()
