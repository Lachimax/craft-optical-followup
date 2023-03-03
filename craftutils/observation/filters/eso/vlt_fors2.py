import os
from datetime import date

from astropy import table as table, units as units

from craftutils import utils as u
from craftutils.observation import filters as filters
from craftutils.retrieve import save_fors2_calib


class FORS2Filter(filters.Filter):
    qc1_retrievable = ['b_HIGH', 'v_HIGH', 'R_SPECIAL', 'I_BESS']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calibration_table = None
        self.calibration_table_path = None
        self.calibration_table_last_updated = None

    def load_calibration_table(self, force: bool = False):
        if self.calibration_table_path is not None:
            if force:
                self.calibration_table = None
            if self.calibration_table is None:
                self.calibration_table = table.QTable.read(self.calibration_table_path)
        else:
            print("calibration_table could not be loaded because calibration_table_path has not been set.")

    def write_calibration_table(self):
        if self.calibration_table is None:
            u.debug_print(1, "calibration_table not yet loaded.")
        else:
            if self.calibration_table_path is None:
                self.calibration_table_path = os.path.join(
                    self.data_path,
                    f"{self.instrument.name}_{self.name}_calibration_table.ecsv")
            u.debug_print(1, "Writing calibration table to", self.calibration_table_path)
            self.calibration_table.write(self.calibration_table_path, format="ascii.ecsv", overwrite=True)

    def calib_retrievable(self):
        return self.name in self.qc1_retrievable

    def retrieve_calibration_table(self, force=False):

        if self.calib_retrievable():
            if self.calibration_table_last_updated != date.today() or force:
                down_path = os.path.join(self.data_path, "fors2_qc.tbl")
                fil = self.name
                if fil == "R_SPECIAL":
                    fil = "R_SPEC"
                save_fors2_calib(
                    output=down_path,
                    fil=fil,
                )
                self.calibration_table = table.QTable.read(down_path, format="ascii")
                self.calibration_table["zeropoint"] *= units.mag
                self.calibration_table["zeropoint_err"] *= units.mag
                self.calibration_table["colour_term"] *= units.mag
                self.calibration_table["colour_term_err"] *= units.mag
                self.calibration_table["extinction"] *= units.mag
                self.calibration_table["extinction_err"] *= units.mag
                self.write_calibration_table()
                self.calibration_table_last_updated = date.today()
            else:
                u.debug_print(1, "Filter calibrations already updated today; skipping.")

        else:
            u.debug_print(1, f"Cannot retrieve calibration table for {self.name}.")

        self.update_output_file()

        return self.calibration_table

    def get_nearest_calib_row(self, mjd: float):
        self.load_calibration_table()
        i, nrst = u.find_nearest(self.calibration_table["mjd_obs"], mjd)
        return self.calibration_table[i]

    def get_extinction(self, mjd: float):
        row = self.get_nearest_calib_row(mjd=mjd)
        return row["extinction"], row["extinction_err"]

    def get_nearest_calib_rows(self, mjd: float, n: int = 7):
        # self.retrieve_calibration_table()
        row_prime = self.get_nearest_calib_row(mjd=mjd)
        rows = [row_prime]
        mjd_low = mjd - 1
        mjd_high = mjd + 1
        while len(rows) < n:
            row = self.get_nearest_calib_row(mjd=mjd_high)
            if row not in rows:
                rows.append(row)
            row = self.get_nearest_calib_row(mjd=mjd_low)
            if row not in rows:
                rows.append(row)
            # print(mjd_low, mjd_high)
            mjd_low -= 1
            mjd_high += 1
        tbl = table.QTable(rows=rows, names=rows[0].colnames)
        tbl.sort("mjd_obs")
        return tbl

    def _output_dict(self):
        output_dict = super()._output_dict()
        output_dict.update(
            {
                "calibration_table_path": self.calibration_table_path,
                "calibration_table_last_updated": self.calibration_table_last_updated
            }
        )
        return output_dict

    def load_output_file(self):
        outputs = super().load_output_file()
        if type(outputs) is dict:
            if "calibration_table_path" in outputs:
                self.calibration_table_path = outputs["calibration_table_path"]
            if "calibration_table_last_updated" in outputs:
                self.calibration_table_last_updated = outputs["calibration_table_last_updated"]
        return outputs
