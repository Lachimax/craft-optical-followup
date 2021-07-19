import os

import astropy.table as table

import craftutils.fits_files as ff
import craftutils.params as p

good_input = os.path.join(p.project_path, "tests", "files", "images", "divided_by_exp_time")


def test_fits_table_all():
    tbl = ff.fits_table_all(good_input)
    print(tbl.colnames)
    assert isinstance(tbl, table.Table)
    assert len(tbl) == 2
    assert isinstance(tbl["EXPTIME"][0], float)
