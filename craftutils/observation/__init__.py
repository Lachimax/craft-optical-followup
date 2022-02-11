import os
from typing import Union

import astropy.table as table
import astropy.units as units

import craftutils.params as p
import craftutils.utils as u

config = p.config

u.mkdir_check_nested(config["table_dir"])


def _construct_column_lists(columns: dict, filters: list = None):
    dtypes = []
    un = []
    colnames = []
    default_data = []
    columns_revised = {}
    for colname in columns:

        if "{:s}" in colname:
            for fil in filters:
                colname_fil = colname.format(fil)
                colnames.append(colname_fil)
                columns_revised[colname_fil] = columns[colname]

        else:
            colnames.append(colname)
            columns_revised[colname] = columns[colname]

    for colname in colnames:
        val = columns_revised[colname]

        if isinstance(val, units.Unit) or isinstance(val, units.IrreducibleUnit):
            dtype = units.Quantity
            un.append(val)
        else:
            dtype = val
            un.append(None)
        if dtype is str:
            dtypes.append("U32")
        else:
            dtypes.append(dtype)
        default_data.append(dtype(0))
    return colnames, dtypes, un


master_imaging_table = None
master_imaging_table_path = os.path.join(config["table_dir"], "master_imaging_table.ecsv")
master_imaging_table_columns = {
    "field_name": str,
    "epoch_name": str,
    "filter_name": str,
    "instrument": str,
    "date_utc": str,
    "mjd": units.day,
    "filter_lambda_eff": units.Angstrom,
    "n_frames": int,
    "n_frames_included": int,
    "frame_exp_time": units.second,
    "total_exp_time": units.second,
    "total_exp_time_included": units.second,
    "psf_fwhm": units.arcsec,
    "program_id": str,
    "zeropoint": units.mag,
    "zeropoint_err": units.mag,
    "zeropoint_source": str,
    "last_processed": str,
    # "extinction_atm": units.mag,
    # "extinction_atm_err": units.mag,
}

master_objects_table = None
master_objects_path = os.path.join(config["table_dir"], "master_select_objects_table.ecsv")
master_objects_columns = {
    "jname": str,
    "field_name": str,
    "object_name": str,
    "ra": units.deg,
    "ra_err": units.deg,
    "dec": units.deg,
    "dec_err": units.deg,
    "a": units.arcsec,
    "a_err": units.arcsec,
    "b": units.arcsec,
    "b_err": units.arcsec,
    "theta": units.deg,
    "kron_radius": float,
    "mag_best_{:s}": units.mag,  # The magnitude from the deepest image in that band
    "mag_best_{:s}_err": units.mag,
    "mag_mean_{:s}": units.mag,
    "mag_mean_{:s}_err": units.mag,
    "ext_gal_{:s}": units.mag,
    "e_bv": units.mag,
    "epoch_best": str
}

furby_table = None
furby_table_columns = {
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


# master_photometry_table = None
# master_photometry_path = os.path.join(config["table_dir"], "photometry")
# u.mkdir_check(master_photometry_path)
# master_photometry_columns = [
#     "field_name",
#     "object_name",
#     ""
# ]

def load_master_table(
        tbl: Union[None, table.Table],
        tbl_columns: dict,
        tbl_path: str,
        force: bool = False,
        filters: list = None
):
    if force or tbl is None:
        colnames, dtypes, un = _construct_column_lists(columns=tbl_columns, filters=filters)
        if not os.path.isfile(tbl_path):
            tbl = table.QTable(data=[[0]] * len(colnames), names=colnames, units=un, dtype=dtypes)
            for i, colname in enumerate(colnames):
                if isinstance(dtypes[i], str):
                    tbl[colname][0] = "0" * 32
            tbl.write(tbl_path, format="ascii.ecsv")
        tbl = table.QTable.read(tbl_path, format="ascii.ecsv")
        for i, colname in enumerate(colnames):
            if colname not in tbl.colnames:
                dtype = dtypes[i]
                if isinstance(dtype, str):
                    dtype = str
                    val = dtype("0" * 32)
                else:
                    val = dtype(0)
                tbl.add_column([val] * len(tbl), name=colname)
                if un[i] is not None:
                    tbl[colname] *= un[i]

    return tbl


def load_master_imaging_table(force: bool = False):
    global master_imaging_table

    master_imaging_table = load_master_table(
        tbl=master_imaging_table,
        tbl_columns=master_imaging_table_columns,
        tbl_path=master_imaging_table_path,
        force=force
    )

    return master_imaging_table


def load_master_objects_table(force: bool = False):
    global master_objects_table

    master_objects_table = load_master_table(
        tbl=master_objects_table,
        tbl_columns=master_objects_columns,
        tbl_path=master_objects_path,
        force=force
    )

    return master_imaging_table


def write_master_imaging_table():
    if master_imaging_table is None:
        raise ValueError("master_table not loaded.")
    else:
        master_imaging_table.sort(["field_name", "epoch_name", "filter_name"])
        master_imaging_table.write(master_imaging_table_path, format="ascii.ecsv", overwrite=True)
        master_imaging_table.write(master_imaging_table_path.replace(".ecsv", ".csv"), format="ascii.csv",
                                   overwrite=True)


def _build_furby_table_path():
    if p.furby_path is not None:
        return os.path.join(p.furby_path, "craco_fu", "data", "craco_fu_db.csv")
    else:
        return None


def load_furby_table(force: bool = False):
    global furby_table

    if force or furby_table is None:
        path = _build_furby_table_path()
        if path is not None:
            _, dtypes, _ = _construct_column_lists(columns=master_imaging_table_columns)
            furby_table = table.Table.read(path, format="ascii.csv", dtype=dtypes)
        else:
            furby_table = None

    return furby_table


def write_furby_table():
    global furby_table

    path = _build_furby_table_path()
    if path is not None:
        furby_table.write(path, format="ascii.csv", overwrite=True)


def get_row(tbl: table.Table, colname: str, colval: str):
    is_name = tbl[colname] == colval
    if sum(is_name) == 0:
        return None, None
    else:
        return tbl[is_name][0], is_name.argmax()


def _get_row_two_conditions(
        tbl: table.Table,
        colname_1: str, colval_1: str,
        colname_2: str, colval_2: str
):
    good = (tbl[colname_1] == colval_1) * (tbl[colname_2] == colval_2)
    if sum(good) == 0:
        return None, None
    else:
        return tbl[good][0], good.argmax()


def get_row_furby(field_name: str):
    if furby_table is not None:
        return get_row(tbl=furby_table, colval=field_name, colname="Name")
    else:
        return None


def get_row_epoch(epoch_name: str, fil: str = None):
    if master_imaging_table is not None:
        row, index = _get_row_two_conditions(
            tbl=master_imaging_table,
            colname_1="epoch_name", colval_1=epoch_name,
            colname_2="filter_name", colval_2=fil)
        return row, index
    else:
        return None, None
