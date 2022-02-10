import os

import astropy.table as table
import astropy.units as units

import craftutils.params as p
import craftutils.utils as u

config = p.config

u.mkdir_check_nested(config["table_dir"])


def _construct_column_lists(columns: dict):
    dtypes = []
    un = []
    colnames = []
    default_data = []
    for colname in columns:
        val = columns[colname]
        colnames.append(colname)
        if isinstance(val, units.Unit) or isinstance(val, units.IrreducibleUnit):
            dtype = units.Quantity
            un.append(val)
        else:
            dtype = val

            un.append(None)
        if dtype is str:
            dtypes.append("U64")
        else:
            dtypes.append(dtype)
        default_data.append(dtype(0))
    return colnames, dtypes, un


master_imaging_table = None
master_imaging_table_path = os.path.join(config["table_dir"], "master_imaging_table.ecsv")
master_imaging_table_columns = {
    "field_name": str,
    "epoch_name": str,
    "date_utc": str,
    "mjd": units.day,
    "instrument": str,
    "filter_name": str,
    "filter_lambda_eff": units.micron,
    "n_frames": int,
    "n_frames_included": int,
    "frame_exp_time": units.second,
    "total_exp_time": units.second,
    "total_exp_time_included": units.second,
    "psf_fwhm": units.arcsec,
    "program_id": str,
    "zeropoint": units.mag,
    "zeropoint_err": units.mag,
    # "extinction_atm": units.mag,
    # "extinction_atm_err": units.mag,
}

objects_table = None
master_objects_path = os.path.join(config["table_dir"], "master_select_objects_table.ecsv")
master_objects_columns = {
    "field_name": None,
    "object_name": None,
    "jname": str,
    "ra": units.deg,
    "ra_err": units.deg,
    "dec": units.deg,
    "dec_err": units.deg,
    "a": units.deg,
    "a_err": units.deg,
    "b": units.deg,
    "b_err": units.deg,
    "kron_radius": None,
    "mag_best_{:s}": units.mag,  # The magnitude from the deepest image in that band
    "mag_best_{:s}_err": units.mag,
    "mag_mean_{:s}": units.mag,
    "mag_mean_{:s}_err": units.mag,
    "ext_gal_{:s}": units.mag,
    "e_bv": units.mag
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
furby_table_path = os.path.join(p.furby_path, "craco_fu", "data", "craco_fu_db.csv")


# photometry_table = None
# master_photometry_path = os.path.join(config["table_dir"], "photometry")
# u.mkdir_check(master_photometry_path)
# master_photometry_columns = [
#     "field_name",
#     "object_name",
#     #""
# ]


def load_master_imaging_table(force: bool = False):
    global master_imaging_table
    if force or master_imaging_table is None:
        if os.path.isfile(master_imaging_table_path):
            master_imaging_table = table.QTable.read(master_imaging_table_path, format="ascii.ecsv")
        else:
            colnames, dtypes, un = _construct_column_lists(columns=master_imaging_table_columns)
            master_imaging_table = table.QTable(data=[[0]] * len(colnames), names=colnames, units=un, dtype=dtypes)
            # master_epoch_table.remove_row(0)

    return master_imaging_table


def write_master_epoch_table():
    if master_imaging_table is None:
        raise ValueError("master_table not loaded.")
    else:
        master_imaging_table.sort(["field_name", "epoch_name"])
        print(master_imaging_table)
        master_imaging_table.write(master_imaging_table_path, format="ascii.ecsv")
        master_imaging_table.write(master_imaging_table_path.replace(".ecsv", ".csv"), format="ascii.csv")


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
        furby_table.write(path, format="ascii.csv")


def get_row(tbl: table.Table, colval: str, colname: str):
    is_name = tbl[colname] == colval
    if sum(is_name) == 0:
        return None, None
    else:
        return tbl[is_name][0], is_name.argmax()


def get_row_furby(field_name: str):
    if furby_table is not None:
        return get_row(tbl=furby_table, colval=field_name, colname="Name")
    else:
        return None


def get_row_epoch(epoch_name: str):
    if master_imaging_table is not None:
        return get_row(tbl=master_imaging_table, colval=epoch_name, colname="epoch_name")
    else:
        return None
