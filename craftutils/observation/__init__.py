import os
from typing import Union, Iterable

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

        dtype, unit = split_dtype(val)
        dtypes.append(dtype)
        un.append(unit)

    return colnames, dtypes, un


def split_dtype(val):
    if isinstance(val, units.Unit) or isinstance(val, units.IrreducibleUnit):
        dtype = units.Quantity
        unit = val
    else:
        dtype = val
        unit = None
    if dtype is str:
        dtype = "U32"

    return dtype, unit


#
#
# def add_columns_by_fil(tbl: table.QTable, coldict: dict, fil: str):
#     for col in coldict:
#         if "{:s}" in col:
#             dtype, unit = split_dtype(coldict[col])
#             col = col.format(fil)
#             add_column(tbl=tbl, colname=col, dtype=dtype, unit=unit)
#
#
# def add_columns_to_master_objects(fil: str):
#     print(f"Adding columns for {fil} to master objects table")
#     load_master_objects_table()
#     global master_objects_table
#     add_columns_by_fil(tbl=master_objects_table, coldict=master_objects_columns, fil=fil)
#     write_master_objects_table()
#
#     load_master_all_objects_table()
#     global master_objects_all_table
#     add_columns_by_fil(tbl=master_objects_all_table, coldict=master_objects_columns, fil=fil)
#     write_master_all_objects_table()


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

master_imaging = None
master_imaging_path = os.path.join(config["table_dir"], "master_imaging_table.yaml")
master_imaging_columns = {
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
    "depth": units.mag
    # "extinction_atm": units.mag,
    # "extinction_atm_err": units.mag,
}

master_objects = None
master_objects_path = os.path.join(config["table_dir"], "master_select_objects_table.yaml")

master_objects_all = None
master_objects_all_path = os.path.join(config["table_dir"], "master_all_objects_table.yaml")
master_objects_columns = {
    "jname": str,
    "field_name": str,
    "object_name": str,
    "ra": units.deg,
    "ra_err": units.deg,
    "dec": units.deg,
    "dec_err": units.deg,
    "epoch_position": str,
    "epoch_position_date": str,
    "a": units.arcsec,
    "a_err": units.arcsec,
    "b": units.arcsec,
    "b_err": units.arcsec,
    "theta": units.deg,
    "epoch_ellipse": str,
    "epoch_ellipse_date": str,
    "theta_err": units.deg,
    "kron_radius": float,
    # "mag_best_{:s}": units.mag,  # The magnitude from the deepest image in that band
    # "mag_best_{:s}_err": units.mag,
    # "snr_best_{:s}": float,
    # "mag_mean_{:s}": units.mag,
    # "mag_mean_{:s}_err": units.mag,
    # "epoch_best_{:s}": str,
    # "epoch_best_date_{:s}": str,
    # "ext_gal_{:s}": units.mag,
    # "e_b-v": units.mag,
    # "class_star": float,
    # "mag_psf_best_{:s}": units.mag,
    # "snr_psf_best_{:s}": float,
    # "mag_psf_best_{:s}_err": units.mag,
    # "mag_psf_mean_{:s}": units.mag,
    # "mag_psf_mean_{:s}_err": units.mag,
}


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
            _, dtypes, _ = _construct_column_lists(columns=furby_table_columns)
            furby_table = table.Table.read(path, format="ascii.csv", dtype=dtypes)
        else:
            furby_table = None

    return furby_table


def load_master_table(
        tbl: Union[None, table.Table],
        tbl_path: str,
        force: bool = False,
):
    if force or tbl is None:
        if not os.path.isfile(tbl_path):
            tbl = {}
            p.save_params(tbl_path, tbl)
        else:
            tbl = p.load_params(tbl_path)
    return tbl


def load_master_imaging_table(force: bool = False):
    global master_imaging

    master_imaging = load_master_table(
        tbl=master_imaging,
        tbl_path=master_imaging_path,
        force=force
    )

    return master_imaging


def load_master_objects_table(force: bool = False):
    global master_objects

    master_objects = load_master_table(
        tbl=master_objects,
        tbl_path=master_objects_path,
        force=force
    )

    return master_objects


def load_master_all_objects_table(force: bool = False):
    global master_objects_all

    master_objects_all = load_master_table(
        tbl=master_objects_all,
        tbl_path=master_objects_all_path,
        force=force
    )

    return master_objects_all


def add_entry(
        tbl: dict,
        key: str,
        entry: dict,
        required: Iterable = ()
):
    for item in required:
        if item not in entry:
            raise ValueError(f"Required key {required} not found in entry.")

    tbl[key] = entry


def add_epoch(
        epoch_name: str,
        fil: str,
        entry: dict
):
    load_master_imaging_table()
    key = f"{epoch_name}_{fil}"
    add_entry(
        tbl=master_imaging,
        key=key,
        entry=entry,
        required=master_imaging_columns
    )


def add_photometry(
        tbl: dict,
        object_name: str,
        entry: dict
):
    if object_name in tbl:
        tbl[object_name].update(entry)
    else:
        add_entry(
            tbl=tbl,
            key=object_name,
            entry=entry,
            required=master_objects_columns
        )


def write_master_table(
        tbl_path: str,
        tbl: dict,
        sort_by: Union[str, list] = None,
):
    if tbl is None:
        tbl_name = os.path.split(tbl_path)[-1][:-5]
        raise ValueError(f"{tbl_name} not loaded.")
    p.save_params(tbl_path, tbl)
    tbl_list = list(map(lambda e: tbl[e], tbl))
    tbl_astropy = table.QTable(tbl_list)
    tbl_astropy.sort(sort_by)
    tbl_astropy.write(
        tbl_path.replace(".yaml", ".ecsv"),
        format="ascii.ecsv",
        overwrite=True
    )
    tbl_astropy.write(
        tbl_path.replace(".yaml", ".csv"),
        format="ascii.csv",
        overwrite=True
    )


def write_master_imaging_table():
    write_master_table(
        tbl_path=master_imaging_path,
        tbl=master_imaging,
        sort_by=["field_name", "epoch_name", "filter_name"]
    )


def write_master_all_objects_table():
    write_master_table(
        tbl_path=master_objects_all_path,
        tbl=master_objects_all,
        sort_by=["field_name", "jname"]
    )


def write_master_objects_table():
    write_master_table(
        tbl_path=master_objects_path,
        tbl=master_objects,
        sort_by=["field_name", "jname"]
    )


def get_row(tbl: table.Table, colname: str, colval: str):
    is_name = tbl[colname] == colval
    if sum(is_name) == 0:
        return None, None
    else:
        return tbl[is_name][0], is_name.argmax()


def get_entry(tbl: dict, key: str):
    return tbl[key]


def get_row_furby(field_name: str):
    if furby_table is not None:
        return get_row(tbl=furby_table, colval=field_name, colname="Name")
    else:
        return None


def get_epoch(epoch_name: str, fil: str):
    load_master_imaging_table()
    key = f"{epoch_name}_{fil}"
    if key in master_imaging:
        return master_imaging[key]
    else:
        return None
