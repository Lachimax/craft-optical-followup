# Code by Lachlan Marnoch 2021

import os

import numpy as np

import astropy.io.fits as fits

import craftutils.utils as u
from craftutils.fits_files import fits_table_all
from craftutils.photometry import gain_median_combine, gain_mean_combine


def image_table(input_directory: str, output_path: str = "images.tbl"):
    """
    Executes the Montage task mImgtbl <input_directory> <output_path>
    :param input_directory:
    :param output_path:
    :return:
    """
    u.sanitise_file_ext(filename=output_path, ext=".tbl")
    return u.system_command("mImgtbl", [input_directory, output_path])


def make_header(table_path: str, output_path: str):
    """
    Executes Montage task mMakeHdr <table_path> <output_path>
    :param table_path:
    :param output_path:
    :return:
    """
    u.sanitise_file_ext(output_path, ".hdr")
    return u.system_command("mMakeHdr", [table_path, output_path])


def inject_header(file_path: str, input_directory: str,
                  extra_items: dict = None, keys: dict = None,
                  coadd_type: str = 'median', ext: int = 0):
    table = fits_table_all(input_directory, science_only=False)
    table.sort("ARCFILE")

    important_keys = {
        "airmass": "AIRMASS",
        "saturate": "SATURATE",
        "object": "OBJECT",
        "filter": "FILTER",
        "mjd-obs": "MJD-OBS",
        "date-obs": "DATE-OBS",
        "gain": "GAIN",
    }

    if keys is not None:
        important_keys.update(keys)

    for key in important_keys:
        important_keys[key] = important_keys[key]

    insert_dict = {
        f"AIRMASS": np.nanmean(np.float64(table[important_keys['airmass']])),
        f"SATURATE": np.nanmean(np.float64(table[important_keys['saturate']])),
        f"OBJECT": table[important_keys['object']][0],
        f"MJD-OBS": float(np.nanmin(np.float64(table[important_keys["mjd-obs"]]))),
        f"DATE-OBS": table[important_keys["date-obs"]][0]
    }

    if important_keys["filter"] in table:
        insert_dict["FILTER"] = table[important_keys["filter"]][0]

    if important_keys["gain"] in table:
        old_gain = table[important_keys["gain"]].mean()
        if coadd_type == "median":
            new_gain = gain_median_combine(old_gain=old_gain)
        elif coadd_type == "mean":
            new_gain = gain_mean_combine(old_gain=old_gain)
        elif coadd_type == "sum":
            new_gain = old_gain
        else:
            raise ValueError(f"Unrecognised coadd_type {coadd_type}")
        insert_dict["GAIN"] = new_gain

    colnames = table.colnames
    for i, frame in enumerate(table):
        for colname in colnames:
            header_key = f"HIERARCH FRAME{i} {colname.upper()}"
            val = frame[colname]
            print(val)
            insert = True
            if isinstance(val, str) and (not val.isascii() or "\n" in val) or np.ma.is_masked(val) or len(
                    f"{header_key}={val}") > 79:
                insert = False
            if insert:
                insert_dict[header_key] = val

    if extra_items is not None:
        insert_dict.update(extra_items)

    with fits.open(file_path, mode="update", output_verify="fix") as file:
        file[ext].header.update(insert_dict)


def dict_to_montage_header(dictionary: dict, header_lines: list = None):
    if header_lines is None:
        header_lines = []
    for key in dictionary:
        header_lines.insert(-1, f"{key} = {dictionary[key]}\n")
    return header_lines


def project_execute(input_directory: str, table_path: str, header_path: str, proj_dir: str, stats_table_path: str):
    """
    Executes mProjExec <table_path> <header_path> <proj_dir> <stats_table_path> -p <input_directory>
    :param input_directory:
    :param table_path:
    :param header_path:
    :param proj_dir:
    :param stats_table_path:
    :return:
    """
    table_path = u.sanitise_file_ext(filename=table_path, ext=".tbl")
    header_path = u.sanitise_file_ext(filename=header_path, ext=".hdr")
    stats_table_path = u.sanitise_file_ext(filename=stats_table_path, ext=".tbl")
    return u.system_command(command="mProjExec",
                            arguments=[table_path, header_path, proj_dir, stats_table_path],
                            p=input_directory)


def overlaps(table_path: str, difference_table_path: str):
    """
    Executes mOverlaps <table_path> <difference_table_path>
    :param table_path:
    :param difference_table_path:
    :return:
    """
    table_path = u.sanitise_file_ext(filename=table_path, ext=".tbl")
    difference_table_path = u.sanitise_file_ext(filename=difference_table_path, ext=".tbl")
    return u.system_command(command="mOverlaps",
                            arguments=[table_path, difference_table_path])


def difference_execute(input_directory: str, difference_table_path: str, header_path: str, diff_dir: str):
    """
    Executes mDiffExec <difference_table_path> <header_file> <diff_dir> -p <input_directory>
    :param input_directory:
    :return:
    """
    difference_table_path = u.sanitise_file_ext(filename=difference_table_path, ext=".tbl")
    header_path = u.sanitise_file_ext(filename=header_path, ext=".hdr")
    return u.system_command(command="mDiffExec",
                            arguments=[difference_table_path, header_path, diff_dir],
                            p=input_directory,
                            )


def fit_execute(difference_table_path: str, fit_table_path: str, diff_dir: str):
    """
    Executes mFitExec <difference_table_path> <fit_table_path> <diff_dir>
    :param difference_table_path:
    :param fit_table_path:
    :param diff_dir:
    :return:
    """
    difference_table_path = u.sanitise_file_ext(filename=difference_table_path, ext=".tbl")
    fit_table_path = u.sanitise_file_ext(filename=fit_table_path, ext=".tbl")
    return u.system_command(command="mFitExec",
                            arguments=[difference_table_path, fit_table_path, diff_dir])


def background_model(table_path: str, fit_table_path: str, correction_table_path: str):
    """
    Executes mBGModel <table_path> <fit_table_path> <correction_table_path>
    :param table_path:
    :param fit_table_path:
    :param correction_table_path:
    :return:
    """
    table_path = u.sanitise_file_ext(filename=table_path, ext=".tbl")
    fit_table_path = u.sanitise_file_ext(filename=fit_table_path, ext=".tbl")
    correction_table_path = u.sanitise_file_ext(filename=correction_table_path, ext=".tbl")
    return u.system_command(command="mBgModel",
                            arguments=[table_path, fit_table_path, correction_table_path])


def background_execute(input_directory: str, table_path: str, correction_table_path: str, corr_dir: str):
    """
    Executes mBgExec <table_path> <corrections_table_path> <corr_dir> -p <input_directory>
    :param input_directory:
    :param table_path:
    :param correction_table_path:
    :param corr_dir:
    :return:
    """
    table_path = u.sanitise_file_ext(filename=table_path, ext=".tbl")
    correction_table_path = u.sanitise_file_ext(filename=correction_table_path, ext=".tbl")
    return u.system_command(command="mBgExec",
                            arguments=[table_path, correction_table_path, corr_dir],
                            p=input_directory)


def add(input_directory: str, table_path: str,
        header_path: str, output_path: str = "coadded.fits", coadd_type: str = 'median'):
    """
    Executes mAdd  <table_path> <template_path> <output_path> -p <input_directory> -a <coadd_type>
    :param input_directory:
    :param table_path:
    :param header_path:
    :param output_path:
    :param coadd_type:
    :return:
    """
    table_path = u.sanitise_file_ext(filename=table_path, ext=".tbl")
    header_path = u.sanitise_file_ext(filename=header_path, ext=".hdr")
    output_path = u.sanitise_file_ext(filename=output_path, ext=".fits")
    return u.system_command(command="mAdd",
                            arguments=[table_path, header_path, output_path],
                            p=input_directory, a=coadd_type)


def standard_script(input_directory: str, output_directory: str, output_file_path: str = None):
    """
    Does a standard median coaddition of fits files in input_directory.
    Adapted from an example bash script found at http://montage.ipac.caltech.edu/docs/first_mosaic_tutorial.html
    :param input_directory:
    :param output_directory:
    :return:
    """
    u.mkdir_check(output_directory)
    print("Creating directories to hold processed images.")
    proj_dir = os.path.join(output_directory, "projdir")
    diff_dir = os.path.join(output_directory, "diffdir")
    corr_dir = os.path.join(output_directory, "corrdir")
    u.mkdir_check(proj_dir, diff_dir, corr_dir)

    print("Creating metadata tables of the input images.")
    table_path = os.path.join(output_directory, "images.tbl")
    image_table(input_directory=input_directory, output_path=table_path)

    print("Creating FITS headers describing the footprint of the mosaic.")
    header_path = os.path.join(output_directory, "template.hdr")
    make_header(table_path=table_path, output_path=header_path)

    print("Reprojecting input images.")
    stats_table_path = os.path.join(output_directory, "stats.tbl")
    project_execute(input_directory=input_directory, table_path=table_path, header_path=header_path,
                    proj_dir=proj_dir, stats_table_path=stats_table_path)

    print("Creating metadata table of the reprojected images.")
    reprojected_table_path = os.path.join(output_directory, "proj.tbl")
    image_table(input_directory=proj_dir, output_path=reprojected_table_path)

    print("Analyzing the overlaps between images.")
    difference_table_path = os.path.join(output_directory, "diffs.tbl")
    fit_table_path = os.path.join(output_directory, "fits.tbl")
    overlaps(table_path=reprojected_table_path, difference_table_path=difference_table_path)
    difference_execute(input_directory=proj_dir, difference_table_path=difference_table_path,
                       header_path=header_path, diff_dir=diff_dir)
    fit_execute(difference_table_path=difference_table_path,
                fit_table_path=fit_table_path, diff_dir=diff_dir)

    print("Performing background modeling and compute corrections for each image.")
    corrections_table_path = os.path.join(output_directory, "corrections.tbl")
    background_model(table_path=reprojected_table_path, fit_table_path=fit_table_path,
                     correction_table_path=corrections_table_path)

    print("Applying corrections to each image")
    background_execute(input_directory=proj_dir, table_path=reprojected_table_path,
                       correction_table_path=corrections_table_path,
                       corr_dir=corr_dir)

    print("Coadding the images to create mosaics with background corrections.")
    if output_file_path is None:
        output_file_path = os.path.join(output_directory, "coadded.fits")
    add(input_directory=corr_dir, coadd_type='median', table_path=reprojected_table_path,
        header_path=header_path, output_path=output_file_path)

    inject_header(file_path=output_file_path, input_directory=input_directory)

    return output_file_path