import os
import shutil

from typing import List, Dict

from astropy.table import Table

import craftutils.params as p
import craftutils.fits_files as ff
import craftutils.utils as u

eso_bin_path = os.path.join(p.config["eso_install_dir"], "bin", "esorex")

eso_calib_path = os.path.join(p.config["eso_install_dir"], "calib")
fors2_calib_path = os.path.join(
    eso_calib_path,
    filter(
        lambda d: os.path.isdir(os.path.join(eso_calib_path, d)) and d.startswith("fors-"),
        os.listdir(eso_calib_path)
    ).__next__())


def write_sof(
        table_path: str,
        output_path: str = 'bias.sof',
        sof_type: str = 'fors_bias',
        chip: int = 1,
        cat_path: str = ""):
    """
    Requires that fits_table has already been run on the directory.
    For fors_zeropoint, if there are multiple STD-type files, it will use the first one listed in the table.
    :param table_path:
    :param output_path:
    :param sof_type:
    :param chip:
    :return:
    """
    sof_types = ['fors_bias', 'fors_img_sky_flat', 'fors_zeropoint']

    chip = ['CHIP1', 'CHIP2'][chip - 1]

    if cat_path != "" and cat_path[-1] != "/":
        cat_path = cat_path + "/"

    if sof_type not in sof_types:
        raise ValueError(sof_type + ' is not a recognised sof_type. Recognised types are:' + str(sof_types))

    if output_path[-4:] != ".sof":
        output_path = output_path + ".sof"

    if os.path.isfile(output_path):
        os.remove(output_path)

    files = Table.read(table_path, format="ascii.csv")
    files = files[files['chip'] == chip]

    # Bias
    if sof_type == 'fors_bias':
        bias_files = files[files['object'] == 'BIAS']['identifier']

        with open(output_path, 'a') as output:
            for file in bias_files:
                output.writelines(file + " BIAS\n")

    # Flats
    if sof_type == 'fors_img_sky_flat':
        flat_files = files[files['object'] == 'FLAT,SKY']['identifier']

        with open(output_path, 'a') as output:
            for file in flat_files:
                output.writelines(file + " SKY_FLAT_IMG\n")
            if chip == 'CHIP1':
                suffix = "up"
            else:
                suffix = "down"
            output.writelines("master_bias_" + suffix + ".fits MASTER_BIAS")

    # Zeropoint
    if sof_type == 'fors_zeropoint':
        if chip == 'CHIP1':
            suffix = "up"
            chip_id = "1453"
        else:
            suffix = "down"
            chip_id = "1456"

        std_image = files[files['object'] == 'STD']['identifier'][0]

        with open(output_path, 'a') as output:
            output.writelines(std_image + " STANDARD_IMG\n")
            output.writelines("master_bias_" + suffix + ".fits MASTER_BIAS\n")
            output.writelines("master_sky_flat_img_" + suffix + ".fits MASTER_SKY_FLAT_IMG\n")
            output.writelines(cat_path + "fors2_landolt_std_UBVRI.fits FLX_STD_IMG\n")
            output.writelines(cat_path + "fors2_" + chip_id + "_phot.fits PHOT_TABLE\n")


def sof(frames: Dict[str, list], output_path: str):
    output_lines = []
    for frame_type in frames:
        u.debug_print(0, output_lines)
        for frame in frames[frame_type]:
            output_lines.append(f"{frame} {frame_type}\n")

    u.mkdir_check_nested(output_path)
    print(f"Writing SOF file to {output_path}")
    with open(output_path, "w") as file:
        file.writelines(output_lines)

    return output_lines


def fors_bias(
        bias_frames: List[str], output_dir: str, output_filename: str = None,
        sof_name: str = "bias.sof"):
    sof_name = u.sanitise_file_ext(sof_name, ".sof")
    old_dir = os.getcwd()
    os.chdir(output_dir)
    sof_path = os.path.join(output_dir, sof_name)
    sof({"BIAS": bias_frames}, sof_path)
    u.system_command_verbose(f"{eso_bin_path} fors_bias {sof_path}")
    master_path = os.path.join(output_dir, "master_bias.fits")
    if output_filename is not None:
        final_path = os.path.join(output_dir, output_filename)
        shutil.move(master_path, final_path)
    else:
        final_path = master_path
    os.chdir(old_dir)
    return final_path


def fors_img_sky_flat(
        flat_frames: List[str],
        master_bias: str,
        output_dir: str,
        output_filename: str = None,
        sof_name: str = "flat.sof"
):
    sof_name = u.sanitise_file_ext(sof_name, ".sof")
    old_dir = os.getcwd()
    os.chdir(output_dir)
    sof_path = os.path.join(output_dir, sof_name)
    sof({
        "SKY_FLAT_IMG": flat_frames,
        "MASTER_BIAS": [master_bias],
    },
        sof_path)
    u.system_command_verbose(f"{eso_bin_path} fors_img_sky_flat {sof_path}")
    master_path = os.path.join(output_dir, "master_sky_flat_img.fits")
    if output_filename is not None:
        final_path = os.path.join(output_dir, output_filename)
        shutil.move(master_path, final_path)
    else:
        final_path = master_path
    os.chdir(old_dir)
    return final_path


fors_flux_std_imgs = ("fors2_stetson_2010Dec09.fits", "fors2_landolt_std_UBVRI.fits")


def select_phot_table(chip_num: int):
    if chip_num == 1:
        phot_table = "fors2_1453_phot.fits"
    elif chip_num == 2:
        phot_table = "fors2_1456_phot.fits"
    else:
        raise ValueError(f"FORS2 chip number must be 1 or 2, not {chip_num}")
    return phot_table


def fors_zeropoint(
        standard_img: str,
        master_bias: str,
        master_sky_flat_img: str,
        output_dir: str,
        output_filename: str = None,
        chip_num: int = 1,
        sof_name: str = "zeropoint.sof",
        flux_std_imgs: List[str] = fors_flux_std_imgs,
):
    sof_name = u.sanitise_file_ext(sof_name, ".sof")
    old_dir = os.getcwd()
    os.chdir(output_dir)
    sof_path = os.path.join(output_dir, sof_name)

    phot_table = select_phot_table(chip_num)

    sof({
        "STANDARD_IMG": [standard_img],
        "MASTER_BIAS": [master_bias],
        "MASTER_SKY_FLAT_IMG": [master_sky_flat_img],
        "FLX_STD_IMG": list(map(lambda f: os.path.join(fors2_calib_path, f), flux_std_imgs)),
        "PHOT_TABLE": [os.path.join(fors2_calib_path, phot_table)],
    },
        sof_path)
    u.system_command_verbose(f"{eso_bin_path} fors_zeropoint {sof_path}")

    master_path = os.path.join(output_dir, "aligned_phot.fits")
    std_path = os.path.join(output_dir, "standard_reduced_img.fits")
    if output_filename is not None:
        final_path = os.path.join(output_dir, output_filename)
        shutil.move(master_path, final_path)
    else:
        final_path = master_path
    os.chdir(old_dir)
    return final_path, std_path


def fors_photometry(
        aligned_phot: List[str],
        master_sky_flat_img: str,
        output_dir: str,
        output_filename: str = None,
        chip_num: int = 1,
        sof_name: str = "photometry.sof"
):
    sof_name = u.sanitise_file_ext(sof_name, ".sof")
    old_dir = os.getcwd()
    os.chdir(output_dir)
    sof_path = os.path.join(output_dir, sof_name)

    phot_table = select_phot_table(chip_num)

    sof({
        "MASTER_SKY_FLAT_IMG": [master_sky_flat_img],
        "ALIGNED_PHOT": aligned_phot,
        "PHOT_TABLE": [os.path.join(fors2_calib_path, phot_table)],
    },
        sof_path)

    u.system_command_verbose(f"{eso_bin_path} fors_photometry --fite=one {sof_path}")

    master_path = os.path.join(output_dir, "phot_coeff_table.fits")
    if output_filename is not None:
        final_path = os.path.join(output_dir, output_filename)
        shutil.move(master_path, final_path)
    else:
        final_path = master_path
    os.chdir(old_dir)
    return final_path
