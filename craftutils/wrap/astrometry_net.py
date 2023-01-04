import os

from distutils.spawn import find_executable

import astropy.units as units
from astropy.coordinates import SkyCoord

from typing import Union

from craftutils.utils import system_command, debug_print, check_quantity


def add_index_directory(path: str):
    bin_path = os.path.dirname(find_executable("astrometry-engine"))
    cfg_path = os.path.abspath(os.path.join(bin_path, "..", "etc", "astrometry.cfg"))
    line = f"add_path {path}\n"
    with open(cfg_path, 'r') as cfg:
        cfg_file = cfg.readlines()
    if line not in cfg_file:
        path_lines = list(filter(lambda l: l.startswith("add_path"), cfg_file))
        if path_lines:
            path_start = cfg_file.index(path_lines[0])
        else:
            path_start = 29
        cfg_file.insert(path_start, line)
        with open(cfg_path, 'w') as cfg:
            cfg.writelines(cfg_file)


def build_astrometry_index(
        input_fits_catalog: str,
        unique_id: int,
        output_index: str = None,
        scale_number: int = 0,
        sort_column: str = 'mag',
        scan_through_catalog: bool = True,
        *flags,
        **params
):
    print(input_fits_catalog)
    params["i"] = input_fits_catalog
    params["I"] = unique_id
    if output_index is not None:
        params["o"] = output_index
    if scale_number is not None:
        params["P"] = scale_number
    if sort_column is not None:
        params["s"] = sort_column

    flags = list(flags)
    if scan_through_catalog:
        flags.append("E")

    system_command("build-astrometry-index", None, False, True, *flags, **params)


def solve_field(
        image_files: Union[str, list],
        base_filename: str = "astrometry",
        overwrite: bool = True,
        tweak: bool = True,
        search_radius: units.Quantity = 1 * units.degree,
        centre: SkyCoord = None,
        guess_scale: bool = True,
        time_limit: units.Quantity = None,
        verify: bool = True,
        odds_to_tune_up: float = 1e6,
        odds_to_solve: float = 1e9,
        am_flags: list = None,
        am_params: dict = None,
):
    """
    Returns True if successful (by checking whether the corrected file is generated); False if not.
    :param image_files:
    :param base_filename:
    :param overwrite:
    :param flags:
    :param params:
    :return:
    """
    if am_params is None:
        am_params = {}
    if am_flags is None:
        am_flags = []
    am_params["o"] = base_filename
    am_params["odds-to-tune-up"] = odds_to_tune_up
    am_params["odds-to-solve"] = odds_to_solve
    if time_limit is not None:
        am_params["l"] = check_quantity(time_limit, units.second).value
    if search_radius is not None:
        am_params["radius"] = search_radius.to(units.deg).value
    if centre is not None:
        am_params["ra"] = centre.ra.to(units.deg).value
        am_params["dec"] = centre.dec.to(units.deg).value
    debug_print(1, "solve_field(): tweak ==", tweak)

    flags = list(am_flags)
    if overwrite:
        flags.append("O")
    if guess_scale:
        flags.append("g")
    if not tweak:
        flags.append("T")
    if not verify:
        flags.append("y")

    system_command("solve-field", image_files, False, True, False, *flags, **am_params)
    if isinstance(image_files, list):
        image_path = image_files[0]
    else:
        image_path = image_files
    check_dir = os.path.split(image_path)[0]
    check_path = os.path.join(check_dir, f"{base_filename}.new")
    print(f"Checking for result file at {check_path}...")
    return os.path.isfile(check_path)
