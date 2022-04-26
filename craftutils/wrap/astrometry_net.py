import os

import astropy.units as units
from astropy.coordinates import SkyCoord

from typing import Union

from craftutils.utils import system_command, debug_print, check_quantity


def build_astrometry_index(input_fits_catalog: str, unique_id: int, output_index: str = None,
                           scale_number: int = 0, sort_column: str = 'mag',
                           scan_through_catalog: bool = True, *flags, **params):
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
        *flags,
        **params):
    """
    Returns True if successful (by checking whether the corrected file is generated); False if not.
    :param image_files:
    :param base_filename:
    :param overwrite:
    :param flags:
    :param params:
    :return:
    """

    params["o"] = base_filename
    params["odds-to-tune-up"] = odds_to_tune_up
    params["odds-to-solve"] = odds_to_solve
    if time_limit is not None:
        params["l"] = check_quantity(time_limit, units.second).value
    if search_radius is not None:
        params["radius"] = search_radius.to(units.deg).value
    if centre is not None:
        params["ra"] = centre.ra.to(units.deg).value
        params["dec"] = centre.dec.to(units.deg).value
    debug_print(1, "solve_field(): tweak ==", tweak)

    flags = list(flags)
    if overwrite:
        flags.append("O")
    if guess_scale:
        flags.append("g")
    if not tweak:
        flags.append("T")
    if not verify:
        flags.append("y")


    system_command("solve-field", image_files, False, True, False, *flags, **params)
    if isinstance(image_files, list):
        image_path = image_files[0]
    else:
        image_path = image_files
    check_dir = os.path.split(image_path)[0]
    check_path = os.path.join(check_dir, f"{base_filename}.new")
    print(f"Checking for result file at {check_path}...")
    return os.path.isfile(check_path)
