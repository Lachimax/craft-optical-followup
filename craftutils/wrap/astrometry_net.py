import os

from typing import Union

from craftutils.utils import system_command


def build_astrometry_index(input_fits_catalog: str, unique_id: str, output_index: str = None,
                           scale_number: int = 0, sort_column: str = 'mag',
                           scan_through_catalog: bool = True, *flags, **params):
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

    system_command("build-astrometry-index", *flags, **params)


def solve_field(image_files: Union[str, list], base_filename: str = "astrometry",
                overwrite: bool = True, *flags, **params):

    params["o"] = base_filename

    flags = list(flags)
    if overwrite:
        flags.append("O")
    system_command("solve-field", image_files, *flags, **params)
