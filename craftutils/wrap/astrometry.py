import os


def build_astrometry_index(input_fits_catalog: str, unique_id: str, output_index: str = None,
                           scale_number: int = 0, sort_column: str = 'mag',
                           scan_through_catalog: bool = True, *flags, **params):
    sys_str = f"build-astrometry-index -i {input_fits_catalog} -I {unique_id}"
    if output_index is not None:
        sys_str += f" -o {output_index}"
    if scale_number is not None:
        sys_str += f" -P {scale_number}"
    if sort_column is not None:
        sys_str += f" -s {sort_column}"
    if scan_through_catalog:
        sys_str += "-E"
    for param in params:
        sys_str += f" -{param.upper()} {params[param]}"
    for flag in flags:
        sys_str += f" -{flag}"
    os.system(sys_str)
