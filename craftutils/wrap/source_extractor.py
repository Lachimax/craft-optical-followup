import os

import astropy.units as units

import craftutils.params as p
import craftutils.utils as u
from craftutils.photometry import gain_unit


def source_extractor(image_path: str,
                     output_dir: str = None,
                     configuration_file: str = None,
                     parameters_file: str = None,
                     catalog_name: str = None,
                     copy_params: bool = True,
                     template_image_path: str = None,
                     **configs):
    """
    :param configs: Any source-extractor (sextractor) parameter, normally read via the config file but that can be
    overridden by passing to the shell command, can be given here.
    """

    if "gain" in configs:
        configs["gain"] = u.check_quantity(number=configs["gain"], unit=gain_unit).to(gain_unit).value

    old_dir = os.getcwd()
    if output_dir is None:
        output_dir = os.getcwd()
    else:
        os.chdir(output_dir)

    if copy_params:
        os.system(f"cp {os.path.join(p.path_to_config_psfex(), '*')} .")

    sys_str = "source-extractor "
    if template_image_path is not None:
        sys_str += f"{template_image_path},"
    sys_str += image_path + " "
    if configuration_file is not None:
        sys_str += f" -c {configuration_file}"
    if catalog_name is None:
        image_name = os.path.split(image_path)[-1]
        catalog_name = f"{image_name}.cat"
    sys_str += f" -CATALOG_NAME {catalog_name}"
    if parameters_file is not None:
        sys_str += f" -PARAMETERS_NAME {parameters_file}"
    for param in configs:
        sys_str += f" -{param.upper()} {configs[param]}"
    print()
    print(sys_str)
    print()
    os.system(sys_str)
    print()
    print(sys_str)
    print()
    catalog_path = os.path.join(output_dir, catalog_name)
    os.chdir(old_dir)
    return catalog_path
