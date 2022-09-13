import os

from typing import Union


def get_scope(lines: list, levels: list):
    this_dict = {}
    this_level = levels[0]
    for i, line in enumerate(lines):
        if levels[i] == this_level:
            scope_start = i + 1
            scope_end = i + 1
            while scope_end < len(levels) and levels[scope_end] > this_level:
                scope_end += 1
            if "=" in line:
                key, value = line.split("=")
                this_dict[key] = value
            else:
                this_dict[line] = get_scope(lines=lines[scope_start:scope_end], levels=levels[scope_start:scope_end])

    return this_dict


def get_pypeit_param_levels(lines: list):
    levels = []
    last_non_zero = 0
    for i, line in enumerate(lines):
        level = line.count("[")
        if level == 0:
            level = levels[last_non_zero] + 1
        else:
            last_non_zero = i
        levels.append(level)
        line = line.replace("\t", "").replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")
        if "#" in line:
            line = line.split("#")[0]
        lines[i] = line
    return lines, levels


def get_pypeit_user_params(file: Union[list, str]):
    if isinstance(file, str):
        with open(file) as f:
            file = f.readlines()

    p_start = file.index("# User-defined execution parameters\n") + 1
    p_end = p_start + 1
    while file[p_end] != "\n":
        p_end += 1

    lines, levels = get_pypeit_param_levels(lines=file[p_start:p_end])
    param_dict = get_scope(lines=lines, levels=levels)

    return param_dict


def pypeit_setup(root: str, output_path: str, spectrograph: str, cfg_split: str = None):
    """
    Wraps the pypeit_setup script.
    :param root:
    :param output_path:
    :param spectrograph:
    :param cfg_split:
    :return:
    """
    system_str = f"pypeit_setup -r {root} -d {output_path} -s {spectrograph}"
    print()
    print(system_str)
    print()
    if type(cfg_split) is str:
        system_str += f" -c {cfg_split}"
    os.system(system_str)
    print()
    print(system_str)


def run_pypeit(pypeit_file: str, redux_path: str, do_not_reuse_masters: bool = False):
    system_str = f"run_pypeit {pypeit_file} -r {redux_path} -o"
    if do_not_reuse_masters:
        system_str += " -m"
    print()
    print(system_str)
    print()
    os.system(system_str)
    print()
    print(system_str)


def pypeit_sensfunc(spec1dfile: str, outfile: str = None, run_dir: str = None):
    cwd = ""
    if run_dir is not None:
        cwd = os.getcwd()
        os.chdir(run_dir)
    system_str = f"pypeit_sensfunc {spec1dfile}"
    if outfile is not None:
        system_str += f" -o {outfile}"
    print()
    print(system_str)
    print()
    os.system(system_str)
    print()
    print(system_str)
    if run_dir is not None:
        os.chdir(cwd)


def pypeit_flux_setup(sci_path: str, run_dir: str = None):
    cwd = ""
    if run_dir is not None:
        cwd = os.getcwd()
        os.chdir(run_dir)
    os.system(f"pypeit_flux_setup {sci_path}")
    if run_dir is not None:
        os.chdir(cwd)


def pypeit_coadd_1dspec(coadd1d_file: str):
    sys_str = f"pypeit_coadd_1dspec {coadd1d_file}"
    print("\n" + sys_str + "\n")
    os.system(sys_str)
    print("\n" + sys_str + "\n")

