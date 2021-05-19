import os


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
