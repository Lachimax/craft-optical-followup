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
    if type(cfg_split) is str:
        system_str += f" -c {cfg_split}"
    os.system(system_str)


def run_pypeit(pypeit_file: str, redux_path: str):
    os.system(f"run_pypeit {pypeit_file} -r {redux_path} -o")
