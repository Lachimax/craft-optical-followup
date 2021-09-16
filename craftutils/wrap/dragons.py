import os
from typing import Union

import craftutils.utils as u


def data_select(redux_dir: str, raw_dir: str, expression: str = None, output: str = None):
    pwd = os.getcwd()
    os.chdir(redux_dir)
    sys_str = f"dataselect --expr '{expression}' {raw_dir}/*.fits"
    if output is not None:
        sys_str += f" >> {output}"
    print(sys_str)
    os.system(sys_str)
    os.chdir(pwd)

def showd(inp: str, d: Union[str, list] = None, output: str = None):
    pass


def config_init(redux_dir: str):
    cfg_text = f"""[calibs]
    standalone = True
    database_dir = {redux_dir}
    """

    cfg_path = os.path.join(os.path.expanduser("~"), ".geminidr", "rsys.cfg")

    with open(cfg_path, "w") as cfg:
        cfg.write(cfg_text)

    os.system("caldb init -w")
