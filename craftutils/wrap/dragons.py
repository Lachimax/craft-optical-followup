import os
from typing import Union
from typing import List

from astropy.table import Table

import craftutils.utils as u


def data_select(redux_dir: str,
                raw_dir: str,
                tags: list = None,
                expression: str = None,
                output: str = None):
    # Switch working directory to reduction directory.
    pwd = os.getcwd()
    os.chdir(redux_dir)
    sys_str = f"dataselect {raw_dir}/*.fits"
    if tags is not None:
        sys_str += " --tags "
        for tag in tags:
            sys_str += tag
            sys_str += ","
        sys_str = sys_str[:-1]
    if expression is not None:
        sys_str += f" --expr '{expression}'"
    if output is not None:
        if os.path.isfile(output):
            os.remove(output)
        sys_str += f" -o {output}"
    print()
    print(sys_str)
    print("In:", os.getcwd())
    print()
    os.system(sys_str)

    data_list = None
    if output is not None:
        with open(output) as data_file:
            data_list = data_file.read()
    os.chdir(pwd)
    return data_list


def showd(input_filenames: Union[str, list],
          descriptors: Union[str, List[str]] = "filter_name,exposure_time,object",
          output: str = None,
          csv: bool = True,
          working_dir: str = None):
    # Switch working directory as specified.
    pwd = os.getcwd()
    if working_dir is not None:
        os.chdir(working_dir)
    if isinstance(descriptors, list):
        descriptors_str = ""
        for d in descriptors:
            descriptors_str += d
            descriptors_str += ","
        descriptors = descriptors_str[:-1]

    sys_str = f"showd -d {descriptors}"
    if csv:
        sys_str += f" --csv"
    if isinstance(input_filenames, str):
        sys_str += " " + input_filenames
    else:
        for line in input_filenames:
            sys_str += " " + line
    if output is not None:
        sys_str += f" >> {output}"

    if os.path.isfile(output):
        os.remove(output)

    print()
    print(sys_str)
    print()
    os.system(sys_str)

    tbl = None
    if output is not None:
        tbl = Table.read(output, format="csv")
        os.chdir(pwd)
    return tbl


def caldb_init(redux_dir: str):
    cfg_text = f"""[calibs]
    standalone = True
    database_dir = {redux_dir}
    """

    cfg_path = os.path.join(os.path.expanduser("~"), ".geminidr", "rsys.cfg")

    with open(cfg_path, "w") as cfg:
        cfg.write(cfg_text)

    print()
    sys_str = "caldb init -w"
    print(sys_str)
    print()
    os.system(sys_str)


def reduce(data_list_path: str, redux_dir: str):
    # Switch working directory to reduction directory.
    pwd = os.getcwd()
    os.chdir(redux_dir)
    sys_str = f"reduce @{data_list_path}"
    print()
    print(sys_str)
    print()
    os.system(sys_str)
    os.chdir(pwd)
