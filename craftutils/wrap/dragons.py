import os
import shutil
from typing import Union
from typing import List

from astropy.table import Table

import craftutils.utils as u


def build_data_select_str(
        directory: str = None,
        file_glob: str = "*.fits",
        tags: list = None,
        expression: str = None,
        output: str = None):
    sys_str = f"dataselect "
    if directory is not None:
        sys_str += f"{directory}/"
    sys_str += file_glob
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
    return sys_str


def data_select(
        redux_dir: str,
        directory: str,
        file_glob: str = "*.fits",
        tags: list = None,
        expression: str = None,
        output: str = None):
    # Switch working directory to reduction directory.
    pwd = os.getcwd()
    os.chdir(redux_dir)
    sys_str = build_data_select_str(
        directory=directory,
        file_glob=file_glob,
        tags=tags,
        expression=expression,
        output=output
    )
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


def caldb_add():
    pass


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


def disco(redux_dir: str,
          tags: list = None,
          expression: str = None,
          output: str = None,
          file_glob: str = "*_skySubtracted.fits",
          refcat: str = None,
          refcat_format: str = "ascii.csv",
          refcat_ra: str = None,
          refcat_dec: str = None,
          ignore_objcat: bool = False
          ):
    # Switch working directory to reduction directory.
    pwd = os.getcwd()
    os.chdir(redux_dir)

    ds_str = build_data_select_str(
        tags=tags,
        expression=expression,
        file_glob=file_glob
    )

    sys_str = f"disco `{ds_str}`"
    if output is not None:
        if os.path.isfile(output):
            os.remove(output)
        sys_str += f" -o {output}"

    if refcat is not None:
        if refcat_ra is not None or refcat_dec is not None:
            filename = os.path.split(refcat)[-1]
            refcat_tbl = Table.read(os.path.join(refcat, filename), format=refcat_format)
            if refcat_ra is not None:
                refcat_tbl["RA"] = refcat_tbl[refcat_ra]
            if refcat_dec is not None:
                refcat_tbl["DEC"] = refcat_tbl[refcat_dec]
            refcat_tbl.write(
                os.path.join(redux_dir, filename),
                overwrite=True)
            refcat = filename
        sys_str += f" --refcat {refcat}"
    if refcat_format is not None:
        sys_str += f" --refcat_format {refcat_format}"
    if ignore_objcat:
        sys_str += " --ignore_objcat"

    print()
    print(sys_str)
    print()
    os.system(sys_str)

    os.chdir(pwd)
