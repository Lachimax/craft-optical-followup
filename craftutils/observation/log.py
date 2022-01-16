import copy
import sys
from typing import Union
from typing import List

from astropy.time import Time

import craftutils.params as p
import craftutils.utils as u

class Log:

    def __init__(self, log_dict: dict = None):
        self.log = {}
        if log_dict is not None:
            self.log = log_dict

    def __getitem__(self, item: str):
        return self.log[item]

    def __copy__(self):
        return Log(log_dict=self.log)

    def __deepcopy__(self, memodict={}):
        return Log(log_dict=copy.deepcopy(self.log, memodict))

    def copy(self):
        return self.__deepcopy__()

    def to_dict(self):
        return self.log

    def update(self, other: Union[dict, 'Log']):
        if isinstance(other, dict):
            self.log.update(other)
        elif isinstance(other, Log):
            self.log.update(other.log)

    def add_log(
            self, action: str, 
            method=None,
            input_path: str = None,
            output_path: str = None,
            packages: List[str] = None
    ):
        """

        :param action: String describing the action to be logged.
        :param method: Python method or function used to achieve this action. If provided
        :param output_path: Path to which new products were written.
        :param packages: A list of any system packages involved; if they can be reached in the terminal using this name,
            the version number will be recorded.
        :return:
        """

        module_versions = {}
        for key in sys.modules:
            try:
                module_versions[key] = sys.modules[key].__version__
            except AttributeError:
                pass

        log_entry = {
            "git_hash": p.get_project_git_hash(),
            "python_version": sys.version,
            "module_versions": module_versions,
            "action": action,
            "output_path": output_path,
            "input_path": input_path
        }

        if packages is not None:
            log_entry["package_versions"] = {}
            for package in packages:
                log_entry["package_versions"][package] = u.system_package_version(package)

        if method is not None:
            if isinstance(method, str):
                log_entry["method"] = method
            else:
                log_entry["method"] = method.__name__

        self.log[Time.now().strftime("%Y-%m-%d")] = log_entry
