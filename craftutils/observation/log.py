import copy
import sys
from typing import Union, List, Dict

from astropy.time import Time

import craftutils.params as p
import craftutils.utils as u


class Log:

    def __init__(
            self,
            log_dict: dict = None
    ):
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
            self,
            action: str,
            method=None,
            method_args: dict = None,
            input_path: str = None,
            output_path: str = None,
            packages: List[str] = None,
            ancestor_logs: Union[Dict[str, Union[dict, 'Log']], List[Union[dict, 'Log']]] = None
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

        if method_args is not None:
            log_entry["method_args"] = method_args

        if ancestor_logs is not None:
            log_entry["ancestor_logs"] = {}
            if isinstance(ancestor_logs, list):
                new_dict = {}
                for i, log in enumerate(ancestor_logs):
                    new_dict[f"ancestor_{i}"] = ancestor_logs[i]
                ancestor_logs = new_dict

            for key in ancestor_logs:
                log = ancestor_logs[key]
                if isinstance(log, Log):
                    log = log.log
                log_entry["ancestor_logs"][key] = log

        self.log[Time.now().strftime("%Y-%m-%dT%H:%M:%S")] = log_entry
