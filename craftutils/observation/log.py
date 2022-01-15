import copy
import sys
from typing import Union

from astropy.time import Time

import craftutils.params as p


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

    def add_log(self, action: str, method=None, path: str = None):

        module_versions = {}
        for key in sys.modules:
            try:
                module_versions[key] = sys.modules[key].__version__
            except AttributeError:
                pass

        log_entry = {
            "craftutils_git_hash": p.get_project_git_hash(),
            "python_version": sys.version,
            "module_versions": module_versions,
            "action": action,
            "path": path
        }

        if method is not None:
            log_entry["method"] = method.__name__
        self.log[Time.now().strftime("%Y-%m-%d")] = log_entry
