import os

from typing import Union

import craftutils.params as p
import craftutils.utils as u


# TODO: Parent class for all param loading etc, with children Instrument, Survey etc.

class Survey:
    def __init__(self, **kwargs):
        self.name = None
        if "name" in kwargs:
            self.name = kwargs["name"]
            self.formatted_name = self.name
        self.formatted_name = None
        if "formatted_name" in kwargs:
            self.formatted_name = kwargs["formatted_name"]
        self.raw_stage_path = None

        if "raw_stage_path" in kwargs and isinstance(kwargs["raw_stage_path"], str):
            self.raw_stage_path = u.make_absolute_path(p.data_dir, kwargs["raw_stage_path"])

        self.refined_stage_path = None
        if "refined_stage_path" in kwargs and isinstance(kwargs["refined_stage_path"], str):
            self.refined_stage_path = u.make_absolute_path(p.data_dir, kwargs["refined_stage_path"])

        self.program_ids = {}
        if "program_ids" in kwargs:
            self.program_ids = kwargs["program_ids"]

        self.extra_commands = []
        if "extra_commands" in kwargs:
            self.extra_commands = kwargs["extra_commands"]

        self.table_dir = None
        if "table_dir" in kwargs:
            self.table_dir = kwargs["table_dir"]

        self.epoch_table = None
        self.photometry_table = None

    def __str__(self):
        return str(self.name)

    def guess_param_dir(self):
        return self._build_param_dir(survey_name=self.name)

    @classmethod
    def _build_survey_dir(cls):
        path = os.path.join(p.param_dir, "surveys")
        u.mkdir_check(path)
        return path

    @classmethod
    def _build_param_dir(cls, survey_name: str):
        path = cls._build_survey_dir()
        path = os.path.join(path, survey_name)
        u.mkdir_check_nested(path, remove_last=False)
        return path

    @classmethod
    def _build_param_path(cls, survey_name: str):
        """
        Get default path to an instrument param .yaml file.
        :param instrument_name:
        :return:
        """
        path = cls._build_param_dir(survey_name=survey_name)
        return os.path.join(path, f"{survey_name}.yaml")

    @classmethod
    def _build_data_path(cls, survey_name: str):
        path = cls._build_survey_dir()
        u.mkdir_check(path)
        path = os.path.join(path, survey_name)
        u.mkdir_check(path)
        return path

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "formatted_name": None,
            "raw_stage_path": None,
            "refined_stage_path": None,
            "program_ids": {},  # Dict should have instrument names as keys and a list of strings for each value.
            "extra_commands": [] # Commands to be executed by shell at completion of last step (NOT YET IMPLEMENTED)
        }
        return default_params

    @classmethod
    def new_yaml(cls, survey_name: str = None, path: str = None, **kwargs):
        param_dict = cls.default_params()
        param_dict["name"] = survey_name
        param_dict.update(kwargs)
        if survey_name is not None:
            param_dict["data_path"] = cls._build_data_path(survey_name=survey_name)
        if path is not None:
            p.save_params(file=path, dictionary=param_dict)
        return param_dict

    @classmethod
    def new_param(cls, survey_name: str = None, **kwargs):
        path = cls._build_param_path(survey_name=survey_name)
        cls.new_yaml(survey_name=survey_name, path=path, **kwargs)

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):
        u.debug_print(1, "Survey.from_file(): param_file ==", param_file)
        name, param_file, param_dict = p.params_init(param_file)
        u.debug_print(1, "Survey.from_file(): name ==", name)
        u.debug_print(1, "Survey.from_file(): param_dict ==", param_dict)
        if param_dict is None:
            raise FileNotFoundError("Param file missing!")
        return cls(**param_dict)

    @classmethod
    def from_params(cls, survey_name: str):
        path = cls._build_param_path(survey_name=survey_name)
        u.debug_print(1, "Survey.from_params(): path ==", path)
        return cls.from_file(param_file=path)

    @classmethod
    def list_surveys(cls):
        param_path = cls._build_survey_dir()
        surveys = list(
            filter(lambda d: os.path.isdir(
                os.path.join(param_path, d)
            ) and os.path.isfile(
                os.path.join(param_path, d, f"{d}.yaml")),
                   os.listdir(param_path)
                   )
        )
        surveys.sort()

        return surveys


    @classmethod
    def new_param_from_input(cls):
        survey_name = u.user_input(
            message="Enter survey name:"
        )
        cls.new_param(survey_name)
        return survey_name

