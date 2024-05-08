import os
from typing import Union, List

from astropy.time import Time
import astropy.units as units

import craftutils.params as p
import craftutils.utils as u
import craftutils.observation.log as log
from craftutils.observation.generic import Generic


class Pipeline(Generic):
    stage_output_dirs = True
    def __init__(
            self,
            name: str = None,
            param_path: str = None,
            data_path: str = None,
            do_stages: Union[list, str] = None,
            **kwargs
    ):
        self.param_path = param_path
        self.name = name

        self.quiet = False
        if "quiet" in kwargs:
            self.quiet = kwargs["quiet"]

        self.data_path = None
        self.data_path_relative = None
        if data_path is not None:
            self.data_path = os.path.join(p.data_dir, data_path)
            self.data_path_relative = data_path
        if data_path is not None:
            u.mkdir_check_nested(self.data_path)

        self.do = do_stages

        # Written attributes
        self.output_file = None  # This will be set during the load_output_file call
        self.stages_complete = {}
        self.log = log.Log()

        # Data reduction paths
        self.paths = {}

        self.do_kwargs = {}
        u.debug_print(2, "do" in kwargs)
        if "do" in kwargs:
            self.do_kwargs = kwargs["do"]

        self.param_file = kwargs

        self.stage_params: dict = {}

    def add_log(
            self,
            action: str,
            method=None,
            method_args=None,
            path: str = None,
            packages: List[str] = None,
    ):
        self.log.add_log(
            action=action,
            method=method,
            method_args=method_args,
            output_path=path,
            packages=packages
        )

    def set_path(self, key: str, value: str):
        self.paths[key] = value

    def update_output_file(self):
        p.update_output_file(self)

    @classmethod
    def stages(cls):
        return {}

    @classmethod
    def enumerate_stages(cls, show: bool = True):
        stages = list(enumerate(cls.stages()))
        if show:
            for i, stage in stages:
                print(f"{i}. {stage}")
        return stages

    def query_stage(self, message: str, stage_name: str, n: float, stage_kwargs: dict = None):
        """Helper method for asking the user if we need to do this stage of processing.
        If self.do is True, skips the query and returns True.

        :param message: Message to display.
        :param stage_name: code-friendly name of stage, eg "coadd" or "initial_setup"
        :param n: Stage number
        :return:
        """
        # Check if n is an integer, and if so cast to int.
        if n == int(n):
            n = int(n)
        if self.do is not None:
            if stage_name in self.do:
                return True
        else:
            message = f"{self.name} {n}. {message}"
            done = self.check_done(stage=stage_name)
            u.debug_print(2, "Epoch.query_stage(): done ==", done)
            if done is not None:
                time_since = (Time.now() - done).sec * units.second
                time_since = u.relevant_timescale(time_since)
                message += f" (last performed at {done.isot}, {time_since.round(1)} ago)"
            if stage_kwargs:
                message += f"\nSpecified config keywords:\n{stage_kwargs}"
            return u.select_yn_exit(message=message)

    def check_done(self, stage: str):
        u.debug_print(2, "Epoch.check_done(): stage ==", stage)
        u.debug_print(2, f"Epoch.check_done(): {self}.stages_complete ==", self.stages_complete)
        if stage not in self.stages():
            raise ValueError(f"{stage} is not a valid stage for this Epoch.")
        if stage in self.stages_complete:
            if isinstance(self.stages_complete[stage], Time):
                return self.stages_complete[stage]
            elif self.stages_complete[stage]["status"] == "skipped":
                return None
            else:
                return self.stages_complete[stage]["time"]
        else:
            return None

    def pipeline(
            self,
            no_query: bool = False,
            **kwargs
    ):
        """Performs the pipeline methods given in stages() for this instance.

        :param no_query: If True, skips the query stage and performs all stages (unless "do" was provided on __init__),
            in which case it will perform only those stages without query no matter what no_query is.
        :return:
        """
        skip_cats = False
        if "skip_cats" in kwargs:
            skip_cats = kwargs["skip_cats"]
        self._pipeline_init(skip_cats=skip_cats)
        # u.debug_print(2, "Epoch.pipeline(): kwargs ==", kwargs)

        # Loop through stages list specified in self.stages()
        stages = self.stages()
        u.debug_print(1, f"Epoch.pipeline(): type(self) ==", type(self))
        u.debug_print(2, f"Epoch.pipeline(): stages ==", stages)
        last_complete = None
        for n, (name, stage) in enumerate(stages.items()):
            message = stage["message"]
            # If default is present, then it defines whether the stage should be performed by default. If True, it
            # must be switched off by the do_key to skip the step; if False, then do_key must be set to True to perform
            # the step.
            if "default" in stage:
                do_this = stage["default"]
            else:
                do_this = True

            # Check if name is in "do" dict. If it is, defer to that setting; if not, defer to default.
            if name in self.do_kwargs:
                do_this = self.do_kwargs[name]

            u.debug_print(2, f"Epoch.pipeline(): {self}.stages_complete ==", self.stages_complete)

            if name in self.param_file:
                stage_kwargs = self.param_file[name]
            else:
                stage_kwargs = {}

            # Check if we should do this stage
            if do_this and (no_query or self.query_stage(
                    message=message,
                    n=n,
                    stage_name=name,
                    stage_kwargs=stage_kwargs
            )):
                self.stage_params[name] = stage_kwargs.copy()
                if not self.quiet:
                    print(f"Performing processing step {n}: {name} with keywords:\n", stage_kwargs)
                # Construct path; if dir_name is None then the step is pathless.
                dir_name = f"{n}-{name}"

                output_dir = os.path.join(self.data_path, dir_name)
                output_dir_backup = output_dir + "_backup"
                if self.stage_output_dirs:
                    u.rmtree_check(output_dir_backup)
                    u.move_check(output_dir, output_dir_backup)
                    u.mkdir_check_nested(output_dir, remove_last=False)
                self.set_path(name, output_dir)

                if stage["method"](self, output_dir=output_dir, **stage_kwargs) is not False:
                    self.stages_complete[name] = {
                        "status": "complete",
                        "time": Time.now(),
                        "kwargs": stage_kwargs
                    }

                    if "log_message" in stage and stage["log_message"] is not None:
                        log_message = stage["log_message"]
                    else:
                        log_message = f"Performed processing step {dir_name}."
                    self.add_log(log_message, method=stage["method"], path=output_dir, method_args=stage_kwargs)

                    u.rmtree_check(output_dir_backup)

                    last_complete = dir_name

                self.update_output_file()

            elif not do_this:
                self.stages_complete[name] = {
                    "status": "skipped",
                    "time": Time.now(),
                    "kwargs": {}
                }

        return last_complete

    def _pipeline_init(self, skip_cats: bool = False):
        if self.data_path is not None:
            u.debug_print(2, f"{self}._pipeline_init(): self.data_path ==", self.data_path)
            u.mkdir_check_nested(self.data_path)
        else:
            raise ValueError(f"data_path has not been set for {self}")

        self.do = _check_do_list(self.do, stages=list(self.stages().keys()))
        if not self.quiet and self.do:
            print(f"Doing stages {self.do}")

    def _output_dict(self):
        return {
            "paths": self.paths,
            "stages": self.stages_complete,
            "stage_params": self.stage_params
        }


def _check_do_list(
        do: Union[list, str],
        stages
):
    if isinstance(do, str):
        try:
            do = [int(do)]
        except ValueError:
            if " " in do:
                char = " "
            elif "," in do:
                char = ","
            else:
                raise ValueError("do string is not correctly formatted.")
            do = list(map(int, do.split(char)))

    if isinstance(do, list):
        do_nu = []
        for n in do:
            if isinstance(n, int):
                do_nu.append(stages[n])
            elif isinstance(n, str):
                if n in stages:
                    do_nu.append(n)
        do = do_nu

    return do


