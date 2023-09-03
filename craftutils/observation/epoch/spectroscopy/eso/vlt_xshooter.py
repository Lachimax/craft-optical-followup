import os
from typing import Union

import astropy.units as units
from astropy.time import Time

import craftutils.utils as u
import craftutils.wrap.pypeit as spec
import craftutils.observation.image as image

from .eso import ESOSpectroscopyEpoch


class XShooterSpectroscopyEpoch(ESOSpectroscopyEpoch):
    _instrument_pypeit = "vlt_xshooter"
    grisms = {'uvb': {"lambda_min": 300 * units.nm,
                      "lambda_max": 550 * units.nm},
              "vis": {"lambda_min": 550 * units.nm,
                      "lambda_max": 1000 * units.nm},
              "nir": {"lambda_min": 1000 * units.nm,
                      "lambda_max": 2500 * units.nm}}

    def __init__(self,
                 param_path: str = None,
                 name: str = None,
                 field: Union[str, 'craftutils.observation.field.Field'] = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 program_id: str = None,
                 ):

        super().__init__(
            param_path=param_path,
            name=name,
            field=field,
            data_path=data_path,
            instrument=instrument,
            date=date,
            program_id=program_id
        )

        self.frames_raw = {"uvb": [],
                           "vis": [],
                           "nir": []}
        self.frames_bias = {"uvb": [],
                            "vis": [],
                            "nir": []}
        self.frames_standard = {"uvb": [],
                                "vis": [],
                                "nir": []}
        self.frames_science = {"uvb": [],
                               "vis": [],
                               "nir": []}
        self.frames_dark = {"uvb": [],
                            "vis": [],
                            "nir": []}
        self._pypeit_file = {"uvb": None,
                             "vis": None,
                             "nir": None}
        self._pypeit_file_std = {"uvb": None,
                                 "vis": None,
                                 "nir": None}
        self._pypeit_sorted_file = {"uvb": None,
                                    "vis": None,
                                    "nir": None}
        self._pypeit_coadd1d_file = {"uvb": None,
                                     "vis": None,
                                     "nir": None}
        self._pypeit_user_param_start = {"uvb": None,
                                         "vis": None,
                                         "nir": None}
        self._pypeit_user_param_end = {"uvb": None,
                                       "vis": None,
                                       "nir": None}

        self.binning = {"uvb": None,
                        "vis": None,
                        "nir": None}
        self.binning_std = {"uvb": None,
                            "vis": None,
                            "nir": None}
        self.decker = {"uvb": None,
                       "vis": None,
                       "nir": None}
        self.decker_std = {"uvb": None,
                           "vis": None,
                           "nir": None}

        self._cfg_split_letters = {"uvb": None,
                                   "vis": None,
                                   "nir": None}

        self.load_output_file()
        self._current_arm = None

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)
        # self.proc_pypeit_coadd()

    def proc_pypeit_setup(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do PypeIt setup?", stage_name='2-pypeit_setup'):
            self._path_2_pypeit()
            setup_files = os.path.join(self.paths["pypeit_dir"], 'setup_files', '')
            self.paths["pypeit_setup_dir"] = setup_files
            os.system(f"rm {setup_files}*")
            for arm in self.grisms:
                self._current_arm = arm
                spec.pypeit_setup(root=self.paths['raw_dir'], output_path=self.paths['pypeit_dir'],
                                  spectrograph=f"{self._instrument_pypeit}_{arm}")
                # Read .sorted file
                self.read_pypeit_sorted_file()
                setup = self._cfg_split_letters[arm]
                spec.pypeit_setup(root=self.paths['raw_dir'], output_path=self.paths['pypeit_dir'],
                                  spectrograph=f"{self._instrument_pypeit}_{arm}", cfg_split=setup)
                # Retrieve text from .pypeit file
                self.read_pypeit_file(config=setup)
                # Add parameter to use dark frames for NIR reduction.
                if arm == "nir":
                    self.add_pypeit_user_param(param=["calibrations", "pixelflatframe", "process", "use_darkimage"],
                                               value="True")
                    self.add_pypeit_user_param(param=["calibrations", "illumflatframe", "process", "use_darkimage"],
                                               value="True")
                    self.add_pypeit_user_param(param=["calibrations", "traceframe", "process", "use_darkimage"],
                                               value="True")
                self.find_science_attributes()
                # For X-Shooter, we need to reduce the standards separately due to the habit of observing them with
                # different decker (who knows)
                self.find_std_attributes()
                # Remove incompatible binnings and frametypes
                if not self.quiet:
                    print(f"\nRemoving incompatible files for {arm} arm:")
                pypeit_file = self._get_pypeit_file()
                # pypeit_file_std = pypeit_file.copy()
                decker = self.get_decker()
                binning = self.get_binning()
                decker_std = self.get_decker_std()
                for raw_frame in self.frames_raw[arm]:
                    # Remove all frames with frame_type "None" from both science and standard lists.
                    if raw_frame.frame_type == "None":
                        pypeit_file.remove(raw_frame.pypeit_line)
                        # pypeit_file_std.remove(raw_frame.pypeit_line)
                    else:
                        # Remove files with incompatible binnings from science reduction list.
                        if raw_frame.binning != binning \
                                or raw_frame.decker not in ["Pin_row", decker, decker_std]:
                            pypeit_file.remove(raw_frame.pypeit_line)
                    # Special behaviour for NIR arm
                    if arm == "nir":
                        # For the NIR arm, PypeIt only works if you use the Science frames for the arc, tilt calib.
                        if raw_frame.frame_type in ["arc,tilt", "tilt,arc"]:
                            pypeit_file.remove(raw_frame.pypeit_line)
                            # pypeit_file_std.remove(raw_frame.pypeit_line)
                        elif raw_frame.frame_type == "science":
                            raw_frame.frame_type = "science,arc,tilt"
                            # Find original line in PypeIt file
                            to_replace = pypeit_file.index(raw_frame.pypeit_line)
                            # Rewrite pypeit line.
                            raw_frame.pypeit_line = raw_frame.pypeit_line.replace("science", "science,arc,tilt")
                            pypeit_file[to_replace] = raw_frame.pypeit_line
                        elif raw_frame.frame_type == "standard":
                            raw_frame.frame_type = "standard,arc,tilt"
                            # Find original line in PypeIt file
                            to_replace = pypeit_file.index(raw_frame.pypeit_line)
                            # Rewrite pypeit line.
                            raw_frame.pypeit_line = raw_frame.pypeit_line.replace("standard", "standard,arc,tilt")
                            pypeit_file[to_replace] = raw_frame.pypeit_line
                self._set_pypeit_file(pypeit_file)
                # self._set_pypeit_file_std(pypeit_file_std)
                self.write_pypeit_file_science()
                # std_path = os.path.join(self.paths["pypeit_dir"], self.get_path("pypeit_run_dir"), "Flux_Standards")
                # u.mkdir_check(std_path)
                # self.set_path("pypeit_dir_std", std_path)
                # self.set_path("pypeit_file_std",
                #              os.path.join(self.get_path("pypeit_run_dir"), f"vlt_xshooter_{arm}_std.pypeit"))
                # self.write_pypeit_file_std()
            self._current_arm = None
            self.stages_complete['2-pypeit_setup'] = Time.now()
            self.update_output_file()

    def proc_pypeit_run(self, no_query: bool = False, do_not_reuse_masters: bool = False, **kwargs):
        for i, arm in enumerate(self.grisms):
            # UVB not yet implemented in PypeIt, so we skip.
            if arm == "uvb":
                continue
            self._current_arm = arm
            if no_query or self.query_stage(f"Run PypeIt for {arm.upper()} arm?",
                                            stage_name=f'3.{i + 1}-pypeit_run_{arm}'):
                spec.run_pypeit(pypeit_file=self.get_path('pypeit_file'),
                                redux_path=self.get_path('pypeit_run_dir'),
                                do_not_reuse_masters=do_not_reuse_masters)
                self.stages_complete[f'3.{i + 1}-pypeit_run_{arm}'] = Time.now()
                self.update_output_file()
            # if arm != "nir" and self.query_stage(f"Run PypeIt on flux standards for {arm.upper()} arm?",
            #                                      stage=f'3.{i + 1}-pypeit_run_{arm}_std'):
            #     print(self.get_path('pypeit_file_std'))
            #     spec.run_pypeit(pypeit_file=self.get_path('pypeit_file_std'),
            #                     redux_path=self.get_path('pypeit_dir_std'),
            #                     do_not_reuse_masters=do_not_reuse_masters)
            #     self.stages_complete[f'3.{i + 1}-pypeit_run_{arm}'] = Time.now()
            #     self.update_output_file()
        self._current_arm = None

    def proc_pypeit_flux(self, no_query: bool = False, **kwargs):
        for i, arm in enumerate(self.grisms):
            # UVB not yet implemented in PypeIt, so we skip.
            if arm == "uvb":
                continue
            self._current_arm = arm
            if no_query or self.query_stage(f"Do PypeIt fluxing for {arm.upper()} arm?",
                                            stage_name=f'4.{i + 1}-pypeit_flux_calib_{arm}'):
                self._current_arm = arm
                self._pypeit_flux()
            self.stages_complete[f'4.{i + 1}-pypeit_flux_calib_{arm}'] = Time.now()
        self._current_arm = None
        self.update_output_file()

    def proc_pypeit_coadd(self, no_query: bool = False, **kwargs):
        for i, arm in enumerate(self.grisms):
            # UVB not yet implemented in PypeIt, so we skip.
            if arm == "uvb":
                continue
            self._current_arm = arm
            if no_query or self.query_stage(f"Do PypeIt coaddition for {arm.upper()} arm?",
                                            stage_name=f'5.{i + 1}-pypeit_coadd_{arm}'):
                run_dir = self.get_path("pypeit_run_dir")
                coadd_file_path = os.path.join(run_dir, f"{self._instrument_pypeit}_{arm}.coadd1d")
                self.set_path("pypeit_coadd1d_file", coadd_file_path)
                with open(coadd_file_path) as file:
                    coadd_file_lines = file.readlines()
                output_path = os.path.join(run_dir, f"{self.name}_{arm}_coadded.fits")
                sensfunc_path = self.get_path("pypeit_sensitivity_file")
                # Remove non-science files
                for line in coadd_file_lines[coadd_file_lines.index("coadd1d read\n"):]:
                    if "STD,FLUX" in line or "STD,TELLURIC" in line:
                        coadd_file_lines.remove(line)

                self._set_pypeit_coadd1d_file(coadd_file_lines)
                # Re-insert parameter lines
                self.add_pypeit_user_param(param=["coadd1d", "coaddfile"], value=output_path, file_type="coadd1d")
                self.add_pypeit_user_param(param=["coadd1d", "sensfuncfile"], value=sensfunc_path, file_type="coadd1d")
                self.add_pypeit_user_param(param=["coadd1d", "wave_method"], value="velocity", file_type="coadd1d")
                u.write_list_to_file(coadd_file_path, self._get_pypeit_coadd1d_file())
                spec.pypeit_coadd_1dspec(coadd1d_file=coadd_file_path)
                self.add_coadded_image(coadd_file_path, key=arm)

            self.stages_complete[f'5.{i + 1}-pypeit_coadd_{arm}'] = Time.now()

        self._current_arm = None

    def proc_convert_to_marz_format(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Convert co-added 1D spectra to Marz format?",
                                        stage_name='6-convert_to_marz_format'):
            for arm in self.coadded:
                self.coadded[arm].convert_to_marz_format()
            self.stages_complete[f'6-convert_to_marz_format'] = Time.now()

    def add_coadded_image(self, img: Union[str, image.Coadded1DSpectrum], **kwargs):
        arm = kwargs["key"]
        if isinstance(img, str):
            img = image.from_path(
                path=img,
                grism=arm,
                cls=image.Coadded1DSpectrum
            )
        img.epoch = self
        self.coadded[arm] = img
        return img

    def add_frame_raw(self, raw_frame: image.Image):
        arm = self._get_current_arm()
        self.frames_raw[arm].append(raw_frame)
        self.sort_frame(raw_frame)

    def sort_frame(self, frame: image.Image):
        arm = self._get_current_arm()
        if frame.frame_type == "bias":
            self.frames_bias[arm].append(frame)
        elif frame.frame_type == "science":
            self.frames_science[arm].append(frame)
        elif frame.frame_type == "standard":
            self.frames_standard[arm].append(frame)
        elif frame.frame_type == "dark":
            self.frames_dark[arm].append(frame)

    def read_pypeit_sorted_file(self):
        arm = self._get_current_arm()
        if "pypeit_setup_dir" in self.paths and self.paths["pypeit_setup_dir"] is not None:
            setup_files = self.paths["pypeit_setup_dir"]
            sorted_path = os.path.join(setup_files,
                                       filter(lambda f: f"vlt_xshooter_{arm}" in f and f.endswith(".sorted"),
                                              os.listdir(setup_files)).__next__())
            with open(sorted_path) as sorted_file:
                file = sorted_file.readlines()
            self._pypeit_sorted_file[arm] = file
            for setup in ["A", "B", "C"]:
                info = self.setup_info(setup=setup)
                arm_this = info["arm"].lower()
                self._cfg_split_letters[arm_this] = setup
        else:
            raise KeyError("pypeit_setup_dir has not been set.")

    def read_pypeit_file(self, config: str):
        if "pypeit_dir" in self.paths and self.paths["pypeit_dir"] is not None:
            arm = self._get_current_arm()
            filename = f"{self._instrument_pypeit}_{arm}_{config}"
            self._read_pypeit_file(filename=filename)
            return self._pypeit_file[arm]
        else:
            raise KeyError("pypeit_run_dir has not been set.")

    def pypeit_flux_title(self):
        return f"{self._instrument_pypeit}_{self._get_current_arm()}.flux"

    def get_path(self, key):
        key = self._get_key_arm(key)
        return self.paths[key]

    def set_path(self, key: str, value: str):
        key = self._get_key_arm(key)
        self.paths[key] = value

    def get_frames_science(self):
        return self.frames_science[self._get_current_arm()]

    def get_frames_standard(self):
        return self.frames_standard[self._get_current_arm()]

    def get_binning(self):
        return self.binning[self._get_current_arm()]

    def set_binning(self, binning: str):
        self.binning[self._get_current_arm()] = binning
        return binning

    def get_binning_std(self):
        return self.binning_std[self._get_current_arm()]

    def set_binning_std(self, binning: str):
        self.binning_std[self._get_current_arm()] = binning
        return binning

    def get_decker(self):
        return self.decker[self._get_current_arm()]

    def set_decker(self, decker: str):
        self.decker[self._get_current_arm()] = decker
        return decker

    def get_decker_std(self):
        return self.decker_std[self._get_current_arm()]

    def set_decker_std(self, decker: str):
        self.decker_std[self._get_current_arm()] = decker
        return decker

    def _get_current_arm(self):
        if self._current_arm is not None:
            return self._current_arm
        else:
            raise ValueError("self._current_arm is not set (no arm currently active).")

    def _set_pypeit_file(self, lines: list):
        self._pypeit_file[self._get_current_arm()] = lines

    def _get_pypeit_file(self):
        return self._pypeit_file[self._get_current_arm()]

    def _get_pypeit_sorted_file(self):
        return self._pypeit_sorted_file[self._get_current_arm()]

    def _set_pypeit_file_std(self, lines: list):
        self._pypeit_file_std[self._get_current_arm()] = lines

    def _get_pypeit_file_std(self):
        return self._pypeit_file_std[self._get_current_arm()]

    def _set_pypeit_coadd1d_file(self, lines: list):
        self._pypeit_coadd1d_file[self._get_current_arm()] = lines

    def _get_pypeit_coadd1d_file(self):
        return self._pypeit_coadd1d_file[self._get_current_arm()]

    def _get_key_arm(self, key):
        arm = self._get_current_arm()
        key = f"{arm}_{key}"
        return key

    @classmethod
    def stages(cls):
        param_dict = super().stages()
        param_dict.update({
            "2-pypeit_setup": None,
            "3.1-pypeit_run_uvb": None,
            "3.2-pypeit_run_vis": None,
            "3.3-pypeit_run_nir": None,
            "4.1-pypeit_flux_calib_uvb": None,
            "4.2-pypeit_flux_calib_vis": None,
            "4.3-pypeit_flux_calib_nir": None,
            "5.1-pypeit_coadd_uvb": None,
            "5.2-pypeit_coadd_vis": None,
            "5.3-pypeit_coadd_nir": None,
            "6-convert_to_marz_format": None
        })
        return param_dict

    def _output_dict(self):
        return {
            "stages": self.stages_complete,
            "paths": self.paths,
        }

    def load_output_file(self, **kwargs):
        outputs = super().load_output_file(mode="spectroscopy", **kwargs)
        if outputs not in [None, True, False]:
            self.stages_complete.update(outputs["stages"])
            if "paths" in outputs:
                self.paths.update(outputs[f"paths"])
        return outputs

    def write_pypeit_file_std(self):
        """
        Rewrites the stored .pypeit file to disk.
        :return: path of .pypeit file.
        """
        pypeit_lines = self._get_pypeit_file_std()
        if pypeit_lines is not None:
            pypeit_file_path = os.path.join(self.get_path("pypeit_file_std"), )
            u.write_list_to_file(path=pypeit_file_path, file=pypeit_lines)
        else:
            raise ValueError("pypeit_file_std has not yet been read.")
        return pypeit_file_path
