import os
from typing import Union
import warnings

from astropy.time import Time

import craftutils.utils as u
import craftutils.params as p
import craftutils.observation.image as image
import craftutils.wrap.pypeit as pypeit

from ..epoch import Epoch, active_epochs


class SpectroscopyEpoch(Epoch):
    instrument_name = "dummy-instrument"
    mode = "spectroscopy"
    grisms = {}
    frame_class = image.Spectrum
    coadded_class = image.Coadded1DSpectrum

    def __init__(
            self,
            param_path: str = None,
            name: str = None,
            field: Union[str, 'fld.Field'] = None,
            data_path: str = None,
            instrument: str = None,
            date: Union[str, Time] = None,
            program_id: str = None,
            target: str = None,
            grism: str = None,
            decker: str = None,
            **kwargs
    ):
        super().__init__(
            param_path=param_path,
            name=name,
            field=field,
            data_path=data_path,
            instrument=instrument,
            date=date,
            program_id=program_id,
            target=target,
            **kwargs
        )

        self.pypeit_dir: str = None

        self.decker = decker
        self.decker_std = decker
        self.grism = grism
        if grism is None or grism not in self.grisms:
            warnings.warn("grism not configured.")

        self.obj = target

        self.standards_raw = []
        self._instrument_pypeit = self.instrument_name.replace('-', '_')

        self._pypeit_file = None
        self._pypeit_sorted_file = None
        self._pypeit_coadd1d_file = None

    def proc_pypeit_flux(self, output_dir: str, **kwargs):
        self._pypeit_flux()
        self.update_output_file()

    def _pypeit_flux(self):
        pypeit_run_dir = self.get_path("pypeit_run_dir")
        pypeit_science_dir = os.path.join(pypeit_run_dir, "Science")
        std_reduced_filename = filter(lambda f: "spec1d" in f and "STD,FLUX" in f and f.endswith(".fits"),
                                      os.listdir(pypeit_science_dir)).__next__()
        std_reduced_path = os.path.join(pypeit_science_dir, std_reduced_filename)
        if not self.quiet:
            print(f"Using {std_reduced_path} for fluxing.")

        sensfunc_path = os.path.join(pypeit_run_dir, "sens.fits")
        # Generate sensitivity function from standard observation
        pypeit.pypeit_sensfunc(spec1dfile=std_reduced_path, outfile=sensfunc_path)
        # Generate flux setup file.
        pypeit.pypeit_flux_setup(sci_path=pypeit_science_dir, run_dir=pypeit_run_dir)
        flux_setup_path = os.path.join(pypeit_run_dir, self.pypeit_flux_title())
        # Insert name of sensitivity file to flux setup file.
        with open(flux_setup_path, "r") as flux_setup:
            flux_lines = flux_setup.readlines()
        file_first = flux_lines.index("flux read\n") + 1
        flux_lines[file_first] = flux_lines[file_first][:-1] + " " + sensfunc_path + "\n"
        # Write back to file.
        u.write_list_to_file(path=flux_setup_path, file=flux_lines)
        # Run pypeit_flux_calib
        os.system(f"pypeit_flux_calib {flux_setup_path}")

        self.set_path("pypeit_sensitivity_file", sensfunc_path)
        self.set_path("pypeit_std_reduced", std_reduced_path)
        self.set_path("pypeit_science_dir", pypeit_science_dir)
        self.set_path("pypeit_flux_setup", flux_setup_path)

    def pypeit_flux_title(self):
        return f"{self._instrument_pypeit}.flux"

    def read_pypeit_sorted_file(self):
        if "pypeit_setup_dir" in self.paths and self.paths["pypeit_setup_dir"] is not None:
            setup_files = self.paths["pypeit_setup_dir"]
            sorted_path = os.path.join(setup_files,
                                       filter(lambda f: f.endswith(".sorted"), os.listdir(setup_files)).__next__())
            with open(sorted_path) as sorted_file:
                self._pypeit_sorted_file = sorted_file.readlines()
        else:
            raise KeyError("pypeit_setup_dir has not been set.")

    def setup_info(self, setup: str):
        """
        Pulls setup info from a pypeit .sorted file.
        :param setup:
        :return:
        """
        file = self._get_pypeit_sorted_file()
        # Find start of setup description
        setup_start = file.index(f"Setup {setup}\n")
        setup_dict = {}
        i = setup_start + 1
        line = file[i]
        # Assemble a dictionary of the setup parameters.
        while line != "#---------------------------------------------------------\n":
            while line[0] == " ":
                line = line[1:]
            line = line[:-1]
            key, value = line.split(": ")
            setup_dict[key] = value
            i += 1
            line = file[i]
        return setup_dict

    def _read_pypeit_file(self, filename):
        pypeit_run_dir = os.path.join(self.paths['pypeit_dir'], filename)
        self.set_path("pypeit_run_dir", pypeit_run_dir)
        self.set_path("pypeit_file", os.path.join(pypeit_run_dir, f"{filename}.pypeit"))
        # Retrieve text from .pypeit file
        with open(self.get_path("pypeit_file"), 'r') as pypeit_file:
            pypeit_lines = pypeit_file.readlines()
            self._set_pypeit_file(pypeit_lines)
        f_start = pypeit_lines.index("data read\n") + 3
        f_end = pypeit_lines.index("data end\n")
        for line in pypeit_lines[f_start:f_end]:
            raw = image.RawSpectrum.from_pypeit_line(line=line, pypeit_raw_path=self.paths["raw_dir"])
            self.add_frame_raw(raw)
        return pypeit_lines

    def read_pypeit_file(self, setup: str):
        if "pypeit_dir" in self.paths and self.paths["pypeit_dir"] is not None:
            filename = f"{self._instrument_pypeit}_{setup}"
            self._read_pypeit_file(filename=filename)
            return self._pypeit_file
        else:
            raise KeyError("pypeit_run_dir has not been set.")

    def write_pypeit_file_science(self):
        """
        Rewrites the stored .pypeit file to disk at its original path.
        :return: path of .pypeit file.
        """
        pypeit_lines = self._get_pypeit_file()
        if pypeit_lines is not None:
            pypeit_file_path = self.get_path("pypeit_file")
            u.write_list_to_file(path=pypeit_file_path, file=pypeit_lines)
        else:
            raise ValueError("pypeit_file has not yet been read.")
        return pypeit_file_path

    def add_pypeit_user_param(self, param: list, value: str, file_type: str = "pypeit"):
        """
        Inserts a parameter for the PypeIt run at the correct point in the stored .pypeit file.
        :param param: For m
        :param value:
        :return:
        """
        if file_type == "pypeit":
            pypeit_file = self._get_pypeit_file()
        elif file_type == "coadd1d":
            pypeit_file = self._get_pypeit_coadd1d_file()
        else:
            raise ValueError(f"file_type {file_type} not recognised.")

        if pypeit_file is not None:
            # Build the final line of the setting specially.
            setting = "\t" * (len(param) - 1) + f"{param.pop()} = {value}\n"
            p_start = pypeit_file.index("# User-defined execution parameters\n") + 1
            insert_here = False
            # For each level of the param list, look to see if it's already there.
            for i, line in pypeit_file[p_start]:
                if param[0] in line:
                    p_start = i
                    break

            for i, par in enumerate(param):
                # Encase each level of the parameter in the correct number of square brackets and tabs.
                par = "\t" * i + "[" * (i + 1) + par + "]" * (i + 1) + "\n"
                # First, check if param sub-headings are already there:
                if par in pypeit_file and not insert_here:
                    p_start = pypeit_file.index(par) + 1
                else:
                    # Insert the line at correct position.
                    pypeit_file.insert(p_start, par)
                    p_start += 1
                    insert_here = True
            # Insert the final line.
            pypeit_file.insert(p_start, setting)

            if file_type == "pypeit":
                self._set_pypeit_file(lines=pypeit_file)
            elif file_type == "coadd1d":
                self._set_pypeit_coadd1d_file(lines=pypeit_file)

        else:
            raise ValueError("pypeit_file has not yet been read.")

    def add_pypeit_file_lines(self, lines: list):
        if self._get_pypeit_file() is not None:
            # Remove last two lines of file ("data end")
            pypeit_lines = self._pypeit_file[:-2]
            # Insert desired lines
            pypeit_lines += lines
            # Reinsert last two lines.
            pypeit_lines += ["data end\n", "\n"]
            self._pypeit_file = pypeit_lines
        else:
            raise ValueError("pypeit_file has not yet been read.")

    def _output_dict(self):
        output_dict = super()._output_dict()
        output_dict.update({"binning": self.binning,
                            "decker": self.decker})
        return output_dict

    def proc_pypeit_setup(self, output_dir: str, **kwargs):
        pass

    def proc_pypeit_run(self, output_dir: str, **kwargs):
        pass

    def proc_pypeit_coadd(self, output_dir: str, **kwargs):
        pass

    def proc_convert_to_marz_format(self, output_dir: str, **kwargs):
        pass

    def find_science_attributes(self):
        frames = self.get_frames_science()
        if frames:
            frame = frames[0]
            self.set_binning(frame.binning)
            self.set_decker(frame.decker)
        else:
            raise ValueError(f"Science frames list is empty.")
        return frame

    def find_std_attributes(self):
        frames = self.get_frames_standard()
        if frames:
            frame = frames[0]
            self.set_binning_std(frame.binning)
            self.set_decker_std(frame.decker)
        else:
            raise ValueError(f"Standard frames list is empty.")
        return frame

    def get_frames_science(self):
        return self.frames_science

    def get_frames_standard(self):
        return self.frames_standard

    def get_decker(self):
        return self.decker

    def set_decker(self, decker: str):
        self.decker = decker
        return decker

    def get_decker_std(self):
        return self.decker_std

    def set_decker_std(self, decker: str):
        self.decker_std = decker
        return decker

    def _set_pypeit_file(self, lines: list):
        self._pypeit_file = lines

    def _get_pypeit_file(self):
        return self._pypeit_file

    def _set_pypeit_coadd1d_file(self, lines: list):
        self._pypeit_coadd1d_file = lines

    def _get_pypeit_coadd1d_file(self):
        return self._pypeit_coadd1d_file

    def _get_pypeit_sorted_file(self):
        return self._pypeit_sorted_file

    @classmethod
    def stages(cls):
        epoch_stages = super().stages()
        epoch_stages.update({
            "pypeit_setup": {
                "method": cls.proc_pypeit_setup,
                "message": "Do PypeIt setup?",
                "default": True,
                "keywords": {}
            },
            "pypeit_run": {
                "method": cls.proc_pypeit_run,
                "message": "Run PypeIt?",
                "default": True,
                "keywords": {
                    "do_not_reuse_masters": False
                }
            },
            "pypeit_flux": {
                "method": cls.proc_pypeit_flux,
                "message": "Flux-calibrate spectrum using PypeIt?",
                "default": True,
                "keywords": {},
            },
            "pypeit_coadd": {
                "method": cls.proc_pypeit_coadd,
                "message": "Coadd spectra using PypeIt?",
                "default": True,
                "keywords": {},
            },
            "convert_to_marz_format": {
                "method": cls.proc_convert_to_marz_format,
                "message": "Convert co-added 1D spectra to Marz format?",
                "default": False,
                "keywords": {}
            }
        })
        return epoch_stages

    @classmethod
    def select_child_class(cls, instrument: str):
        instrument = instrument.lower()
        from .eso import XShooterSpectroscopyEpoch, FORS2SpectroscopyEpoch
        if instrument == "vlt-fors2":
            return FORS2SpectroscopyEpoch
        elif instrument == "vlt-xshooter":
            return XShooterSpectroscopyEpoch
        elif instrument in p.instruments_spectroscopy:
            return SpectroscopyEpoch
        else:
            raise ValueError(f"Unrecognised instrument {instrument}")

    @classmethod
    def from_file(
            cls,
            param_file: Union[str, dict],
            field: 'fld.Field' = None
    ):
        name, param_file, param_dict = p.params_init(param_file)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")
        instrument = param_dict["instrument"].lower()
        fld_from_dict = param_dict.pop("field")
        if field is None:
            field = fld_from_dict
        if 'target' in param_dict:
            target = param_dict.pop('target')
        else:
            target = None
        sub_cls = cls.select_child_class(instrument=instrument)
        # if sub_cls is SpectroscopyEpoch:
        print(sub_cls)
        return sub_cls(
            # name=name,
            field=field,
            # param_path=param_file,
            # data_path=os.path.join(config["top_data_dir"], param_dict['data_path']),
            # instrument=instrument,
            date=param_dict.pop("date"),
            program_id=param_dict.pop("program_id"),
            target=target,
            **param_dict
        )
        # else:
        # return sub_cls.from_file(param_file=param_file, field=field)

    @classmethod
    def from_params(
            cls,
            name: str,
            field: Union['fld.Field', str] = None,
            instrument: str = None
    ):
        if name in active_epochs:
            return active_epochs[name]
        print("Initializing epoch...")
        instrument = instrument.lower()
        field_name, field = cls._from_params_setup(name=name, field=field)
        path = cls.build_param_path(
            field_name=field_name,
            instrument_name=instrument,
            epoch_name=name)
        epoch = cls.from_file(param_file=path, field=field)
        print("In from_params:", epoch)
        return epoch

    @classmethod
    def build_param_path(cls, field_name: str, instrument_name: str, epoch_name: str):
        return os.path.join(p.param_dir, "fields", field_name, "spectroscopy", instrument_name, epoch_name)
