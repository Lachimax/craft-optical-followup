import os

import astropy.units as units
from astropy.time import Time

import craftutils.wrap.pypeit as spec
import craftutils.observation.image as image

from .eso import ESOSpectroscopyEpoch


class FORS2SpectroscopyEpoch(ESOSpectroscopyEpoch):
    instrument_name = "vlt-fors2"
    _instrument_pypeit = "vlt_fors2"
    grisms = {
        "GRIS_300I": {
            "lambda_min": 6000 * units.angstrom,
            "lambda_max": 11000 * units.angstrom
        }}

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)

    def proc_pypeit_setup(
            self,
            output_dir:
            str, **kwargs
    ):
        if "setups" in kwargs and kwargs["setups"]:
            setups = kwargs["setups"]
        else:
            setups = ["G"]
        setup_files = os.path.join(output_dir, 'setup_files', '')
        self.paths["pypeit_setup_dir"] = setup_files
        self.paths["pypeit_dir"] = output_dir
        # os.system(f"rm {setup_files}*")
        # Generate .sorted file and others
        spec.pypeit_setup(
            root=self.paths['download'],
            output_path=output_dir,
            spectrograph=self._instrument_pypeit
        )
        # Generate files to use for run. Set cfg_split to "A" because that corresponds to Chip 1, which is the only
        # one we need to worry about.

        # Read .sorted file
        self.read_pypeit_sorted_file()

        for config in setups:
            spec.pypeit_setup(
                root=self.paths['download'],
                output_path=output_dir,
                spectrograph=self._instrument_pypeit,
                cfg_split=config
            )

            # Retrieve bias files from .sorted file.
            bias_lines = list(filter(lambda s: "bias" in s and "CHIP1" in s, self._pypeit_sorted_file))
            # Find line containing information for standard observation.
            std_line = filter(lambda s: "standard" in s and "CHIP1" in s, self._pypeit_sorted_file).__next__()
            std_raw = image.RawSpectrum.from_pypeit_line(std_line, pypeit_raw_path=self.paths['download'])
            self.standards_raw.append(std_raw)
            std_start_index = self._pypeit_sorted_file.index(std_line)
            # Find last line of the std-obs configuration (encapsulating the required calibration files)
            std_end_index = self._pypeit_sorted_file[std_start_index:].index(
                "##########################################################\n") + std_start_index
            std_lines = self._pypeit_sorted_file[std_start_index:std_end_index]
            # Read in .pypeit file
            self.read_pypeit_file(setup=config)
            # Add lines to set slit prediction to "nearest" in .pypeit file.
            self.add_pypeit_user_param(param=["calibrations", "slitedges", "sync_predict"], value="nearest")
            # Insert bias lines from .sorted file
            self.add_pypeit_file_lines(lines=bias_lines + std_lines)
            # Write modified .pypeit file back to disk.
            self.write_pypeit_file_science()

    def proc_pypeit_run(self, no_query: bool = False, do_not_reuse_masters: bool = False, **kwargs):
        spec.run_pypeit(
            pypeit_file=self.paths['pypeit_file'],
            redux_path=self.paths['pypeit_run_dir'],
            do_not_reuse_masters=do_not_reuse_masters
        )

    def proc_pypeit_coadd(self, no_query: bool = False, **kwargs):
        for file in filter(lambda f: "spec1d" in f, os.listdir(self.paths["pypeit_science_dir"])):
            path = os.path.join(self.paths["pypeit_science_dir"], file)
            os.system(f"pypeit_show_1dspec {path}")
