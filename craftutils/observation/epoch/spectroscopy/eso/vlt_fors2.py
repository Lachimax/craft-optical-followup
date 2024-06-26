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
            output_dir: str,
            **kwargs
    ):
        if "setups" in kwargs and kwargs["setups"]:
            setups = kwargs["setups"]
        else:
            setups = ["G"]

        pypeit_dir = self.get_pypeit_path("pypeit_dir")
        setup_files = os.path.join(pypeit_dir, 'setup_files')

        self.set_pypeit_path("pypeit_setup_dir", setup_files)

        # os.system(f"rm {setup_files}*")
        # Generate .sorted file and others
        spec.pypeit_setup(
            root=self.get_path('download'),
            output_path=pypeit_dir,
            spectrograph=self._instrument_pypeit
        )
        # Generate files to use for run. Set cfg_split to "A" because that corresponds to Chip 1, which is the only
        # one we need to worry about.

        # Read .sorted file
        self.read_pypeit_sorted_file()

        for config in setups:

            self.add_configuration(config)
            config_dir = os.path.join(pypeit_dir, self._config_filename(config))
            self.set_configuration_property(
                config=config,
                key="pypeit_run_dir",
                value=config_dir
            )
            self.set_configuration_property(
                config=config,
                key="pypeit_science_dir",
                value=os.path.join(config_dir, 'Science')
            )

            spec.pypeit_setup(
                root=self.get_path('download'),
                output_path=pypeit_dir,
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
            cfg_break = "##########################################################\n"
            if cfg_break in self._pypeit_sorted_file[std_start_index:]:
                std_end_index = self._pypeit_sorted_file[std_start_index:].index(cfg_break) + std_start_index
            else:
                std_end_index = self._pypeit_sorted_file[std_start_index:].index("##end\n")
            std_lines = self._pypeit_sorted_file[std_start_index:std_end_index]
            # Read in .pypeit file
            self.read_pypeit_file(config=config)
            # Add lines to set slit prediction to "nearest" in .pypeit file.
            self.add_pypeit_user_param(
                param=["calibrations", "slitedges", "sync_predict"],
                value="nearest",
                config=config
            )
            # Insert bias lines from .sorted file
            self.add_pypeit_file_lines(
                config=config,
                lines=bias_lines + std_lines
            )
            # Write modified .pypeit file back to disk.
            self.write_pypeit_file_science(config=config)

    def proc_pypeit_run(
            self,
            output_dir: str,
            **kwargs
    ):
        do_not_reuse_masters = False
        if "do_not_reuse_masters" in kwargs:
            do_not_reuse_masters = kwargs["do_not_reuse_masters"]
        for config in self.configurations:
            spec.run_pypeit(
                pypeit_file=self.get_configuration_property(
                    config=config,
                    key='pypeit_file'
                ),
                redux_path=self.get_configuration_property(
                    config=config,
                    key='pypeit_run_dir'
                ),
                do_not_reuse_masters=do_not_reuse_masters
            )

    def proc_pypeit_coadd(self, no_query: bool = False, **kwargs):
        for config in self.configurations:
            for file in filter(lambda f: "spec1d" in f, os.listdir(self.get_configuration_property(config, "pypeit_science_dir"))):
                path = os.path.join(self.get_configuration_property(
                    config=config,
                    key="pypeit_science_dir",
                ))
                # os.system(f"pypeit_show_1dspec {path}")
