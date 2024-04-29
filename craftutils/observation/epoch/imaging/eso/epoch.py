import os
import shutil
from typing import Union

import numpy as np

import astropy.units as units
from astropy.time import Time

import craftutils.utils as u
import craftutils.params as p
import craftutils.observation.image as image
from craftutils.fits_files import detect_edges
from craftutils.observation.survey import Survey

from craftutils.observation.epoch.imaging import ImagingEpoch
from ...epoch import _retrieve_eso_epoch


class ESOImagingEpoch(ImagingEpoch):
    instrument_name = "dummy-instrument"
    mode = "imaging"
    eso_name = None

    def __init__(
            self,
            name: str = None,
            field: 'fld.Field' = None,
            param_path: str = None,
            data_path: str = None,
            instrument: str = None,
            program_id: str = None,
            date: Union[str, Time] = None,
            target: str = None,
            standard_epochs: list = None,
            source_extractor_config: dict = None,
            **kwargs
    ):
        u.debug_print(2, f"ESOImagingEpoch.__init__(): kwargs ==", kwargs)
        super().__init__(
            name=name,
            field=field,
            param_path=param_path,
            data_path=data_path,
            instrument=instrument,
            date=date,
            program_id=program_id,
            target=target,
            standard_epochs=standard_epochs,
            source_extractor_config=source_extractor_config,
            **kwargs)

        self.frames_esoreflex_backgrounds = {}

        self.load_output_file(mode="imaging")

    @classmethod
    def stages(cls):
        super_stages = super().stages()

        super_stages["initial_setup"].update(
            {
                "keywords": {"skip_esoreflex_copy": False}
            }
        )

        stages = {
            "download": {
                "method": cls.proc_download,
                "message": "Download raw data from ESO archive?",
                "default": True,
                "keywords": {
                    "alternate_dir": None
                }
            },
            "initial_setup": super_stages["initial_setup"],
            "sort_reduced": {
                "method": cls.proc_sort_reduced,
                "message": "Sort ESOReflex products? Requires reducing data with ESOReflex first.",
                "default": True,
                "keywords": {
                    "alternate_dir": None,  # alternate directory to pull reduced files from.
                    "delete_eso_output": False
                }
            },
            "trim_reduced": {
                "method": cls.proc_trim_reduced,
                "message": "Trim reduced images?",
                "default": True,
            },
            "convert_to_cs": {
                "method": cls.proc_convert_to_cs,
                "message": "Convert image values to counts/second?",
                "default": True,
                "keywords": {
                    "upper_only": False
                }
            },
        }
        return stages

    def proc_download(self, output_dir: str, **kwargs):

        # Check for alternate directory.
        alt_dir = None
        if "alternate_dir" in kwargs and isinstance(kwargs["alternate_dir"], str):
            alt_dir = kwargs["alternate_dir"]

        if alt_dir is None:
            r = self.retrieve(output_dir)
            if r:
                return True
        else:
            u.rmtree_check(output_dir)
            shutil.copytree(alt_dir, output_dir)
            return True

    def retrieve(self, output_dir: str):
        """
        Check ESO archive for the epoch raw frames, and download those frames and associated files.

        :return:
        """
        r = []
        r = _retrieve_eso_epoch(self, path=output_dir)
        return r

    def _initial_setup(self, output_dir: str, **kwargs):
        u.debug_print(2, f"ESOImagingEpoch._initial_setup(): {self.paths=}")
        raw_dir = self.get_path("download")
        data_dir = self.data_path
        data_title = self.name

        p.set_eso_user()

        self.frames_science = {}
        self.frames_flat = {}
        self.frames_bias = []
        self.frames_raw = []
        self.filters = []

        # Write tables of fits files to main directory; firstly, science images only:
        tbl = image.fits_table(
            input_path=raw_dir,
            output_path=os.path.join(data_dir, data_title + "_fits_table_science.csv"),
            science_only=True
        )
        # Then including all calibration files
        tbl_full = image.fits_table(
            input_path=raw_dir,
            output_path=os.path.join(data_dir, data_title + "_fits_table_all.csv"),
            science_only=False
        )
        image.fits_table_all(
            input_path=raw_dir,
            output_path=os.path.join(data_dir, data_title + "_fits_table_detailed.csv"),
            science_only=False
        )

        not_science = []
        # We do this in two pieces so that we don't add calibration frames that aren't for relevant filters
        # (which the ESO archive often associates anyway, especially with HAWK-I)
        for i, row in enumerate(tbl_full):
            path = os.path.join(raw_dir, row["identifier"])
            cls = image.ImagingImage.select_child_class(instrument_name=self.instrument_name, mode="imaging")
            img = image.from_path(path, cls=cls)
            img.extract_frame_type()
            img.extract_filter()
            u.debug_print(1, self.instrument_name, cls, img.name, img.frame_type)
            # The below will also update the filter list.
            u.debug_print(
                2,
                f"_initial_setup(): Adding frame {img.name}, type {img.frame_type}/{type(img)}, to {self}, type {type(self)}")
            if img.frame_type == "science":
                self.add_frame_raw(img)
            else:
                not_science.append(img)

        for img in not_science:
            if img.filter_name in self.filters or img.frame_type == "bias":
                self.add_frame_raw(img)

        u.debug_print(2, f"ESOImagingEpoch._initial_setup(): {self.frames_science=}")
        # Collect and save some stats on those filters:
        for i, fil in enumerate(self.filters):
            if len(self.frames_science[fil]) == 0:
                self.filters.remove(fil)
                self.frames_science.pop(fil)
                continue
            exp_times = list(map(lambda frame: frame.extract_exposure_time().value, self.frames_science[fil]))
            u.debug_print(1, "exposure times:")
            u.debug_print(1, exp_times)
            self.exp_time_mean[fil] = np.nanmean(exp_times) * units.second
            self.exp_time_err[fil] = np.nanstd(exp_times) * units.second

            airmasses = list(map(lambda frame: frame.extract_airmass(), self.frames_science[fil]))

            self.airmass_mean[fil] = np.nanmean(airmasses)
            self.airmass_err[fil] = max(
                np.nanmax(airmasses) - self.airmass_mean[fil],
                self.airmass_mean[fil] - np.nanmin(airmasses)
            )

        inst_reflex_dir = {
            "vlt-fors2": "fors",
            "vlt-hawki": "hawki"
        }[self.instrument_name]

        inst_reflex_dir = os.path.join(p.config["esoreflex_input_dir"], inst_reflex_dir)
        u.mkdir_check_nested(inst_reflex_dir, remove_last=False)

        survey_raw_path = None
        if isinstance(self.field.survey, Survey) and self.field.survey.raw_stage_path is not None:
            survey_raw_path = os.path.join(self.field.survey.raw_stage_path, self.field.name, self.instrument_name)
            u.mkdir_check_nested(survey_raw_path, remove_last=False)

        if not ("skip_esoreflex_copy" in kwargs and kwargs["skip_esoreflex_copy"]):
            for file in os.listdir(raw_dir):
                if not self.quiet:
                    print(f"Copying {file} to ESOReflex input directory...")
                origin = os.path.join(raw_dir, file)
                shutil.copy(origin, os.path.join(p.config["esoreflex_input_dir"], inst_reflex_dir))
                if not self.quiet:
                    print("Done.")

                if survey_raw_path is not None:
                    survey_raw_path_file = os.path.join(
                        survey_raw_path,
                        file
                    )
                    if not self.quiet:
                        print(f"Copying {file} to {survey_raw_path_file}...")
                    shutil.copy(
                        origin,
                        survey_raw_path_file
                    )
                    if not self.quiet:
                        print("Done.")

        # This line looks for a non-empty frames_science list
        i = 0
        while not self.frames_science[self.filters[i]]:
            i += 1
        tmp = self.frames_science[self.filters[i]][0]
        if self.date is None:
            self.set_date(tmp.extract_date_obs())
        if self.target is None:
            self.set_target(tmp.extract_object())
        if self.program_id is None:
            self.set_program_id(tmp.extract_program_id())

        self.update_output_file()

        # if str(self.field.survey) == "FURBY":
        #     u.system_command_verbose(
        #         f"furby_vlt_ob {self.field.name} {tmp.filter.band_name} --observed {self.date_str()}"
        #     )
        # u.system_command_verbose(f"furby_vlt_ob {self.field.name} {tmp.filter.band_name} --completed")

        try:
            u.system_command_verbose("esoreflex")
        except SystemError:
            print("Could not open ESO Reflex; may not be installed, or installed to other environment.")

    def proc_sort_reduced(self, output_dir: str, **kwargs):
        self.sort_after_esoreflex(output_dir=output_dir, **kwargs)

    def sort_after_esoreflex(self, output_dir: str, **kwargs):
        """
        Scans through the ESO Reflex directory for the files matching this epoch, and puts them where we want them.
        :param output_dir:
        :param kwargs:
        :return:
        """

        self.frames_reduced = {}
        self.frames_esoreflex_backgrounds = {}

        # Check for alternate directory.
        if "alternate_dir" in kwargs and isinstance(kwargs["alternate_dir"], str):
            eso_dir = kwargs["alternate_dir"]
            expect_sorted = True
            if "expect_sorted" in kwargs and isinstance(kwargs["expect_sorted"], bool):
                expect_sorted = kwargs["expect_sorted"]
        else:
            eso_dir = os.path.join(p.config['esoreflex_output_dir'], "reflex_end_products")
            expect_sorted = False

        if "delete_eso_output" in kwargs:
            delete_output = kwargs["delete_eso_output"]
        else:
            delete_output = False

        if not self.quiet:
            print(f"Copying files from {eso_dir} to {output_dir}")
            print(self.date_str())

        if os.path.isdir(eso_dir):
            if expect_sorted:
                shutil.rmtree(output_dir)
                shutil.copytree(
                    eso_dir,
                    output_dir,
                )

                science = os.path.join(output_dir, "science")
                for fil in filter(lambda d: os.path.isdir(os.path.join(science, d)), os.listdir(science)):
                    output_subdir = os.path.join(science, fil)
                    if not self.quiet:
                        print(f"Adding reduced science images from {output_subdir}")
                    for file in filter(lambda f: f.endswith(".fits"), os.listdir(output_subdir)):
                        path = os.path.join(output_subdir, file)
                        # TODO: This (and other FORS2Image instances in this method) WILL NOT WORK WITH HAWKI. Must make more flexible.
                        img = image.from_path(
                            path,
                            cls=image.FORS2Image
                        )
                        self.add_frame_reduced(img)
                backgrounds = os.path.join(output_dir, "backgrounds")
                for fil in filter(lambda d: os.path.isdir(os.path.join(backgrounds, d)), os.listdir(backgrounds)):
                    output_subdir = os.path.join(backgrounds, fil)
                    if not self.quiet:
                        print(f"Adding background images from {output_subdir}")
                    for file in filter(lambda f: f.endswith(".fits"), os.listdir(output_subdir)):
                        path = os.path.join(output_subdir, file)
                        img = image.from_path(
                            path,
                            cls=image.FORS2Image
                        )
                        self.add_frame_background(img)

            else:

                # The ESOReflex output directory is structured in a very specific way, which we now traverse.
                mjd = int(self.mjd())
                obj = self.target.lower()
                if not self.quiet:
                    print(f"Looking for data with object '{obj}' and MJD of observation {mjd} inside {eso_dir}")
                # Look for files with the appropriate object and MJD, as recorded in output_values

                # List directories in eso_output_dir; these are dates on which data was reduced using ESOReflex.
                date_dirs = filter(
                    lambda d: os.path.isdir(os.path.join(eso_dir, d)),
                    os.listdir(eso_dir)
                )
                date_dirs = map(lambda d: os.path.join(eso_dir, d), date_dirs)
                for date_dir in date_dirs:
                    if not self.quiet:
                        print(f"Searching {date_dir}")
                    eso_subdirs = filter(
                        lambda d: os.path.isdir(os.path.join(date_dir, d)) and self.eso_name in d,
                        os.listdir(date_dir)
                    )
                    eso_subdirs = list(map(
                        lambda d: os.path.join(os.path.join(date_dir, d)),
                        eso_subdirs
                    ))
                    for subpath in eso_subdirs:
                        if not self.quiet:
                            print(f"\tSearching {subpath}")
                        self._sort_after_esoreflex(
                            output_dir=output_dir,
                            date_dir=date_dir,
                            obj=obj,
                            mjd=mjd,
                            delete_output=delete_output,
                            subpath=subpath,
                            **kwargs
                        )

        else:
            raise IOError(f"ESO output directory '{eso_dir}' not found.")

        if not self.frames_reduced:
            u.debug_print(2, "ESOImagingEpoch._sort_after_esoreflex(): kwargs ==", kwargs)

            print(f"WARNING: No reduced frames were found in the target directory {eso_dir}.")

    def _sort_after_esoreflex(
            self,
            output_dir: str,
            date_dir: str,
            obj: str,
            mjd: int,
            delete_output: bool,
            subpath: str,
            **kwargs
    ):
        """

        :param output_dir:
        :param date_dir:
        :param obj:
        :param mjd:
        :param kwargs:
        :return:
        """

    def proc_trim_reduced(self, output_dir: str, **kwargs):
        self.trim_reduced(
            output_dir=output_dir,
            **kwargs
        )

    def trim_reduced(
            self,
            output_dir: str,
            **kwargs
    ):

        u.mkdir_check(os.path.join(output_dir, "backgrounds"))
        u.mkdir_check(os.path.join(output_dir, "science"))

        u.debug_print(
            2, f"ESOImagingEpoch.trim_reduced(): {self}.frames_esoreflex_backgrounds ==",
            self.frames_esoreflex_backgrounds)

        self.frames_trimmed = {}
        for fil in self.filters:
            self.check_filter(fil)

        edged = False

        up_left = 0
        up_right = 0
        up_bottom = 0
        up_top = 0

        dn_left = 0
        dn_right = 0
        dn_bottom = 0
        dn_top = 0

        for fil in self.filters:
            fil_path_back = os.path.join(output_dir, "backgrounds", fil)
            fil_path_science = os.path.join(output_dir, "science", fil)
            u.mkdir_check(fil_path_back)
            u.mkdir_check(fil_path_science)

            if not edged:
                # Find borders of noise frame using backgrounds.
                # First, make sure that the background we're using is for the top chip.
                i = 0
                img = self.frames_esoreflex_backgrounds[fil][i]
                while img.extract_chip_number() != 1:
                    u.debug_print(1, i, img.extract_chip_number())
                    i += 1
                    img = self.frames_esoreflex_backgrounds[fil][i]
                up_left, up_right, up_bottom, up_top = detect_edges(img.path)
                # Ditto for the bottom chip.
                i = 0
                img = self.frames_esoreflex_backgrounds[fil][i]
                while img.extract_chip_number() != 2:
                    i += 1
                    img = self.frames_esoreflex_backgrounds[fil][i]
                dn_left, dn_right, dn_bottom, dn_top = detect_edges(img.path)
                up_left = up_left + 5
                up_right = up_right - 5
                up_top = up_top - 5
                dn_left = dn_left + 5
                dn_right = dn_right - 5
                dn_bottom = dn_bottom + 5

                edged = True

            for i, frame in enumerate(self.frames_esoreflex_backgrounds[fil]):
                if self.is_excluded(frame):
                    continue
                new_path = os.path.join(
                    fil_path_back,
                    frame.filename.replace(".fits", "_trim.fits")
                )
                if not self.quiet:
                    print(f'Trimming {i} {frame}')

                # Split the files into upper CCD and lower CCD
                if frame.extract_chip_number() == 1:
                    frame.trim(left=up_left, right=up_right, top=up_top, bottom=up_bottom, output_path=new_path)
                elif frame.extract_chip_number() == 2:
                    frame.trim(left=dn_left, right=dn_right, top=dn_top, bottom=dn_bottom, output_path=new_path)
                else:
                    raise ValueError('Invalid chip ID; could not trim based on upper or lower chip.')

            # Repeat for science images

            for i, frame in enumerate(self.frames_reduced[fil]):
                if self.is_excluded(frame):
                    continue
                # Split the files into upper CCD and lower CCD
                new_file = frame.filename.replace(".fits", "_trim.fits")
                new_path = os.path.join(fil_path_science, new_file)
                frame.set_header_item(
                    key='GAIN',
                    value=frame.extract_gain())
                frame.set_header_item(
                    key='SATURATE',
                    value=65535.)
                frame.set_header_item(
                    key='BUNIT',
                    value="ct"
                )

                frame.write_fits_file()

                if frame.extract_chip_number() == 1:
                    trimmed = frame.trim(
                        left=up_left,
                        right=up_right,
                        top=up_top,
                        bottom=up_bottom,
                        output_path=new_path)
                    self.add_frame_trimmed(trimmed)

                elif frame.extract_chip_number() == 2:
                    trimmed = frame.trim(
                        left=dn_left,
                        right=dn_right,
                        top=dn_top,
                        bottom=dn_bottom,
                        output_path=new_path)
                    self.add_frame_trimmed(trimmed)

    def proc_convert_to_cs(self, output_dir: str, **kwargs):
        self.convert_to_cs(
            output_dir=output_dir,
            **kwargs
        )

    def convert_to_cs(self, output_dir: str, **kwargs):

        self.frames_normalised = {}

        if "upper_only" in kwargs:
            upper_only = kwargs["upper_only"]
        else:
            upper_only = False

        u.mkdir_check(output_dir)
        u.mkdir_check(os.path.join(output_dir, "science"))
        u.mkdir_check(os.path.join(output_dir, "backgrounds"))

        for fil in self.filters:
            fil_path_science = os.path.join(output_dir, "science", fil)
            fil_path_back = os.path.join(output_dir, "backgrounds", fil)
            u.mkdir_check(fil_path_science)
            u.mkdir_check(fil_path_back)
            for frame in self.frames_trimmed[fil]:
                if self.is_excluded(frame):
                    continue
                do = True
                if upper_only:
                    if frame.extract_chip_number() != 1:
                        do = False

                if do:
                    science_destination = os.path.join(
                        output_dir,
                        "science",
                        fil,
                        frame.filename.replace("trim", "norm"))

                    # Divide by exposure time to get an image in counts/second.
                    normed = frame.convert_to_cs(output_path=science_destination)
                    self.add_frame_normalised(normed)

    def add_frame_background(self, background_frame: Union[image.ImagingImage, str]):
        self._add_frame(
            frame=background_frame,
            frames_dict=self.frames_esoreflex_backgrounds,
            frame_type="reduced"
        )

    def check_filter(self, fil: str):
        not_none = super().check_filter(fil)
        if not_none:
            if fil not in self.frames_esoreflex_backgrounds:
                self.frames_esoreflex_backgrounds[fil] = []
            if fil not in self.frames_trimmed:
                self.frames_trimmed[fil] = []
        return not_none

    def _output_dict(self):
        output_dict = super()._output_dict()
        from ...epoch import _output_img_dict_list
        output_dict.update({
            "frames_trimmed": _output_img_dict_list(self.frames_trimmed),
            "frames_esoreflex_backgrounds": _output_img_dict_list(self.frames_esoreflex_backgrounds)
        })
        return output_dict

    def load_output_file(self, **kwargs):
        outputs = super().load_output_file(**kwargs)
        if type(outputs) is dict:
            # cls = image.Image.select_child_class(instrument=self.instrument_name, mode='imaging')
            if "frames_trimmed" in outputs:
                for fil in outputs["frames_trimmed"]:
                    if outputs["frames_trimmed"][fil] is not None:
                        for frame in outputs["frames_trimmed"][fil]:
                            self.add_frame_trimmed(frame=frame)
            if "frames_esoreflex_backgrounds" in outputs:
                for fil in outputs["frames_esoreflex_backgrounds"]:
                    if outputs["frames_esoreflex_backgrounds"][fil] is not None:
                        for frame in outputs["frames_esoreflex_backgrounds"][fil]:
                            self.add_frame_background(background_frame=frame)

        return outputs

    @classmethod
    def from_file(
            cls,
            param_file: Union[str, dict],
            name: str = None,
            field: 'fld.Field' = None
    ):

        print(param_file)
        print(name)
        name, param_file, param_dict = p.params_init(param_file, name=name)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")

        if field is None:
            field = param_dict.pop("field")
        if 'target' in param_dict:
            target = param_dict.pop('target')
        else:
            target = None

        if "field" in param_dict:
            param_dict.pop("field")
        if "instrument" in param_dict:
            param_dict.pop("instrument")
        if "name" in param_dict:
            param_dict.pop("name")
        if "param_path" in param_dict:
            param_dict.pop("param_path")

        u.debug_print(2, f"ESOImagingEpoch.from_file(), cls ==", cls)
        u.debug_print(2, 'ESOImagingEpoch.from_file(), config["top_data_dir"] == ', p.config["top_data_dir"])
        u.debug_print(2, 'ESOImagingEpoch.from_file(), param_dict["data_path"] == ', param_dict["data_path"])

        u.debug_print(2, "ESOImagingEpoch.from_file(): param_dict ==", param_dict)

        if "sextractor" in param_dict:
            se = param_dict.pop("sextractor")
        else:
            se = None

        return cls(
            name=name,
            field=field,
            param_path=param_file,
            data_path=param_dict.pop('data_path'),
            instrument=cls.instrument_name,
            program_id=param_dict.pop('program_id'),
            date=param_dict.pop('date'),
            target=target,
            source_extractor_config=se,
            **param_dict
        )
