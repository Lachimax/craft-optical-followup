# Code by Lachlan Marnoch, 2021 - 2024

import os
from typing import Union, List, Dict

from astropy.time import Time
import astropy.units as units

import craftutils.retrieve as retrieve
import craftutils.utils as u
import craftutils.params as p
import craftutils.wrap.dragons as dragons
import craftutils.observation.image as image
from ..epoch import ImagingEpoch


class GSAOIImagingEpoch(ImagingEpoch):
    """
    This class works a little differently to the other epochs; instead of keeping track of the files internally, we let
    DRAGONS do that for us. Thus, many of the dictionaries and lists of files used in other Epoch classes
    will be empty even if the files are actually being tracked correctly. See eg science_table instead.
    """
    instrument_name = "gs-aoi"
    frame_class = image.GSAOIImage
    coadded_class = image.GSAOIImage

    def __init__(
            self,
            name: str = None,
            field: Union[str, 'fld.Field'] = None,
            param_path: str = None,
            data_path: str = None,
            instrument: str = None,
            date: Union[str, Time] = None,
            program_id: str = None,
            target: str = None,
            source_extractor_config: dict = None,
            **kwargs
    ):
        super().__init__(
            name=name,
            field=field,
            param_path=param_path,
            data_path=data_path,
            instrument=instrument,
            date=date,
            program_id=program_id,
            target=target,
            source_extractor_config=source_extractor_config)
        self.science_table = None
        self.flats_lists = {}
        self.std_lists = {}

        self.load_output_file(mode="imaging")

    @classmethod
    def stages(cls):
        stages_super = super().stages()
        stages = {
            "download": {
                "method": cls.proc_download,
                "message": "Download raw data from Gemini archive?",
                "default": True,
                "keywords": {
                    "overwrite_download": True,
                }
            },
            "initial_setup": stages_super["initial_setup"],
            "reduce_flats": {
                "method": cls.proc_reduce_flats,
                "message": "Reduce flat-field images?",
                "default": True,
            },
            "reduce_science": {
                "method": cls.proc_reduce_science,
                "message": "Reduce science images?",
                "default": True,
            },
            "stack_science": {
                "method": cls.proc_stack_science,
                "message": "Stack science images with DISCO-STU?",
                "default": True,
            }
        }
        return stages

    def proc_download(self, output_dir: str, **kwargs):
        if 'overwrite_download' in kwargs:
            overwrite = kwargs['overwrite_download']
        else:
            overwrite = False
        self.retrieve(output_dir=output_dir, overwrite=overwrite)

    def retrieve(self, output_dir: str, overwrite: bool = False):
        # Get the science files
        science_files = retrieve.save_gemini_epoch(
            output=output_dir,
            program_id=self.program_id,
            coord=self.field.centre_coords,
            overwrite=overwrite
        )

        # Get the observation date from image headers if we don't have one specified
        if self.date is None:
            self.set_date(science_files["ut_datetime"][0])

        # Set up filters from retrieved science files.
        for img in science_files:
            fil = str(img["filter_name"])
            self.check_filter(fil)

        # Get the calibration files for the retrieved filters
        for fil in self.filters:
            print()
            print(f"Retrieving calibration files for {fil} band...")
            print()
            retrieve.save_gemini_calibs(
                output=output_dir,
                obs_date=self.date,
                fil=fil,
                overwrite=overwrite
            )

    def _initial_setup(self, output_dir: str, **kwargs):
        data_dir = self.data_path
        raw_dir = self.paths["download"]
        self.paths["redux_dir"] = redux_dir = os.path.join(data_dir, "redux")
        u.mkdir_check(redux_dir)
        # DO the initial database setup for DRAGONS.
        dragons.caldb_init(redux_dir=redux_dir)

        # Get a list of science files from the raw directory, using DRAGONS.
        science_list_name = "science.list"
        science_list = dragons.data_select(
            redux_dir=redux_dir,
            directory=raw_dir,
            expression="observation_class==\"science\"",
            output=science_list_name
        ).splitlines(False)[3:]
        self.paths["science_list"] = os.path.join(redux_dir, science_list_name)

        science_tbl_name = "science.csv"
        science_tbl = dragons.showd(
            input_filenames=science_list,
            descriptors="filter_name,exposure_time,object",
            output=science_tbl_name,
            csv=True,
            working_dir=redux_dir
        )
        # # Set up filters.
        # for img in science_tbl:
        #     fil = img["filter_name"]
        #     fil = fil[:fil.find("_")]
        #     self.check_filter(fil)
        u.debug_print(1, f"GSAOIImagingEpoch._inital_setup(): {self}.filters ==", self.filters)
        self.science_table = science_tbl

        # Get lists of flats for each filter.
        for fil in self.filters:
            flats_list_name = f"flats_{fil}.list"
            flats_list = dragons.data_select(
                redux_dir=redux_dir,
                directory=raw_dir,
                tags=["FLAT"],
                expression=f"filter_name==\"{fil}\"",
                output=flats_list_name
            ).splitlines(False)[3:]

            self.flats_lists[fil] = os.path.join(redux_dir, flats_list_name)
            self.frames_flat[fil] = flats_list

        # Get list of standard observations:
        std_tbl_name = "std_objects.csv"
        self.paths["std_tbl"] = os.path.join(redux_dir, std_tbl_name)
        std_list_name = "std_objects.list"
        self.paths["std_list"] = os.path.join(redux_dir, std_list_name)
        std_list = dragons.data_select(
            redux_dir=redux_dir,
            directory=raw_dir,
            expression=f"observation_class==\"partnerCal\"",
            output=std_list_name
        ).splitlines(False)[3:]

        std_tbl = dragons.showd(
            input_filenames=std_list,
            descriptors="object",
            output=std_tbl_name,
            csv=True,
            working_dir=redux_dir
        )

        # Set up dictionary of standard objects
        # TODO: ACCOUNT FOR MULTIPLE FILTERS.
        for std in std_tbl:
            if std["object"] not in self.std_objects:
                self.std_objects[std["object"]] = None

        for obj in self.std_objects:
            # And get the individual objects imaged like so:
            std_list_obj_name = f"std_{obj}.list"
            std_list_obj = dragons.data_select(
                redux_dir=redux_dir,
                directory=raw_dir,
                expression=f"object==\"{obj}\"",
                output=std_list_obj_name
            ).splitlines(False)[3:]
            self.std_objects[obj] = std_list_obj
            self.std_lists[obj] = os.path.join(redux_dir, std_list_obj_name)

    def proc_reduce_flats(self, output_dir: str, **kwargs):
        for fil in self.flats_lists:
            dragons.reduce(self.flats_lists[fil], redux_dir=self.paths["redux_dir"])
        flat_dir = os.path.join(self.paths["redux_dir"], "calibrations", "processed_flat")
        for flat in os.listdir(flat_dir):
            flat = os.path.join(flat_dir, flat)
            if not self.quiet:
                print(f"Adding {flat} to database.")
            sys_str = f"caldb add {flat}"
            if not self.quiet:
                print(sys_str)
            os.system(sys_str)

    def proc_reduce_science(self, output_dir: str, **kwargs):
        dragons.reduce(self.paths["science_list"], redux_dir=self.paths["redux_dir"])

    def proc_stack_science(self, output_dir: str, **kwargs):
        for fil in self.filters:
            dragons.disco(
                redux_dir=self.paths["redux_dir"],
                expression=f"filter_name==\"{fil}\" and observation_class==\"science\"",
                output=f"{self.name}_{fil}_stacked.fits",
                file_glob="*_sky*ed.fits",
                # refcat=self.field.paths["cat_csv_gaia"],
                # refcat_format="ascii.csv",
                # refcat_ra="ra",
                # refcat_dec="dec",
                # ignore_objcat=False
            )

    def check_filter(self, fil: str):
        not_none = super().check_filter(fil)
        if not_none:
            self.flats_lists[fil] = None
        return not_none

    def _output_dict(self):
        output_dict = super()._output_dict()
        output_dict.update({
            "flats_lists": self.flats_lists,
            "std_lists": self.std_lists
        })
        return output_dict

    def load_output_file(self, **kwargs):
        outputs = super().load_output_file(**kwargs)
        if type(outputs) is dict:
            if "flats_list" in outputs:
                self.flats_lists = outputs["flats_lists"]
            if "std" in outputs:
                self.std_lists = outputs["std"]
            if "flats" in outputs:
                self.frames_flat = outputs["flats"]
        return outputs

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        # default_params.update({})
        return default_params

    @classmethod
    def from_file(
            cls,
            param_file: Union[str, dict],
            name: str = None,
            field: 'fld.Field' = None
    ):

        name, param_file, param_dict = p.params_init(param_file)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")

        if field is None:
            field = param_dict.pop("field")

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

        print(f"Loading field {field}...")
        u.debug_print(2, f"GSAOIImagingEpoch.from_file(): {param_dict=}")

        return cls(
            name=name,
            field=field,
            param_path=param_file,
            data_path=os.path.join(p.config["top_data_dir"], param_dict.pop('data_path')),
            instrument='gs-aoi',
            program_id=param_dict.pop('program_id'),
            date=param_dict.pop('date'),
            target=target,
            source_extractor_config=param_dict.pop('sextractor'),
            **param_dict
        )

    @classmethod
    def sort_files(cls, input_dir: str, output_dir: str = None, tolerance: units.Quantity = 3 * units.arcmin):
        """
        A routine to sort through a directory containing an arbitrary number of GSAOI observations and assign epochs to
        them.

        :param input_dir:
        :param tolerance: Images will be grouped if they are < tolerance from each other (or, specifically, from the
        first encountered in that cluster).
        :return:
        """
        pointings = {}
        if output_dir is None:
            output_dir = input_dir
        u.mkdir_check(output_dir)
        files = os.listdir(input_dir)
        files.sort()
        for file in filter(lambda f: f.endswith(".fits"), files):
            # Since GSAOI science files cannot be relied upon to include the object/target in the header, we group
            # images by RA and Dec.
            path = os.path.join(input_dir, file)
            img = image.from_path(
                path,
                cls=image.GSAOIImage
            )
            pointing = img.extract_pointing()
            associated = False
            for pointing_str in pointings:
                pointings_list = pointings[pointing_str]
                other_pointing = pointings_list[0]
                if pointing.separation(other_pointing) <= tolerance:
                    pointings_list.append(pointing)
                    associated = True
                    shutil.move(path, pointing_str)
                    break

            if not associated:
                pointing_str = os.path.join(output_dir, pointing.to_string())
                u.mkdir_check(pointing_str)
                pointings[pointing_str] = [pointing]
                shutil.move(path, pointing_str)

        return pointings
