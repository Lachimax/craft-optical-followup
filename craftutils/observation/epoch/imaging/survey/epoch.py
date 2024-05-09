import os
from typing import Union

import astropy.units as units

import craftutils.utils as u
import craftutils.params as p
import craftutils.observation.image as image
from ..epoch import ImagingEpoch

class SurveyImagingEpoch(ImagingEpoch):
    mode = "imaging"
    catalogue = None
    coadded_class = image.SurveyCutout
    preferred_zeropoint = "calib_pipeline"

    def __init__(
            self,
            name: str = None,
            field: Union[str, 'fld.Field'] = None,
            param_path: str = None,
            data_path: str = None,
            source_extractor_config: dict = None,
            **kwargs
    ):
        super().__init__(
            name=name,
            field=field,
            param_path=param_path,
            data_path=data_path,
            source_extractor_config=source_extractor_config,
            instrument=self.instrument_name
        )
        self.load_output_file(mode="imaging")
        # if isinstance(field, Field):
        # self.field.retrieve_catalogue(cat_name=self.catalogue)

        u.debug_print(1, f"SurveyImagingEpoch.__init__(): {self}.filters ==", self.filters)

    @classmethod
    def stages(cls):
        super_stages = super().stages()
        super_stages["source_extraction"]["do_astrometry_diagnostics"] = False
        stages = {
            "download": super_stages["download"],
            "initial_setup": super_stages["initial_setup"],
            "source_extraction": super_stages["source_extraction"],
            "photometric_calibration": super_stages["photometric_calibration"],
            # "dual_mode_source_extraction": super_stages["dual_mode_source_extraction"],
            "get_photometry": super_stages["get_photometry"]
        }
        return stages

    def _pipeline_init(self, skip_cats: bool = False):
        super()._pipeline_init(skip_cats=skip_cats)
        self.coadded_final = "coadded"
        self.paths["download"] = os.path.join(self.data_path, "0-download")

    # TODO: Automatic cutout download; don't worry for now.
    def proc_download(self, output_dir: str, **kwargs):
        """
        Automatically download survey cutout.
        :param output_dir:
        :param kwargs:
        :return:
        """
        pass

    def proc_source_extraction(self, output_dir: str, **kwargs):
        self.source_extraction(
            output_dir=output_dir,
            do_astrometry_diagnostics=False,
            do_psf_diagnostics=True,
            **kwargs
        )

    def proc_get_photometry(self, output_dir: str, **kwargs):
        self.load_output_file()
        self.get_photometry(output_dir, image_type="coadded", **kwargs)

    def _initial_setup(self, output_dir: str, **kwargs):
        download_dir = self.paths["download"]
        # for file in filter(lambda f: f.endswith(".fits"), os.listdir("download")):
        #     shutil.move(os.path.join(self.data_path, file), output_dir)
        self.set_path("imaging_dir", download_dir)
        # Write a table of fits files from the 0-imaging directory.
        table_path_all = os.path.join(self.data_path, f"{self.name}_fits_table_all.csv")
        self.set_path("fits_table", table_path_all)
        image.fits_table_all(input_path=download_dir, output_path=table_path_all, science_only=False)
        for file in filter(lambda f: f.endswith(".fits"), os.listdir(download_dir)):
            path = os.path.join(download_dir, file)
            img = self.coadded_class(path=path)
            fil = img.extract_filter()
            u.debug_print(2, f"PanSTARRS1ImagingEpoch._initial_setup(): {fil=}")
            self.exp_time_mean[fil] = img.extract_exposure_time() / img.extract_ncombine()
            img.set_header_item('INTTIME', img.extract_integration_time())
            self.add_coadded_image(img, key=fil)
            self.check_filter(img.filter_name)
            img.write_fits_file()

    def guess_data_path(self):
        if self.data_path is None and self.field is not None and self.field.data_path is not None:
            self.data_path = os.path.join(self.field.data_path, "imaging", self.catalogue)
        return self.data_path

    def zeropoint(
            self,
            output_path: str,
            distance_tolerance: units.Quantity = 1 * units.arcsec,
            snr_min: float = 3.,
            star_class_tolerance: float = 0.95,
            **kwargs
    ):
        u.debug_print(2, f"", self.filters)
        deepest = None
        for fil in self.coadded:
            img = self.coadded[fil]
            zp = img.zeropoint(
                cat=self.field.get_path(f"cat_csv_{self.catalogue}"),
                output_path=os.path.join(output_path, img.name),
                cat_name=self.catalogue,
                dist_tol=distance_tolerance,
                show=False,
                snr_cut=snr_min,
                star_class_tol=star_class_tolerance,
                image_name=f"{self.catalogue}",
            )
            img.select_zeropoint(True, preferred=self.preferred_zeropoint)
            img.estimate_depth(zeropoint_name=self.catalogue)  # , do_magnitude_calibration=False)
            if deepest is not None:
                deepest = image.deepest(deepest, img)
            else:
                deepest = img

        return deepest

    def n_frames(self, fil: str):
        img = self.coadded[fil]
        return img.extract_ncombine()

    def add_coadded_image(self, img: Union[str, image.Image], key: str, **kwargs):
        if isinstance(img, str):
            if os.path.isfile(img):
                cls = self.coadded_class
                img = image.from_path(path=img, cls=cls)
            else:
                return None
        img.epoch = self
        self.coadded[key] = img
        self.coadded_unprojected[key] = img
        return img

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

        if "field" in param_dict:
            param_dict.pop("field")
        if "instrument" in param_dict:
            param_dict.pop("instrument")
        if "name" in param_dict:
            param_dict.pop("name")
        if "param_path" in param_dict:
            param_dict.pop("param_path")

        epoch = cls(
            name=name,
            field=field,
            param_path=param_file,
            data_path=os.path.join(p.config["top_data_dir"], param_dict.pop('data_path')),
            source_extractor_config=param_dict.pop('sextractor'),
            **param_dict
        )
        # epoch.instrument = cls.instrument_name
        return epoch
