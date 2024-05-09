import os
from typing import Union

import astropy.units as units
from astropy.time import Time

import craftutils.params as p
import craftutils.observation.image as image
from .epoch import ImagingEpoch


class HubbleImagingEpoch(ImagingEpoch):
    instrument_name = "hst-dummy"
    coadded_class = image.HubbleImage

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
        super().__init__(
            name=name,
            field=field,
            param_path=param_path,
            data_path=data_path,
            instrument=instrument,
            date=date,
            program_id=program_id,
            target=target,
            standard_epochs=standard_epochs, source_extractor_config=source_extractor_config)

        self.load_output_file(mode="imaging")

    @classmethod
    def stages(cls):
        super_stages = super().stages()
        stages = {
            "download": super_stages["download"],
            "initial_setup": super_stages["initial_setup"],
            "source_extraction": super_stages["source_extraction"],
            "photometric_calibration": super_stages["photometric_calibration"],
            "get_photometry": super_stages["get_photometry"]
        }
        return stages

    def _pipeline_init(self, skip_cats: bool = False):
        super()._pipeline_init(skip_cats=skip_cats)
        self.coadded_final = "coadded"
        self.paths["download"] = os.path.join(self.data_path, "0-download")

    def _initial_setup(self, output_dir: str, **kwargs):
        download_dir = self.paths["download"]
        # for file in filter(lambda f: f.endswith(".fits"), os.listdir(self.data_path)):
        #     shutil.move(os.path.join(self.data_path, file), output_dir)
        for file in filter(lambda f: f.endswith(".fits"), os.listdir(download_dir)):
            path = os.path.join(download_dir, file)
            img = image.from_path(
                path,
                cls=image.HubbleImage
            )
            if self.instrument_name in [None, "hst-dummy"]:
                self.instrument_name = img.instrument_name
            fil = img.extract_filter()
            img.extract_date_obs()
            self.set_date(img.date)
            self.exp_time_mean[fil] = img.extract_header_item('TEXPTIME') * units.second / img.extract_ncombine()
            img.set_header_item('INTTIME', img.extract_header_item('TEXPTIME'))
            self.add_coadded_image(img, key=fil)
            self.add_coadded_unprojected_image(img, key=fil)
            self.check_filter(img.filter_name)

    def photometric_calibration(
            self,
            image_dict: dict,
            output_path: str,
            **kwargs):

        for fil in image_dict:
            image_dict[fil].zeropoint()
            image_dict[fil].estimate_depth()
            self.deepest = image_dict[fil]

    def proc_get_photometry(self, output_dir: str, **kwargs):
        self.get_photometry(output_dir, image_type="coadded", dual=False, **kwargs)

    def psf_diagnostics(
            self,
            images: dict = None
    ):
        if images is None:
            images = self._get_images("final")

        for fil in images:
            img = images[fil]
            if fil == "F300X":
                self.psf_stats[fil] = {
                    "n_stars": 0,
                    "fwhm_psfex": -999 * units.arcsec,
                    "gauss": {
                        "fwhm_median": -999 * units.arcsec,
                        "fwhm_mean": -999 * units.arcsec,
                        "fwhm_max": -999 * units.arcsec,
                        "fwhm_min": -999 * units.arcsec,
                        "fwhm_sigma": -999 * units.arcsec,
                        "fwhm_rms": -999 * units.arcsec
                    },
                    "moffat": {
                        "fwhm_median": -999 * units.arcsec,
                        "fwhm_mean": -999 * units.arcsec,
                        "fwhm_max": -999 * units.arcsec,
                        "fwhm_min": -999 * units.arcsec,
                        "fwhm_sigma": -999 * units.arcsec,
                        "fwhm_rms": -999 * units.arcsec
                    },
                    "sextractor": {
                        "fwhm_median": -999 * units.arcsec,
                        "fwhm_mean": -999 * units.arcsec,
                        "fwhm_max": -999 * units.arcsec,
                        "fwhm_min": -999 * units.arcsec,
                        "fwhm_sigma": -999 * units.arcsec,
                        "fwhm_rms": -999 * units.arcsec
                    }
                }
                img.set_header_items(
                    {
                        "PSF_FWHM": -999,
                        "PSF_FWHM_ERR": -999,
                    },
                    write=True
                )
            else:
                if not self.quiet:
                    print(f"Performing PSF measurements on {img}...")
                self.psf_stats[fil], _ = img.psf_diagnostics()

        self.update_output_file()
        return self.psf_stats

    def add_coadded_image(self, img: Union[str, image.Image], key: str, **kwargs):
        try:
            if isinstance(img, str):
                img = image.from_path(
                    img,
                    cls=image.HubbleImage
                )
            img.epoch = self
            self.coadded[key] = img
            return img
        except FileNotFoundError:
            return None

    def n_frames(self, fil: str):
        return self.coadded[fil].extract_ncombine()

    @classmethod
    def from_file(
            cls,
            param_file: Union[str, dict],
            name: str = None,
            field: 'craftutils.observation.field.Field' = None
    ):

        name, param_file, param_dict = p.params_init(param_file)
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
        if "name" in param_dict:
            param_dict.pop("name")
        if "param_path" in param_dict:
            param_dict.pop("param_path")

        return cls(
            name=name,
            field=field,
            param_path=param_file,
            data_path=os.path.join(p.config["top_data_dir"], param_dict.pop('data_path')),
            instrument=param_dict.pop("instrument"),
            program_id=param_dict.pop('program_id'),
            date=param_dict.pop('date'),
            target=target,
            source_extractor_config=param_dict.pop('sextractor'),
            **param_dict
        )
