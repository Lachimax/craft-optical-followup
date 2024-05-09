import craftutils.observation.image as image
from .epoch import SurveyImagingEpoch

class DESEpoch(SurveyImagingEpoch):
    instrument_name = "decam"
    catalogue = "des"
    coadded_class = image.DESCutout

    def n_frames(self, fil: str):
        return 1

    def proc_split(self, output_dir: str, **kwargs):
        if "image_type" not in kwargs:
            kwargs["image_type"] = "coadded"
        self.split(output_dir=output_dir, **kwargs)

    def split(self, output_dir: str, image_type):
        image_dict = self._get_images(image_type=image_type)
        for fil in image_dict:
            img = image_dict[fil]
            split_imgs = img.split_fits(output_dir=output_dir)
            self.add_coadded_image(split_imgs["SCI"], key=fil)

    @classmethod
    def stages(cls):
        super_stages = super().stages()
        stages = {
            "download": super_stages["download"],
            "initial_setup": super_stages["initial_setup"],
            "split": {
                "method": cls.proc_split,
                "message": "Split fits files into components?",
                "default": True,
                "keywords": {}
            },
            "source_extraction": super_stages["source_extraction"],
            "photometric_calibration": super_stages["photometric_calibration"],
            # "dual_mode_source_extraction": super_stages["dual_mode_source_extraction"],
            "get_photometry": super_stages["get_photometry"]
        }
        return stages

