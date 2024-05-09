import os
import shutil
from typing import Union, List, Dict

import astropy.io.fits as fits

import craftutils.observation.image as image
import craftutils.utils as u
import craftutils.params as p

from ..epoch import ImagingEpoch
from .epoch import ESOImagingEpoch


class HAWKIImagingEpoch(ESOImagingEpoch):
    instrument_name = "vlt-hawki"
    frame_class = image.HAWKIImage
    coadded_class = image.HAWKICoaddedImage
    eso_name = "HAWKI"

    def __init__(
            self,
            **kwargs
    ):
        self.coadded_esoreflex = {}
        self.frames_split = {}
        super().__init__(**kwargs)

    def n_frames(self, fil: str):
        return self.coadded_astrometry[fil].extract_ncombine()

    @classmethod
    def stages(cls):
        eso_stages = super().stages()
        ie_stages = ImagingEpoch.stages()
        stages = {
            "download": eso_stages["download"],
            "initial_setup": eso_stages["initial_setup"],
            "sort_reduced": eso_stages["sort_reduced"],
            "split_frames": {
                "method": cls.proc_split_frames,
                "message": "Split ESO Reflex frames into separate files?",
                "log_message": "Split ESO Reflex frames into separate .fits files",
                "default": True,
            },
            "coadd": ie_stages["coadd"],
            "correct_astrometry_coadded": ie_stages["correct_astrometry_coadded"],
            "source_extraction": ie_stages["source_extraction"],
            "photometric_calibration": ie_stages["photometric_calibration"],
            "finalise": ie_stages["finalise"],
            "get_photometry": ie_stages["get_photometry"]
        }
        stages["coadd"]["default"] = False
        stages["coadd"]["frames"] = "split"
        stages["correct_astrometry_coadded"]["default"] = True
        return stages

    def add_coadded_esoreflex_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_esoreflex)

    def add_frame_split(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_split, frame_type="reduced")

    def check_filter(self, fil: str):
        not_none = super().check_filter(fil)
        if not_none:
            if fil not in self.frames_split:
                if isinstance(self.frames_split, dict):
                    self.frames_split[fil] = []
            if fil not in self.coadded_esoreflex:
                self.coadded_esoreflex[fil] = None
        return not_none

    def _output_dict(self):
        output_dict = super()._output_dict()
        from ...epoch import _output_img_dict_single, _output_img_dict_list
        output_dict.update({
            "coadded_esoreflex": _output_img_dict_single(self.coadded_esoreflex),
            "frames_split": _output_img_dict_list(self.frames_split)
        })

        return output_dict

    def load_output_file(self, **kwargs):
        outputs = super().load_output_file(**kwargs)
        if isinstance(outputs, dict):
            if "coadded_esoreflex" in outputs:
                for fil in outputs["coadded_esoreflex"]:
                    if outputs["coadded_esoreflex"][fil] is not None:
                        u.debug_print(1, f"Attempting to load coadded_esoreflex[{fil}]")
                        self.add_coadded_esoreflex_image(img=outputs["coadded_esoreflex"][fil], key=fil, **kwargs)
            if "frames_split" in outputs:
                for fil in outputs["frames_split"]:
                    if outputs["frames_split"][fil] is not None:
                        for frame in outputs["frames_split"][fil]:
                            self.add_frame_split(frame=frame)

    def _pipeline_init(self, skip_cats: bool = False):
        super()._pipeline_init(skip_cats=skip_cats)
        self.coadded_final = "coadded_astrometry"
        self.frames_final = "frames_split"

    def sort_after_esoreflex(self, output_dir: str, **kwargs):
        """
        Scans through the ESO Reflex directory for the files matching this epoch, and puts them where we want them.
        :param output_dir:
        :param kwargs:
        :return:
        """
        self.frames_reduced = {}
        self.coadded_esoreflex = {}

        super().sort_after_esoreflex(
            output_dir=output_dir,
            **kwargs
        )

        esodir_root = p.config['esoreflex_output_dir']

        eso_tmp_dir = os.path.join(
            esodir_root,
            "reflex_tmp_products",
            "hawki",
            "hawki_science_process_1"
        )

        tmp_subdirs = os.listdir(eso_tmp_dir)
        mjd = int(self.mjd())
        obj = self.target.lower()

        # Also grab the intermediate, individual chip frames from the reflex temp products directory
        for subdir in tmp_subdirs:
            subpath = os.path.join(eso_tmp_dir, subdir)
            if os.path.isfile(os.path.join(subpath, "exp_1.fits")):
                with fits.open(os.path.join(subpath, "exp_1.fits")) as file:
                    if "OBJECT" in file[0].header:
                        file_obj = file[0].header["OBJECT"].lower()
                    else:
                        continue
                    if "MJD-OBS" in file[0].header:
                        file_mjd = int(file[0].header["MJD-OBS"])
                    else:
                        continue
                    if "FILTER" in file[0].header:
                        fil = file[0].header["FILTER"]
                if file_obj == obj and file_mjd == mjd:
                    fil_destination = os.path.join(
                        output_dir,
                        fil,
                        "frames",
                    )
                    u.mkdir_check(fil_destination)
                    i = 1
                    while os.path.isfile(os.path.join(subpath, f"exp_{i}.fits")):
                        file_path = os.path.join(subpath, f"exp_{i}.fits")
                        new_file_name = f"{self.name}_{self.date_str()}_{fil}_exp_{i}.fits"
                        file_destination = os.path.join(
                            fil_destination,
                            new_file_name
                        )
                        if not self.quiet:
                            print(f"Copying: {file_path} \n\tto \n\t {file_destination}")
                        shutil.copy(file_path, file_destination)
                        img = image.HAWKIImage(path=file_path, frame_type="science")
                        self.add_frame_reduced(img)
                        i += 1

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
        files = filter(
            lambda f: os.path.isfile(os.path.join(subpath, f)) and f.endswith(".fits"),
            os.listdir(subpath)
        )
        good_dir = False
        for file_name in files:
            file_path = os.path.join(subpath, file_name)
            with fits.open(file_path) as file:
                if "OBJECT" in file[0].header:
                    file_obj = file[0].header["OBJECT"].lower()
                else:
                    continue
                if "MJD-OBS" in file[0].header:
                    file_mjd = int(file[0].header["MJD-OBS"])
                else:
                    continue
                if "FILTER" in file[0].header:
                    fil = file[0].header["FILTER"]
            if file_obj == obj and file_mjd == mjd:
                suffix = file_name[file_name.find("_") + 1:-5]
                new_file_name = f"{self.name}_{self.date_str()}_{fil}_{suffix}.fits"
                fil_destination = os.path.join(
                    output_dir,
                    fil
                )
                u.mkdir_check(fil_destination)
                file_destination = os.path.join(
                    fil_destination,
                    new_file_name
                )
                if not self.quiet:
                    print(f"Copying: {file_path} \n\tto \n\t {file_destination}")
                shutil.copy(file_path, file_destination)
                if file_name.endswith("TILED_IMAGE.fits"):
                    img = self.add_coadded_esoreflex_image(
                        img=file_destination,
                        key=fil
                    )
                    img.set_header_items({
                        "EXPTIME": 1.0,
                        "INTIME": img.extract_header_item("TEXPTIME"),
                    })
                if delete_output and os.path.isfile(file_destination):
                    os.remove(file_path)

    def proc_split_frames(self, output_dir: str, **kwargs):
        self.split_frames(output_dir=output_dir, **kwargs)

    def split_frames(
            self,
            output_dir: str,
            **kwargs
    ):
        for fil in self.frames_reduced:
            for frame in self.frames_reduced[fil]:

                results = frame.split_fits(
                    output_dir=output_dir
                )
                if not self.quiet:
                    print(f"Split {frame} into:")
                for name in results:
                    if not self.quiet:
                        print(f"\t{name}")
                    self.add_frame_split(frame=results[name])

    def coadd(self, output_dir: str, frames: str = "split", sigma_clip: float = 1.5):
        return super().coadd(
            output_dir=output_dir,
            frames=frames,
            sigma_clip=sigma_clip
        )

    def correct_astrometry_coadded(
            self,
            output_dir: str,
            image_type: str = None,
            **kwargs
    ):
        if image_type is None:
            image_type = "coadded_esoreflex"
        super().correct_astrometry_coadded(
            output_dir=output_dir,
            image_type=image_type,
            **kwargs
        )
        if not self.coadded_astrometry:
            self.coadded_final = "coadded_esoreflex"
            self.coadded_unprojected = self.coadded_esoreflex.copy()
        else:
            self.coadded_unprojected = self.coadded_astrometry.copy()

    def _get_images(self, image_type: str) -> Dict[str, image.CoaddedImage]:
        if image_type in ("final", "coadded_final"):
            if self.coadded_final is not None:
                image_type = self.coadded_final
            else:
                raise ValueError("coadded_final has not been set.")

        if image_type in ("coadded_esoreflex", "esoreflex"):
            return self.coadded_esoreflex
        else:
            return super()._get_images(image_type=image_type)

    def _get_frames(self, frame_type: str) -> Dict[str, List[image.ImagingImage]]:
        if frame_type == "final":
            if self.frames_final is not None:
                frame_type = self.frames_final
            else:
                raise ValueError("frames_final has not been set.")

        if frame_type in ("split", "frames_split"):
            image_dict = self.frames_split
        else:
            image_dict = super()._get_frames(
                frame_type=frame_type
            )

        return image_dict
