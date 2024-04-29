import os
import math
from typing import Union

import astropy.units as units
from astropy.coordinates import SkyCoord

import craftutils.utils as u
import craftutils.fits_files as ff

from .image import ImagingImage
from ..image import noise_read_unit, detect_instrument


# __all__ = []


# @u.export
class CoaddedImage(ImagingImage):

    def __init__(
            self,
            path: str,
            frame_type: str = None,
            instrument_name: str = None,
    ):
        super().__init__(
            path=path,
            frame_type=frame_type,
            instrument_name=instrument_name,
            load_outputs=False
        )

        self.area_file = None

        self.load_output_file()

        if self.area_file is None:
            self.area_file = self.path.replace(".fits", "_area.fits")

    def trim(
            self,
            left: Union[int, units.Quantity] = None,
            right: Union[int, units.Quantity] = None,
            bottom: Union[int, units.Quantity] = None,
            top: Union[int, units.Quantity] = None,
            output_path: str = None
    ):
        # Let the super method take care of this image
        trimmed = super().trim(left=left, right=right, bottom=bottom, top=top, output_path=output_path)
        # Trim the area file in the same way.
        if output_path is None:
            output_path = trimmed.path
        new_area_path = output_path.replace(".fits", "_area.fits")

        if os.path.isfile(self.area_file):
            ff.trim_file(
                path=self.area_file,
                left=left, right=right, bottom=bottom, top=top,
                new_path=new_area_path
            )
            trimmed.area_file = new_area_path
            trimmed.add_log(
                action=f"Trimmed area file to margins left={left}, right={right}, bottom={bottom}, top={top}",
                method=self.trim,
                output_path=new_area_path
            )
            trimmed.update_output_file()
        return trimmed

    def trim_from_area(self, output_path: str = None):
        left, right, bottom, top = ff.detect_edges_area(self.area_file)
        trimmed = self.trim(left=left, right=right, bottom=bottom, top=top, output_path=output_path)
        return trimmed

    def register(
            self,
            target: 'ImagingImage',
            output_path: str,
            ext: int = 0,
            trim: bool = True,
            **kwargs
    ):
        new_img = super().register(
            target=target,
            output_path=output_path,
            ext=ext,
            trim=trim,
            **kwargs
        )
        import reproject as rp
        area = new_img.copy(new_img.path.replace(".fits", "_area.fits"))
        area.load_data()
        reprojected, footprint = rp.reproject_exact(self.area_file, new_img.headers[ext], parallel=True)
        area.data[ext] = reprojected * area.data[ext].unit
        area.area_file = None
        area.write_fits_file()

        new_img.area_file = area.path
        return new_img

    def _output_dict(self):
        outputs = super()._output_dict()
        outputs.update({
            "area_file": self.area_file
        })
        return outputs

    def load_output_file(self):
        outputs = super().load_output_file()
        if outputs is not None:
            if "area_file" in outputs:
                self.area_file = outputs["area_file"]
        return outputs

    def correct_astrometry(self, output_dir: str = None, tweak: bool = True, **kwargs):
        new_image = super().correct_astrometry(
            output_dir=output_dir,
            tweak=tweak,
            **kwargs
        )
        if new_image is not None:
            new_image.area_file = self.area_file
            new_image.update_output_file()
        return new_image

    def copy(self, destination: str, suffix: str = ""):
        new_image = super().copy(destination, suffix=suffix)
        new_image.area_file = self.area_file
        new_image.update_output_file()
        return new_image

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "ncombine": "NCOMBINE"
        })
        return header_keys

    def extract_ncombine(self):
        key = self.header_keys()["ncombine"]
        return self.extract_header_item(key)


class F4CoaddedImage(CoaddedImage):
    def zeropoint(self, **kwargs):
        self.zeropoint_best = self.add_zeropoint(
            catalogue=self.extract_header_item("ZPTFILE"),
            zeropoint=self.extract_header_item("ZPTMAG") * units.mag,
            zeropoint_err=self.extract_header_item("ZPTMUCER") * units.mag,
            extinction=0.0 * units.mag,
            extinction_err=0.0 * units.mag,
            airmass=0.0,
            airmass_err=0.0
        )
        return self.zeropoint_best


class GMOSCoaddedImage(CoaddedImage):
    instrument_name = "gs-gmos"

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "filter": "FILTER2",
            "noise_read": "RDNOISE"
        })
        return header_keys


class GSAOIImage(CoaddedImage):
    instrument_name = "gs-aoi"

    def extract_pixel_scale(self, ext: int = 1, force: bool = False):
        return super().extract_pixel_scale(ext=ext, force=force)

    def extract_pointing(self):
        # GSAOI images keep the WCS information in the second HDU header.
        key = self.header_keys()["ra"]
        ra = self.extract_header_item(key, 1)
        key = self.header_keys()["dec"]
        dec = self.extract_header_item(key, 1)
        self.pointing = SkyCoord(ra, dec, unit=units.deg)
        return self.pointing


class HubbleImage(CoaddedImage):
    instrument_name = "hst-dummy"

    def __init__(
            self,
            path: str,
            frame_type: str = None,
            instrument_name: str = None
    ):
        if instrument_name is None:
            instrument_name = detect_instrument(path)
        super().__init__(
            path=path,
            frame_type=frame_type,
            instrument_name=instrument_name,
        )

    def extract_exposure_time(self):
        self.exposure_time = 1.0 * units.second
        return self.exposure_time

    def extract_noise_read(self):
        self.noise_read = 0.0 * noise_read_unit
        return self.noise_read

    def zeropoint(self, **kwargs):
        """
        Returns the AB magnitude zeropoint of the image, according to
        https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
        :return:
        """
        photflam = self.extract_header_item("PHOTFLAM")
        photplam = self.extract_header_item("PHOTPLAM")

        zeropoint = (-2.5 * math.log10(photflam) - 5 * math.log10(photplam) - 2.408) * units.mag

        self.add_zeropoint(
            catalogue="calib_pipeline",
            zeropoint=zeropoint,
            zeropoint_err=0.0 * units.mag,
            extinction=0.0 * units.mag,
            extinction_err=0.0 * units.mag,
            airmass=0.0,
            airmass_err=0.0,
            image_name="self"
        )
        self.select_zeropoint(True)
        self.update_output_file()
        self.add_log(
            action=f"Calculated zeropoint {zeropoint} from PHOTFLAM and PHOTPLAM header keys.",
            method=self.trim,
        )
        self.update_output_file()
        return self.zeropoint_best

    def mask_nearby(self):
        if self.instrument_name == "hst-wfc3_uvis2":
            mask = False
        else:
            mask = True
        u.debug_print(2, "mask_nearby", self.instrument_name, mask, type(self))
        return mask

    def detection_threshold(self):
        if self.instrument_name == "hst-wfc3_uvis2":
            thresh = 5.
        else:
            thresh = 5.
        return thresh

    def do_subtract_background(self):
        return False

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "gain": "CCDGAIN",
            "mjd-obs": "EXPSTART"
        })
        return header_keys


def _set_class_dict():
    from .__init__ import (
        CoaddedImage,
        DESCutout,
        GSAOIImage,
        HubbleImage,
        PanSTARRS1Cutout,
        FORS2CoaddedImage,
        HAWKICoaddedImage,
        WISECutout
    )

    CoaddedImage.class_dict = {
        "none": CoaddedImage,
        "decam": DESCutout,
        "gs-aoi": GSAOIImage,
        "hst-wfc3_uvis2": HubbleImage,
        "hst-wfc3_ir": HubbleImage,
        "panstarrs1": PanSTARRS1Cutout,
        "vlt-fors2": FORS2CoaddedImage,
        "vlt-hawki": HAWKICoaddedImage,
        "wise": WISECutout
    }


_set_class_dict()
