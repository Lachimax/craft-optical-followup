import os

import astropy.units as units

import craftutils.utils as u
import craftutils.retrieve as retrieve
import craftutils.observation.image as image
from ...epoch import ImagingEpoch
from ....std import StandardEpoch


class FORS2StandardEpoch(StandardEpoch, ImagingEpoch):
    frame_class = image.FORS2Image
    coadded_class = image.FORS2CoaddedImage
    instrument_name = "vlt-fors2"

    def source_extraction(self, output_dir: str, do_diagnostics: bool = True, **kwargs):
        for fil in self.frames_reduced:
            for img in self.frames_reduced[fil]:
                img.remove_extra_extensions()
                configs = self.source_extractor_config

                img.psfex_path = None
                img.source_extraction_psf(
                    output_dir=output_dir,
                    phot_autoparams=f"3.5,1.0"
                )

    def reduce(self, output_dir: str = None):
        if output_dir is None:
            output_dir = self.data_path
        import craftutils.wrap.esorex as esorex
        u.mkdir_check_nested(output_dir, False)
        aligned_phots = {}
        for fil, stds in self.frames_standard.items():
            fil_dir = os.path.join(self.data_path, fil)
            u.mkdir_check(fil_dir)
            for std in stds:
                chip = std.extract_chip_number()
                chip_dir = os.path.join(fil_dir, f"chip_{chip}")
                u.mkdir_check(chip_dir)
                if chip not in aligned_phots:
                    aligned_phots[chip] = {}
                if fil not in aligned_phots[chip]:
                    aligned_phots[chip][fil] = []
                master_bias = self.get_master_bias(chip=chip)
                master_sky_flat_img = self.get_master_flat(chip=chip, fil=fil)
                print(f"\tReducing with ESORex")
                aligned_phot, std_reduced = esorex.fors_zeropoint(
                    standard_img=std.path,
                    master_bias=master_bias,
                    master_sky_flat_img=master_sky_flat_img,
                    output_dir=chip_dir,
                    chip_num=chip
                )
                print("\tReduced std is at:", std_reduced)
                aligned_phots[chip][fil].append(aligned_phot)
                self.add_frame_reduced(std_reduced)
        return aligned_phots

    def photometric_calibration(
            self,
            output_path: str = None,
            skip_retrievable: bool = False,
            **kwargs
    ):
        self.load_output_file()
        zeropoints = {}

        if output_path is None:
            output_path = os.path.join(self.data_path, "photometric_calibration")

        u.mkdir_check_nested(output_path)

        self.source_extraction(
            output_dir=output_path,
            do_diagnostics=False,
            skip_retrievable=skip_retrievable
        )

        self.zeropoint(
            image_dict=self.frames_reduced,
            output_path=output_path,
            suppress_select=True,
            zp_dict=zeropoints,
            skip_retrievable=skip_retrievable,
            **kwargs
        )
        self.update_output_file()

    def zeropoint(
            self,
            image_dict: dict,
            output_path: str,
            distance_tolerance: units.Quantity = None,
            snr_min: float = 3.,
            star_class_tolerance: float = 0.9,
            suppress_select: bool = True,
            **kwargs
    ):
        from craftutils.observation.filters import FORS2Filter

        if "zp_dict" in kwargs:
            zp_dict = kwargs["zp_dict"]
        else:
            zp_dict = {}

        skip_retrievable = False
        if "skip_retrievable" in kwargs:
            skip_retrievable = kwargs["skip_retrievable"]

        zp_dict[1] = {}
        zp_dict[2] = {}
        for fil in self.filters:
            if skip_retrievable and fil in FORS2Filter.qc1_retrievable:
                continue
            for img in image_dict[fil]:
                cats = retrieve.photometry_catalogues
                # cats.append("eso_calib_cats")
                for cat_name in cats:
                    if cat_name == "gaia":
                        continue
                    if cat_name in retrieve.cat_systems and retrieve.cat_systems[cat_name] == "vega":
                        vega = True
                    else:
                        vega = False
                    fil_path = os.path.join(output_path, fil)
                    u.mkdir_check_nested(fil_path, remove_last=False)
                    if f"in_{cat_name}" in self.field.cats and self.field.cats[f"in_{cat_name}"]:
                        zp = img.zeropoint(
                            cat=self.field.get_path(f"cat_csv_{cat_name}"),
                            output_path=os.path.join(fil_path, cat_name),
                            cat_name=cat_name,
                            dist_tol=distance_tolerance,
                            show=False,
                            snr_cut=snr_min,
                            star_class_tol=star_class_tolerance,
                            iterate_uncertainty=True,
                            vega=vega
                        )

                        chip = img.extract_chip_number()

                if "preferred_zeropoint" in kwargs and fil in kwargs["preferred_zeropoint"]:
                    preferred = kwargs["preferred_zeropoint"][fil]
                else:
                    preferred = None

                img.select_zeropoint(suppress_select, preferred=preferred)
