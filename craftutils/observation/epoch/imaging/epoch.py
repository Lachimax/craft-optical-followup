# Code by Lachlan Marnoch, 2021 - 2024

import os
import shutil
from typing import Union, List, Dict

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as units
import astropy.table as table
from astropy.time import Time
from astropy.coordinates import SkyCoord

import ccdproc

import craftutils.observation.image as image
import craftutils.observation.field as fld
import craftutils.wrap.montage as montage
import craftutils.plotting as pl
import craftutils.retrieve as retrieve
import craftutils.utils as u
import craftutils.params as p
import craftutils.astrometry as astm
from craftutils.observation.instrument import Instrument
from craftutils.observation.survey import Survey
from ..epoch import Epoch, active_epochs


class ImagingEpoch(Epoch):
    instrument_name = "dummy-instrument"
    mode = "imaging"
    frame_class = image.ImagingImage
    coadded_class = image.CoaddedImage
    frames_for_combined = "astrometry"
    skip_for_combined = [
        "download",
        "initial_setup",
        "sort_reduced",
        "trim_reduced",
        "convert_to_cs",
        "correct_astrometry_frames"
    ]
    validation_stages = [
        "insert_synthetic_frames"
    ]

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
            standard_epochs: list = None,
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
            **kwargs
        )
        self.guess_data_path()
        self.source_extractor_config = source_extractor_config
        if self.source_extractor_config is None:
            self.source_extractor_config = {
                "dual_mode": False,
                "threshold": 1.5,
                "kron_factor": 3.5,
                "kron_radius_min": 1.0
            }

        self.filters = []
        self.deepest = None
        self.deepest_filter = None

        self.exp_time_mean = {}
        self.exp_time_err = {}
        self.airmass_mean = {}
        self.airmass_err = {}

        self.frames_standard_extra = {}
        self.frames_science = {}
        self.frames_reduced = {}
        self.frames_trimmed = {}
        self.frames_subtracted = {}
        self.frames_normalised = {}
        self.frames_registered = {}
        self.frames_astrometry = {}
        self.astrometry_successful = {}
        self.frames_diagnosed = {}
        self.frames_final = None

        self.std_pointings = []
        self.std_objects = {}
        self.std_epochs = {}

        self.coadded_trimmed = {}
        self.coadded_unprojected = {}
        self.coadded_astrometry = {}
        self.coadded_subtracted = {}
        self.coadded_subtracted_trimmed = {}
        self.coadded_subtracted_patch = {}
        self.coadded_final = None
        self.coadded_derivatives = (
            self.coadded_unprojected,
            self.coadded_subtracted,
            self.coadded_subtracted_trimmed,
            self.coadded_subtracted_patch
        )

        self.gaia_catalogue = None
        self.astrometry_indices = []

        self.frame_stats = {}
        self.astrometry_stats = {}
        self.psf_stats = {}

        self.validation_copy_of = None
        if "validation_copy_of" in kwargs:
            self.validation_copy_of = kwargs["validation_copy_of"]
            if isinstance(self.validation_copy_of, str):
                self.validation_copy_of = self.from_params(
                    name=self.validation_copy_of,
                    instrument=self.instrument_name,
                    field=self.field,
                    quiet=self.quiet
                )
        self.validation_catalogue_path = None
        self.validation_catalogue = None

        # self.load_output_file(mode="imaging")

    def _pipeline_init(self, skip_cats: bool = False):
        super()._pipeline_init(skip_cats=skip_cats)
        for fil in self.filters:
            self.check_filter(fil)

    @classmethod
    def stages(cls):

        stages = super().stages()
        stages.update({
            "download": {
                "method": cls.proc_download,
                "message": "Pretend to download files? (download not actualy implemented for this class)",
                "default": False,
                "keywords": {
                    "alternate_dir": None
                }
            },
            "insert_synthetic_frames": {
                "method": cls.proc_insert_synthetic_frames,
                "message": "Insert synthetic sources in frames for validation?",
                "default": False,
                "keywords": {}
            },
            "defringe": {
                "method": cls.proc_defringe,
                "message": "Defringe frames?",
                "default": False,
                "keywords": {}
            },
            "register_frames": {
                "method": cls.proc_register,
                "message": "Register frames using astroalign?",
                "default": False,
                "keywords": {
                    "template": 0,
                    "include_chips": "all"
                }
            },
            "correct_astrometry_frames": {
                "method": cls.proc_correct_astrometry_frames,
                "message": "Correct astrometry of individual frames?",
                "default": True,
                "keywords": {
                    "tweak": True,
                    "upper_only": False,
                    "method": "individual",
                    "back_subbed": False,
                    "correct_to_epoch": True,
                    "registration_template": None,
                    "odds_to_tune_up": None,
                    "odds-to-solve": None
                }
            },
            "frame_diagnostics": {
                "method": cls.proc_frame_diagnostics,
                "message": "Run diagnostics on individual frames?",
                "default": False,
            },
            "subtract_background_frames": {
                "method": cls.proc_subtract_background_frames,
                "message": "Subtract local background from frames?",
                "default": False,
                "keywords": {
                    "do_not_mask": False,
                    "centre": None,
                    "frame": 15 * units.arcsec,
                    "frames": "astrometry",
                    "mask_kwargs": {},
                    "method": "local",
                    # "polynomial_degree": 3
                }
            },
            "coadd": {
                "method": cls.proc_coadd,
                "message": "Coadd frames with Montage?",
                "default": True,
                "keywords": {
                    "frames": "final",  # normalised, trimmed
                    "sigma_clip": 1.0
                }
            },
            "correct_astrometry_coadded": {
                "method": cls.proc_correct_astrometry_coadded,
                "message": "Correct astrometry of coadded images?",
                "default": False,
                "keywords": {
                    "tweak": True,
                    "astroalign_template": None,
                }
            },
            "trim_coadded": {
                "method": cls.proc_trim_coadded,
                "message": "Trim / reproject coadded images to same footprint?",
                "default": True,
                "keywords": {
                    "reproject": False  # Reproject to same footprint?
                }
            },
            "source_extraction": {
                "method": cls.proc_source_extraction,
                "message": "Do source extraction and diagnostics?",
                "default": True,
                "keywords": {
                    "do_astrometry_diagnostics": True
                }
            },
            "photometric_calibration": {
                "method": cls.proc_photometric_calibration,
                "message": "Do photometric calibration?",
                "default": True,
                "keywords": {
                    "distance_tolerance": None,
                    "snr_min": 3.,
                    "class_star_tolerance": 0.95,
                    "image_type": "final",
                    "preferred_zeropoint": {},
                    "suppress_select": True
                },
            },
            "finalise": {
                "method": cls.proc_finalise,
                "message": "Finalise science files?",
                "default": True,
                "keywords": {},
            },
            "dual_mode_source_extraction": {
                "method": cls.proc_dual_mode_source_extraction,
                "message": "Do source extraction in dual-mode, using deepest image as footprint?",
                "default": False,
            },
            "get_photometry": {
                "method": cls.proc_get_photometry,
                "message": "Get photometry?",
                "default": True,
                "keywords": {
                    "image_type": "final",
                    "skip_plots": False,
                    "skip_path": False
                }
            },
            # "get_photometry_all": {
            #     "method": cls.proc_get_photometry_all,
            #     "message": "Get all photometry?",
            #     "default": True
            # }
        }
        )
        return stages

    def n_frames(self, fil: str):
        return len(self.frames_reduced[fil])

    def proc_download(self, output_dir: str, **kwargs):
        pass

    def proc_insert_synthetic_frames(
            self,
            output_dir: str,
            **kwargs
    ):
        self.insert_synthetic_frames(frame_type=self.frames_for_combined, **kwargs)

    def generate_validation_catalogue(self, force=True, n: int = 100):
        if force or self.validation_catalogue_path is None:
            pass
        else:
            self.validation_catalogue = table.QTable.read(self.validation_catalogue_path)

    def insert_synthetic_frames(self, frame_type: str, **kwargs):
        frames_original = self.validation_copy_of._get_frames(frame_type)
        for frame in frames_original:
            pass
        self.generate_validation_catalogue()


    def proc_defringe(
            self,
            output_dir: str,
            **kwargs
    ):
        for fil in self.filters:
            self.generate_fringe_map(
                fil=fil,
                output_dir=output_dir,
                **kwargs
            )

    def generate_master_biases(self, output_dir: str = None, force: bool = False):
        """
        This does nothing, but will eventually do what the function name says.

        :return:
        """

    def generate_master_flats(self, output_dir: str = None, force: bool = False):
        """
        This does nothing, but will eventually do what the function name says.

        :return:
        """

    def generate_fringe_map(
            self,
            fil: str,
            output_dir: str = None,
            force: bool = False,
            frames: Union[dict, list] = None
    ):
        """

        :return:
        """

        self.check_filter(fil)
        if frames is None:
            frames = self._get_frames(frame_type="frames_normalised")[fil]
        elif isinstance(frames, dict):
            frames = frames[fil]
        frames_chip = self.sort_by_chip(frames)

        self.generate_master_biases()
        self.generate_master_flats()

        for chip, frames in frames_chip.items():
            frame_paths = list(map(lambda f: f.path, frames))
            self.fringe_maps[fil][chip] = {}
            path_mean = os.path.join(
                output_dir,
                f"fringemap_{self.name}_{fil}_chip-{chip}_mean.fits"
            )
            combined_mean = ccdproc.combine(
                img_list=frame_paths,
                method="average",
                sigma_clip=True,
                output_file=path_mean
            )
            self.fringe_maps[fil][chip]["mean"] = path_mean

            path_median = os.path.join(
                output_dir,
                f"fringemap_{self.name}_{fil}_chip-{chip}_median.fits"
            )
            combined_median = ccdproc.combine(
                img_list=frame_paths,
                method="median",
                sigma_clip=True,
                output_file=path_median
            )
            self.fringe_maps[fil][chip]["median"] = path_median

        return self.fringe_maps

    def proc_subtract_background_frames(self, output_dir: str, **kwargs):
        self.frames_subtracted = {}
        if "frames" not in kwargs:
            if "correct_astrometry_frames" in self.do_param and self.do_param["correct_astrometry_frames"]:
                kwargs["frames"] = "astrometry"
            else:
                kwargs["frames"] = "normalised"
        # if "polynomial_degree" in kwargs:
        #     kwargs["init_params"]["degree"]
        self.subtract_background_frames(
            output_dir=output_dir,
            **kwargs
        )

    def subtract_background_frames(
            self,
            output_dir: str,
            frames: Union[dict, str] = None,
            method: str = "local",
            **kwargs
    ):
        if isinstance(frames, str):
            frames = self._get_frames(frames)

        if "do_not_mask" in kwargs:
            do_not_mask = kwargs.pop("do_not_mask")
            for i, obj in enumerate(do_not_mask):
                if isinstance(obj, str):
                    if obj in self.field.objects:
                        obj = self.field.objects[obj].position
                    elif not isinstance(obj, SkyCoord):
                        obj = astm.attempt_skycoord(obj)
                do_not_mask[i] = obj
            if "mask_kwargs" not in kwargs:
                kwargs["mask_kwargs"] = {}
            kwargs["mask_kwargs"]["do_not_mask"] = do_not_mask

        fit_info = {}
        for fil in frames:
            frame_list = frames[fil]
            fit_info[fil] = {}
            for frame in frame_list:
                if self.is_excluded(frame):
                    continue
                subbed_path = os.path.join(output_dir, fil, frame.name + "_backsub.fits")
                back_path = os.path.join(output_dir, fil, frame.name + "_background.fits")
                if method in ("sep", "photutils"):
                    frame.model_background_photometry(
                        write_subbed=subbed_path,
                        write=back_path,
                        do_mask=True,
                        method=method,
                        **kwargs
                    )
                elif method == "local":
                    if "centre" not in kwargs or kwargs["centre"] is None:
                        if isinstance(self.field, fld.FRBField):
                            kwargs["centre"] = self.field.frb.position
                        else:
                            kwargs["centre"] = frame.extract_pointing()
                    else:
                        kwargs["centre"] = astm.attempt_skycoord(kwargs["centre"])
                    if "frame" not in kwargs:
                        kwargs["frame"] = 15 * units.arcsec
                    if isinstance(self.field, fld.FRBField):
                        a, b = self.field.frb.position_err.uncertainty_quadrature_equ()
                        mask_ellipses = [{
                            "a": a, "b": b,
                            "theta": self.field.frb.position_err.theta,
                            "centre": self.field.frb.position
                        }]
                    else:
                        mask_ellipses = None
                    model, model_eval, data, subbed, mask, weights = frame.model_background_local(
                        write_subbed=subbed_path,
                        write=back_path,
                        generate_mask=True,
                        mask_ellipses=mask_ellipses,
                        **kwargs
                    )
                    model_info = {}
                    if "init_params" in kwargs:
                        model_info.update(kwargs["init_params"])
                    model_info.update(dict(zip(model.param_names, model.parameters)))
                    fit_info[fil][frame.name] = model_info

                new_frame = type(frame)(subbed_path)
                self.add_frame_subtracted(new_frame)
        self.stage_params["subtract_background_frames"].update(kwargs)
        self.stage_params["subtract_background_frames"]["models"] = fit_info

    def proc_register(self, output_dir: str, **kwargs):
        self.frames_registered = {}
        self.register(
            output_dir=output_dir,
            **kwargs,
        )

    def register(
            self,
            output_dir: str,
            frames: dict = None,
            template: Union[int, dict, image.ImagingImage, str] = 0,
            **kwargs
    ):
        """

        :param output_dir:
        :param frames:
        :param template: There are three options for this parameter:
            int: An integer specifying the position of the image in the list to use as the template for
            alignment (ie, each filter will use the same list position)
            dict: a dictionary with keys reflecting the filter names, with values specifying the list position as above
            ImagingImage: an image from outside this epoch to use as template. You can also pass the path to the image
                as a string.
        :param kwargs:
        :return:
        """

        u.mkdir_check(output_dir)
        u.debug_print(1, f"{self}.register(): template ==", template)

        if frames is None:
            frames = self.frames_normalised

        for fil in frames:
            if not self.quiet:
                print(f"Registering frames for {fil}")
            if isinstance(template, int):
                tmp = frames[fil][template]
                n_template = template
            elif isinstance(template, image.ImagingImage):
                # When
                tmp = template
                n_template = -1
            elif isinstance(template, str):
                tmp = image.from_path(
                    path=template,
                    cls=image.ImagingImage
                )
                n_template = -1
            else:
                tmp = frames[fil][template[fil]]
                n_template = template[fil]
            u.debug_print(1, f"{self}.register(): tmp", tmp)

            output_dir_fil = os.path.join(output_dir, fil)
            u.mkdir_check(output_dir_fil)

            self._register(frames=frames, fil=fil, tmp=tmp, output_dir=output_dir_fil, n_template=n_template, **kwargs)

    def _register(self, frames: dict, fil: str, tmp: image.ImagingImage, n_template: int, output_dir: str, **kwargs):

        include_chips = list(range(1, self.frame_class.num_chips + 1))
        if "include_chips" in kwargs and isinstance(kwargs["include_chips"], list):
            include_chips = kwargs["include_chips"]

        frames_by_chip = self.sort_by_chip(frames[fil])

        for chip in include_chips:
            for i, frame in enumerate(frames_by_chip[chip]):
                if self.is_excluded(frame):
                    continue
                if i != n_template:
                    registered = frame.register(
                        target=tmp,
                        output_path=os.path.join(
                            output_dir,
                            frame.filename.replace(".fits", "_registered.fits"))
                    )
                    self.add_frame_registered(registered)
                else:
                    registered = frame.copy(
                        os.path.join(
                            output_dir,
                            tmp.filename.replace(".fits", "_registered.fits")))
                    self.add_frame_registered(registered)

    def proc_correct_astrometry_frames(
            self,
            output_dir: str,
            **kwargs
    ):

        if "correct_to_epoch" in kwargs:
            if not self.quiet:
                print(f"correct_to_epoch 1: {kwargs['correct_to_epoch']}")
            correct_to_epoch = kwargs.pop("correct_to_epoch")
            if not self.quiet:
                print(f"correct_to_epoch 2: {correct_to_epoch}")
        else:
            correct_to_epoch = True

        u.debug_print(2, kwargs)

        self.generate_astrometry_indices(correct_to_epoch=correct_to_epoch)

        self.frames_astrometry = {}

        if "frames" in kwargs:
            frames = self._get_frames(frame_type=kwargs.pop("frames"))
        elif "register_frames" in self.do_param and self.do_param["register_frames"]:
            frames = self._get_frames(frame_type="registered")
        else:
            frames = self._get_frames(frame_type="normalised")

        self.correct_astrometry_frames(
            output_dir=output_dir,
            frames=frames,
            **kwargs
        )

    def correct_astrometry_frames(
            self,
            output_dir: str,
            frames: dict = None,
            am_params: dict = {},
            background_kwargs: dict = {},
            **kwargs
    ):
        self.frames_astrometry = {}
        self.astrometry_successful = {}

        if "back_subbed" in kwargs:
            back_subbed = kwargs.pop("back_subbed")
        else:
            back_subbed = False

        if frames is None:
            frames = self.frames_reduced

        for fil in frames:
            self.astrometry_successful[fil] = {}
            frames_by_chip = self.sort_by_chip(frames[fil])
            for chip in frames_by_chip:
                if not self.quiet:
                    print()
                    print(f"Processing frames for chip {chip} in astrometry.net:")
                    print()
                first_success = None
                astrometry_fil_path = os.path.join(output_dir, fil)
                for frame in frames_by_chip[chip]:
                    frame_alt = None
                    if back_subbed:
                        # For some fields, we want to subtract the background before attempting to solve, because of
                        # bright stars or the like.
                        frame_alt = frame
                        # Store the original frame for later.
                        subbed_path = os.path.join(output_dir, fil, frame.name + "_backsub.fits")
                        back_path = os.path.join(output_dir, fil, frame.name + "_background.fits")
                        # Use sep to subtract a background model.
                        frame.model_background_photometry(
                            write_subbed=subbed_path,
                            write=back_path,
                            do_mask=True,
                            method="sep",
                            **background_kwargs
                        )
                        # Assign frame to the subtracted file
                        frame = type(frame)(subbed_path)

                    new_frame = frame.correct_astrometry(
                        output_dir=astrometry_fil_path,
                        am_params=am_params,
                        **kwargs
                    )

                    if new_frame is not None:
                        if not self.quiet:
                            print(f"{frame} astrometry successful.")
                        if back_subbed:
                            new_frame = frame_alt.correct_astrometry_from_other(
                                new_frame,
                                output_dir=astrometry_fil_path
                            )
                            frame = frame_alt
                        self.add_frame_astrometry(new_frame)
                        self.astrometry_successful[fil][frame.name] = "astrometry.net"
                        if first_success is None:
                            first_success = new_frame
                    else:
                        if not self.quiet:
                            print(f"{frame} Astrometry.net unsuccessful; adding frame to astroalign queue.")
                        self.astrometry_successful[fil][frame.name] = False

                    u.debug_print(1, f"ImagingEpoch.correct_astrometry_frames(): {self}.astrometry_successful ==\n",
                                  self.astrometry_successful)
                    self.update_output_file()

                # Allow unsolved frames to be registered to an external image instead of against a solved frame
                if 'registration_template' in kwargs and kwargs['registration_template'] is not None:
                    first_success = image.from_path(
                        kwargs['registration_template'],
                        cls=image.ImagingImage
                    )
                elif first_success is None:
                    tmp = frames_by_chip[chip][0]
                    if not self.quiet:
                        print(
                            f"There were no successful frames for chip {chip} using astrometry.net; performing coarse correction on {tmp}.")
                    first_success, _ = tmp.correct_astrometry_coarse(
                        output_dir=astrometry_fil_path,
                        cat=self.gaia_catalogue,
                        cat_name="gaia"
                    )
                    self.add_frame_astrometry(first_success)
                    self.astrometry_successful[fil][tmp.name] = "coarse"
                    self.update_output_file()

                u.debug_print(2, "first_success", first_success)

                if not self.quiet:
                    print()
                    print(
                        f"Re-processing failed frames for chip {chip} with astroalign, with template {first_success}:")
                    print()
                for frame in frames_by_chip[chip]:
                    if self.is_excluded(frame):
                        continue
                    if not self.astrometry_successful[fil][frame.name]:
                        if not self.quiet:
                            print(f"Running astroalign on {frame}...")
                        new_frame = frame.register(
                            target=first_success,
                            output_path=os.path.join(
                                astrometry_fil_path,
                                frame.filename.replace(".fits", "_astrometry.fits")),
                        )
                        self.add_frame_astrometry(new_frame)
                        self.astrometry_successful[fil][frame.name] = "astroalign"
                    self.update_output_file()

    def proc_frame_diagnostics(self, output_dir: str, **kwargs):
        if "frames" in kwargs:
            frames = kwargs["frames"]
        else:
            frames = self.frames_final

        frame_dict = self._get_frames(frames)

        self.frame_psf_diagnostics(output_dir, frame_dict=frame_dict)
        self.frames_final = "diagnosed"

    def frame_psf_diagnostics(self, output_dir: str, frame_dict: dict, chip: int = 1, sigma: float = 1.):
        for fil in frame_dict:
            frame_list = frame_dict[fil]
            # Grab one chip only, to save time
            frame_lists = self.sort_by_chip(images=frame_list)
            frame_list_chip = frame_lists[chip]
            match_cat = None

            names = []

            fwhms_mean_psfex = []
            fwhms_mean_gauss = []
            fwhms_mean_moffat = []
            fwhms_mean_se = []

            fwhms_median_psfex = []
            fwhms_median_gauss = []
            fwhms_median_moffat = []
            fwhms_median_se = []

            sigma_gauss = []
            sigma_moffat = []
            sigma_se = []

            for frame in frame_list_chip:
                configs = self.source_extractor_config
                frame.psfex_path = None
                frame.source_extraction_psf(
                    output_dir=output_dir,
                    phot_autoparams=f"{configs['kron_factor']},{configs['kron_radius_min']}"
                )
                if match_cat is None:
                    match_cat = frame.source_cat
                offset_tolerance = 0.5 * units.arcsec
                # If the frames haven't been astrometrically corrected, give some extra leeway
                if "correct_astrometry_frames" in self.do_param and not self.do_param["correct_astrometry_frames"]:
                    offset_tolerance = 1.0 * units.arcsec
                frame_stats, stars_moffat, stars_gauss, stars_sex = frame.psf_diagnostics(
                    match_to=match_cat
                )

                names.append(frame.name)

                fwhms_mean_psfex.append(frame_stats["fwhm_psfex"].value)
                fwhms_mean_gauss.append(frame_stats["gauss"]["fwhm_mean"].value)
                fwhms_mean_moffat.append(frame_stats["moffat"]["fwhm_mean"].value)
                fwhms_mean_se.append(frame_stats["sextractor"]["fwhm_mean"].value)

                fwhms_median_psfex.append(frame_stats["fwhm_psfex"].value)
                fwhms_median_gauss.append(frame_stats["gauss"]["fwhm_median"].value)
                fwhms_median_moffat.append(frame_stats["moffat"]["fwhm_median"].value)
                fwhms_median_se.append(frame_stats["sextractor"]["fwhm_median"].value)

                sigma_gauss.append(frame_stats["gauss"]["fwhm_sigma"].value)
                sigma_moffat.append(frame_stats["moffat"]["fwhm_sigma"].value)
                sigma_se.append(frame_stats["sextractor"]["fwhm_sigma"].value)

                self.frame_stats[fil][frame.name] = frame_stats

            median_all = np.median(fwhms_mean_gauss)
            sigma_all = np.std(fwhms_median_gauss)
            upper_limit = median_all + (sigma * sigma_all)

            plt.close()

            plt.title(f"PSF FWHM Mean")
            plt.ylabel("FWHM (\")")
            plt.errorbar(names, fwhms_mean_gauss, yerr=sigma_gauss, fmt="o", label="Gaussian")
            plt.errorbar(names, fwhms_mean_moffat, yerr=sigma_gauss, fmt="o", label="Moffat")
            plt.errorbar(names, fwhms_mean_se, yerr=sigma_gauss, fmt="o", label="Source Extractor")
            plt.plot([0, len(names)], [upper_limit, upper_limit], c="black", label="Clip threshold")
            plt.legend()
            plt.xticks(rotation=-90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{fil}_psf_diagnostics_mean.png"))
            plt.close()

            plt.title(f"PSF FWHM Median")
            plt.ylabel("FWHM (\")")
            plt.errorbar(names, fwhms_median_gauss, yerr=sigma_gauss, fmt="o", label="Gaussian")
            plt.errorbar(names, fwhms_median_moffat, yerr=sigma_gauss, fmt="o", label="Moffat")
            plt.errorbar(names, fwhms_median_se, yerr=sigma_gauss, fmt="o", label="Source Extractor")
            plt.plot([0, len(names)], [upper_limit, upper_limit], c="black", label="Clip threshold")
            plt.legend()
            plt.xticks(rotation=-90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{fil}_psf_diagnostics_median.png"))
            plt.close()

            self.frames_diagnosed[fil] = []
            for i, fwhm_median in enumerate(fwhms_median_gauss):
                if fwhm_median < upper_limit:
                    print(f"Median PSF FWHM {fwhm_median} < upper limit {upper_limit}")
                    for chip in frame_lists:
                        if not self.quiet:
                            print(f"\tAdding {frame_lists[chip][i]}")
                        self.add_frame_diagnosed(frame_lists[chip][i])
                elif not self.quiet:
                    print(f"Median PSF FWHM {fwhm_median} > upper limit {upper_limit}")

    def proc_coadd(self, output_dir: str, **kwargs):
        if "frames" not in kwargs:
            kwargs["frames"] = self.frames_final
        self.coadd(output_dir, **kwargs)
        if np.any(list(map(lambda k: len(self.frames_subtracted[k]) > 0, self.frames_subtracted))):
            kwargs["frames"] = "subtracted"
            self.coadd(
                output_dir + "_background_subtracted",
                out_dict="subtracted",
                **kwargs
            )

    def coadd(
            self,
            output_dir: str,
            frames: str = "astrometry",
            out_dict: Union[dict, str] = "coadded",
            sigma_clip: float = 1.5
    ):
        """
        Use Montage and ccdproc to coadd individual frames.
        :param output_dir: Directory in which to write data products.
        :param frames: Name of frames list to coadd.
        :param sigma_clip: Multiple of pixel stack standard deviation to clip when doing sigma-clipped stack.
        :return:
        """
        if isinstance(out_dict, str):
            if not self.quiet:
                print("out_dict:", out_dict)
            out_dict = self._get_images(image_type=out_dict)

        u.mkdir_check(output_dir)
        frame_dict = self._get_frames(frame_type=frames)
        if not self.quiet:
            print(f"Coadding {frames} frames.")
            if frames == "final":
                print(f"(final is {self.frames_final})")
        for fil in self.filters:
            frame_list = frame_dict[fil]
            output_directory_fil = os.path.join(output_dir, fil)
            u.rmtree_check(output_directory_fil)
            u.mkdir_check(output_directory_fil)
            input_directory_fil = os.path.join(output_directory_fil, "inputdir")
            u.mkdir_check(input_directory_fil)
            for frame in frame_list:
                if self.is_excluded(frame):
                    continue
                frame.copy_with_outputs(input_directory_fil)

            coadded_path = montage.standard_script(
                input_directory=input_directory_fil,
                output_directory=output_directory_fil,
                output_file_name=f"{self.name}_{self.date_str()}_{fil}_coadded.fits",
                coadd_types=["median", "mean"],
                add_with_ccdproc=False,
                sigma_clip=True,
                # unit="electron / second"
                # sigma_clip_low_threshold=5,
            )[0]

            sigclip_path = coadded_path.replace("median", "mean-sigmaclip")
            area_final = sigclip_path.replace(".fits", "_area.fits")
            shutil.copy(coadded_path.replace(".fits", "_area.fits"), area_final)

            corr_dir = os.path.join(output_directory_fil, "corrdir")
            coadded_median = image.FORS2CoaddedImage(coadded_path)
            coadded_median.add_log(
                "Co-added image using Montage; see ancestor_logs for images.",
                input_path=input_directory_fil,
                output_path=coadded_path,
                ancestors=frame_list
            )
            if self.combined_epoch:
                coadded_median.set_header_item("M_EPOCH", True, write=True)
            else:
                coadded_median.set_header_item("M_EPOCH", False, write=True)
            ccds = []
            # Here we gather the projected images in preparation for custom reprojection / coaddition
            for proj_img_path in list(
                    map(
                        lambda m: os.path.join(corr_dir, m),
                        filter(
                            lambda f: f.endswith(".fits") and not f.endswith("area.fits"),
                            os.listdir(corr_dir)
                        )
                    )
            ):
                proj_img = image.FORS2Image(proj_img_path)
                reproj_img = proj_img.reproject(coadded_median, include_footprint=True)
                reproj_img_ccd = reproj_img.to_ccddata(unit="electron / second")
                ccds.append(reproj_img_ccd)

            combined_ccd = ccdproc.combine(
                img_list=ccds,
                method="average",
                sigma_clip=True,
                sigma_clip_func=np.nanmean,
                sigma_clip_dev_func=np.nanstd,
                sigma_clip_high_thresh=sigma_clip,
                sigma_clip_low_thresh=sigma_clip
            )
            combined_img = coadded_median.copy(sigclip_path)
            combined_img.area_file = area_final
            coadded_median.load_headers()
            combined_img.load_data()
            combined_img.data[0] = combined_ccd.data * coadded_median.extract_unit(astropy=True)
            u.debug_print(3, f"ImagingEpoch.coadd(): {combined_img}.headers ==", combined_img.headers)
            combined_img.add_log(
                "Co-added image using Montage for reprojection & ccdproc for coaddition; see ancestor_logs for input images.",
                input_path=input_directory_fil,
                output_path=coadded_path,
                ancestors=frame_list
            )
            combined_img.write_fits_file()
            combined_img.update_output_file()

            self._add_coadded(img=sigclip_path, key=fil, image_dict=out_dict)

    def proc_correct_astrometry_coadded(self, output_dir: str, **kwargs):
        self.correct_astrometry_coadded(
            output_dir=output_dir,
            **kwargs
        )

    def correct_astrometry_coadded(
            self,
            output_dir: str,
            image_type: str = None,
            **kwargs
    ):
        self.generate_astrometry_indices()

        self.coadded_astrometry = {}

        if image_type is None:
            image_type = "coadded"

        images = self._get_images(image_type)

        if "tweak" in kwargs:
            tweak = kwargs["tweak"]
        else:
            tweak = True

        if "astroalign_template" in kwargs:
            aa_template = kwargs["astroalign_template"]
        else:
            aa_template = None
        if aa_template is not None:
            aa_template = p.join_data_dir(aa_template)

        first_success = None
        unsuccessful = []
        for fil in images:
            img = images[fil]
            new_img = img.correct_astrometry(
                output_dir=output_dir,
                tweak=tweak
            )
            if new_img is None:
                if not self.quiet:
                    print(f"{img} Astrometry.net unsuccessful; adding image to astroalign queue.")
                unsuccessful.append(fil)
            else:
                if first_success is None:
                    first_success = new_img
                self.add_coadded_astrometry_image(new_img, key=fil)

        if first_success is None and aa_template is not None:
            cls = image.detect_instrument(path=aa_template)
            first_success = image.from_path(path=aa_template, cls=cls)

        if first_success is not None:
            for fil in unsuccessful:
                img = images[fil]
                new_img = img.register(
                    target=first_success,
                    output_path=os.path.join(
                        output_dir,
                        img.filename.replace(".fits", "_astrometry.fits")
                    )
                )
                self.add_coadded_astrometry_image(new_img, key=fil)

        for fil in images:
            if fil in self.coadded_subtracted and self.coadded_subtracted[fil] is not None:
                self.coadded_subtracted[fil] = self.coadded_subtracted[fil].correct_astrometry_from_other(
                    other_image=self.coadded_astrometry[fil],
                    output_dir=output_dir + "_background_subtracted"
                )
                self.coadded_subtracted[fil].area_file = self.coadded_astrometry[fil].area_file

    def proc_trim_coadded(self, output_dir: str, **kwargs):
        if "correct_astrometry_coadded" in self.do_param and self.do_param["correct_astrometry_coadded"]:
            images = self.coadded_astrometry
        else:
            images = self.coadded

        if "reproject" in kwargs:
            reproject = kwargs["reproject"]
        else:
            reproject = False
        self.trim_coadded(output_dir, images=images, reproject=reproject)

    def trim_coadded(
            self,
            output_dir: str,
            images: dict = None,
            reproject: bool = False,
    ):
        if images is None:
            images = self.coadded

        u.mkdir_check(output_dir)
        template = None
        for fil in images:
            img = images[fil]
            if not self.quiet:
                print()
                print("Coadded Image Path:")
                print(img.path)
            output_path = os.path.join(output_dir, img.filename.replace(".fits", "_trimmed.fits"))
            u.debug_print(2, "trim_coadded img.path:", img.path)
            u.debug_print(2, "trim_coadded img.area_file:", img.area_file)
            trimmed = img.trim_from_area(output_path=output_path)
            # trimmed.write_fits_file()
            self.add_coadded_unprojected_image(trimmed, key=fil)
            if reproject:
                if template is None:
                    template = trimmed
                else:
                    # Using the first image as a template, reproject this one into the pixel space (for alignment)
                    trimmed = trimmed.reproject(
                        other_image=template,
                        output_path=output_path.replace(".fits", "_reprojected.fits")
                    )
            self.add_coadded_trimmed_image(trimmed, key=fil)
            if fil in self.coadded_subtracted and self.coadded_subtracted[fil] is not None:
                img_sub = self.coadded_subtracted[fil]
                output_path_sub = os.path.join(output_dir, img_sub.filename.replace(".fits", "_bgsub_trimmed.fits"))
                trimmed_sub = img_sub.trim_from_area(output_path=output_path_sub)
                self.add_coadded_subtracted_trimmed_image(trimmed_sub, key=fil)

                bg_kwargs = self.stage_params["subtract_background_frames"]
                if "method" not in bg_kwargs or bg_kwargs["method"] == "local":
                    output_path_patch_unsub = os.path.join(
                        output_dir,
                        img_sub.filename.replace(".fits", "_unsubbed_patch.fits"))
                    _, p_scale = img.extract_pixel_scale()
                    left, right, bottom, top = trimmed_sub.frame_from_coord(
                        frame=bg_kwargs["frame"].to("arcsec", p_scale) - 1 * units.arcsec,
                        centre=bg_kwargs["centre"]
                    )
                    trimmed_patch = img.trim(
                        left=left, right=right, bottom=bottom, top=top,
                        output_path=output_path_patch_unsub
                    )
                    output_path_patch = os.path.join(output_dir, img_sub.filename.replace(".fits", "_bgsub_patch.fits"))
                    trimmed_patch_sub = trimmed_sub.trim(
                        left=left, right=right, bottom=bottom, top=top,
                        output_path=output_path_patch
                    )
                    self.add_coadded_subtracted_patch_image(
                        img=trimmed_patch_sub,
                        key=fil
                    )

    def proc_source_extraction(self, output_dir: str, **kwargs):
        if "do_astrometry_diagnostics" not in kwargs:
            kwargs["do_astrometry_diagnostics"] = True
        if "do_psf_diagnostics" not in kwargs:
            kwargs["do_psf_diagnostics"] = True
        if "image_type" not in kwargs:
            kwargs["image_type"] = "final"
        self.source_extraction(
            output_dir=output_dir,
            **kwargs
        )

    def source_extraction(
            self,
            output_dir: str,
            do_astrometry_diagnostics: bool = True,
            do_psf_diagnostics: bool = True,
            image_type: str = "final",
            **kwargs
    ):
        images = self._get_images(image_type)
        if not self.quiet:
            print("\nExtracting sources for", image_type, "with", len(list(images.keys())), "\n")
        for fil, img in images.items():
            if not self.quiet:
                print(f"Extracting sources from {fil} image: {img}")
            configs = self.source_extractor_config

            img.psfex_path = None
            img.source_extraction_psf(
                output_dir=output_dir,
                phot_autoparams=f"{configs['kron_factor']},{configs['kron_radius_min']}"
            )

        if do_astrometry_diagnostics:
            if "offset_tolerance" in kwargs:
                offset_tolerance = kwargs["offset_tolerance"]
            else:
                offset_tolerance = 0.5 * units.arcsec
            self.astrometry_diagnostics(
                images=images,
                offset_tolerance=offset_tolerance
            )

        if do_psf_diagnostics:
            self.psf_diagnostics(images=images)

        # Transfer psf stuff from
        if self.did_local_background_subtraction():
            for img_type in ("coadded_subtracted_patch", "coadded_subtracted"):
                images_subbed = self._get_images(img_type)
                for fil, img_subbed in images_subbed.items():
                    img = images[fil]
                    img_subbed.clone_psf(other=img)
                    img_subbed.clone_astrometry_info(other=img)
                    if img_type == "coadded_subtracted_patch":
                        img_subbed.source_extraction_psf(
                            output_dir=output_dir,
                            phot_autoparams=f"{configs['kron_factor']},{configs['kron_radius_min']}"
                        )

    def proc_photometric_calibration(self, output_dir: str, **kwargs):
        if "image_type" in kwargs and kwargs["image_type"] is not None:
            image_type = kwargs["image_type"]
        else:
            image_type = "final"

        if "distance_tolerance" in kwargs and kwargs["distance_tolerance"] is not None:
            kwargs["distance_tolerance"] = u.check_quantity(kwargs["distance_tolerance"], units.arcsec, convert=True)
        if "snr_min" not in kwargs or kwargs["snr_min"] is None:
            kwargs["snr_min"] = 3.
        if "suppress_select" not in kwargs:
            kwargs["suppress_select"] = True

        image_dict = self._get_images(image_type=image_type)
        for fil in image_dict:
            img = image_dict[fil]
            img.zeropoints = {}
            img.zeropoint_best = None

        self.photometric_calibration(
            output_path=output_dir,
            image_dict=image_dict,
            **kwargs
        )

    def photometric_calibration(
            self,
            output_path: str,
            image_dict: dict,
            **kwargs
    ):
        u.mkdir_check(output_path)

        self.zeropoint(
            image_dict=image_dict,
            output_path=output_path,
            **kwargs
        )

        for fil in image_dict:
            if self.coadded_unprojected[fil] is not None and self.coadded_unprojected[fil] is not image_dict[fil]:
                img = self.coadded_unprojected[fil]
                img.clone_zeropoints(image_dict[fil])
            if self.coadded_subtracted[fil] is not None:
                img = self.coadded_subtracted[fil]
                img.clone_zeropoints(image_dict[fil])
            if self.coadded_subtracted_trimmed[fil] is not None:
                img = self.coadded_subtracted_trimmed[fil]
                img.clone_zeropoints(image_dict[fil])
            if self.coadded_subtracted_patch[fil] is not None:
                img = self.coadded_subtracted_patch[fil]
                img.clone_zeropoints(image_dict[fil])

    def zeropoint(
            self,
            image_dict: dict,
            output_path: str,
            distance_tolerance: units.Quantity = None,
            snr_min: float = 3.,
            star_class_tolerance: int = 0.95,
            suppress_select: bool = True,
            **kwargs
    ):

        for fil in self.filters:
            img = image_dict[fil]
            for cat_name in retrieve.photometry_catalogues:
                if cat_name == "gaia":
                    continue
                if cat_name in retrieve.cat_systems and retrieve.cat_systems[cat_name] == "vega":
                    vega = True
                else:
                    vega = False
                fil_path = os.path.join(output_path, fil)
                u.mkdir_check(fil_path)
                if f"in_{cat_name}" in self.field.cats and self.field.cats[f"in_{cat_name}"]:
                    img.zeropoint(
                        cat=self.field.get_path(f"cat_csv_{cat_name}"),
                        output_path=os.path.join(fil_path, cat_name),
                        cat_name=cat_name,
                        dist_tol=distance_tolerance,
                        show=False,
                        snr_cut=snr_min,
                        star_class_tol=star_class_tolerance,
                        vega=vega,
                        **kwargs
                    )

            if "preferred_zeropoint" in kwargs and fil in kwargs["preferred_zeropoint"]:
                preferred = kwargs["preferred_zeropoint"][fil]
            else:
                preferred = None

            zeropoint, cat = img.select_zeropoint(suppress_select, preferred=preferred)

    def proc_dual_mode_source_extraction(self, output_dir: str, **kwargs):
        if "image_type" in kwargs and isinstance(kwargs["image_type"], str):
            image_type = kwargs["image_type"]
        else:
            image_type = "final"
        self.dual_mode_source_extraction(output_dir, image_type)

    def dual_mode_source_extraction(self, path: str, image_type: str = "coadded_trimmed"):
        image_dict = self._get_images(image_type=image_type)
        u.mkdir_check(path)
        if self.deepest is None:
            if self.deepest_filter is not None:
                self.deepest = image_dict[self.deepest_filter]
            else:
                raise ValueError(f"deepest for {self.name} is None; make sure you have run photometric_calibration.")
        for fil in image_dict:
            img = image_dict[fil]
            configs = self.source_extractor_config
            img.source_extraction_psf(
                output_dir=path,
                phot_autoparams=f"{configs['kron_factor']},{configs['kron_radius_min']}",
                template=self.deepest
            )

    def proc_finalise(
            self,
            output_dir: str,
            **kwargs
    ):

        self.finalise(output_path=output_dir, **kwargs)

    def finalise(
            self,
            image_type: str = "final",
            output_path: str = None,
            **kwargs
    ):
        """Performs a number of wrap-up actions:
        - Ensures final fits files have correct header information
        - Renames and copies final files to appropriate paths
        - Updates epoch tables

        :return:
        """

        staging_dir = os.path.join(
            p.data_dir,
            "Finalised"
        )

        image_dict = self._get_images(image_type=image_type)

        deepest = list(self.coadded_unprojected.values())[0]
        for fil, img in self.coadded_unprojected.items():

            img_prime = image_dict[fil]

            if img is None:
                continue

            if isinstance(self.instrument, Instrument):
                inst_name = self.instrument.nice_name().replace('/', '-')
            else:
                inst_name = self.instrument_name

            if self.combined_epoch:
                date = "combined"
            else:
                date = self.date_str()

            nice_name = f"{self.field.name}_{inst_name}_{fil.replace('_', '-')}_{date}.fits"

            img.estimate_depth(
                zeropoint_name="best",
                output_dir=output_path,
                test_coord=self.target
            )

            img.select_depth()
            img.write_fits_file()

            # Make sure various common properties are transferred to image derivatives.
            if img != img_prime:
                img.clone_diagnostics(img_prime)
            if isinstance(self.coadded_subtracted[fil], image.CoaddedImage):
                self.coadded_subtracted[fil].clone_diagnostics(img_prime)
            if isinstance(self.coadded_subtracted_trimmed[fil], image.CoaddedImage):
                self.coadded_subtracted_trimmed[fil].clone_diagnostics(img_prime)
            if isinstance(self.coadded_subtracted_patch[fil], image.CoaddedImage):
                self.coadded_subtracted_patch[fil].clone_diagnostics(img_prime)

            img_final = img.copy_with_outputs(os.path.join(
                self.data_path,
                nice_name)
            )

            img.copy_with_outputs(
                os.path.join(
                    staging_dir,
                    nice_name
                )
            )

            if isinstance(self.field.survey, Survey):
                refined_path = self.field.survey.refined_stage_path

                if refined_path is not None:
                    img.copy_with_outputs(
                        os.path.join(
                            refined_path,
                            nice_name
                        )
                    )

            deepest = image.deepest(deepest, img)
            if self.did_local_background_subtraction():
                img_subbed = self.coadded_subtracted_patch[fil]
                self.field.add_image(img_subbed)
            else:
                self.field.add_image(img_final)

        self.deepest_filter = deepest.filter_name
        self.deepest = deepest

        self.push_to_table()
        return deepest

    def proc_get_photometry(
            self,
            output_dir: str,
            **kwargs
    ):
        if "image_type" in kwargs and isinstance(kwargs["image_type"], str):
            image_type = kwargs.pop("image_type")
        else:
            image_type = "final"
        u.debug_print(2, f"{self}.proc_get_photometry(): image_type ==:", image_type)
        # Run PATH on imaging if we're doing FRB stuff

        # skip_path = False
        # if "skip_path" in kwargs:
        #     skip_path = kwargs.pop("skip_path")
        #
        # if not skip_path and isinstance(self.field, fld.FRBField):
        #     path_kwargs = {
        #         "priors": {"U": 0.1},
        #         "config": {"radius": 10}
        #     }
        #     if 'path_kwargs' in kwargs:
        #         path_kwargs.update(kwargs["path_kwargs"])
        #     image_type_path = image_type
        #     if self.did_local_background_subtraction():
        #         image_type_path = "coadded_subtracted_patch"
        #     self.probabilistic_association(
        #         image_type=image_type_path,
        #         **path_kwargs
        #     )
        self.get_photometry(output_dir, image_type=image_type, **kwargs)

    def did_local_background_subtraction(self):
        if "subtract_background_frames" in self.do_param:
            if "subtract_background_frames" in self.stage_params:
                stage_params = self.stage_params["subtract_background_frames"]
                if "method" not in stage_params or stage_params["method"] == "local":
                    return True
        return False

    def best_for_path(
            self,
            image_type: str = "final",
            exclude: list = ()
    ):
        image_dict = self._get_images(image_type=image_type)
        return best_for_path(image_dict, exclude=exclude)

    def probabilistic_association(
            self,
            image_type: str = "final",
            **path_kwargs
    ):
        if not isinstance(self.field, fld.FRBField):
            raise TypeError(
                f"To run probabilistic_association, {self.name}.field must be an FRBField, not {type(self.field)}")
        image_dict = self._get_images(image_type=image_type)
        self.field.frb.load_output_file()

        failed = []
        for fil in image_dict:
            img = image_dict[fil]
            try:
                self.field.frb.probabilistic_association(
                    img=img,
                    do_plot=True,
                    **path_kwargs
                )
            except ValueError:
                print(f"PATH failed on {fil} image.")
                failed.append(fil)

        best_img = self.best_for_path(image_type=image_type, exclude=failed)

        self.field.frb.consolidate_candidate_tables(
            sort_by="P_Ox",
            reverse_sort=True,
            p_ox_assign=best_img.name
        )
        # Add the candidates to the field's object list.
        # self.field.add_path_candidates()

        # yaml_dict = {}
        #
        # for i, o in enumerate(path_cat_02):
        #     obj_name = f"HC{o['id']}-20210912A"
        #     obj_dict = objects.Galaxy.default_params()
        #     obj_dict["name"] = obj_name
        #     obj_dict["position"]["alpha"]["decimal"] = o["ra"].value
        #     obj_dict["position"]["delta"]["decimal"] = o["dec"].value
        #     yaml_dict[obj_name] = obj_dict
        #
        # p.save_params(os.path.join(output_path, "PATH", "host_candidates.yaml"), yaml_dict)

    def get_photometry(
            self,
            path: str,
            image_type: str = "final",
            dual: bool = False,
            match_tolerance: units.Quantity = 1 * units.arcsec,
            **kwargs
    ):
        """Retrieve photometric properties of key objects and write to disk.

        :param path: Path to which to write the data products.
        :param image_type:
        :param dual:
        :param match_tolerance:
        :param kwargs:
        :return:
        """

        from craftutils.observation.output.epoch import imaging_table

        print(self.field.objects.keys())

        if not self.quiet:
            print(f"Getting finalised photometry for key objects, in {image_type}.")

        match_tolerance = u.check_quantity(match_tolerance, unit=units.arcsec)

        imaging_table.load_table()

        skip_plots = False
        if "skip_plots" in kwargs:
            skip_plots = kwargs["skip_plots"]

        image_dict = self._get_images(image_type=image_type)
        u.mkdir_check(path)

        # Loop through filters
        for fil, img in image_dict.items():

            fil_output_path = os.path.join(path, fil)
            u.mkdir_check(fil_output_path)

            if "secure" not in img.depth:
                img.estimate_depth(test_coord=self.target)
            if not self.quiet:
                print("Getting photometry for", img)

            # Transform source extractor magnitudes using zeropoint.
            img.calibrate_magnitudes(zeropoint_name="best", dual=dual, force=True)

            # Set up lists for matches to turn into a subset of the SE catalogue
            rows = []
            names = []
            separations = []
            ra_target = []
            dec_target = []

            if "SNR_PSF" in img.depth["secure"]:
                depth = img.depth["secure"]["SNR_PSF"][f"5-sigma"]
            else:
                depth = img.depth["secure"]["SNR_AUTO"][f"5-sigma"]

            img.load_data()

            if not skip_plots:
                fig, ax, vals = self.field.plot_host(
                    img=img,
                    frame=10 * units.arcsec,
                    centre=self.field.frb.position,
                )
                for obj_name, obj in self.field.objects.items():
                    x, y = img.world_to_pixel(obj.position)
                    plt.scatter(x, y, marker="x", label=obj_name)
                plt.legend(loc=(1.0, 0.))
                fig.savefig(os.path.join(fil_output_path, "plot_quick.pdf"))

            # Get astrometric uncertainty from images
            img.extract_astrometry_err()
            # Add that to the matching tolerance in quadrature (seems like the right thing to do?)

            print("Specified tolerance:", match_tolerance)
            print("Image astrometric uncertainty:", img.astrometry_err)
            if img.astrometry_err is None:
                tolerance_eff = match_tolerance
            else:
                tolerance_eff = np.sqrt(match_tolerance ** 2 + img.astrometry_err ** 2)
            print("Effective tolerance:", tolerance_eff)
            # Loop through this field's 'objects' dictionary and try to match them with the SE catalogue
            for obj_name, obj in self.field.objects.items():
                # If the object is not expected to be visible in the optical/NIR, skip it.
                if obj is None or obj.position is None or not obj.optical:
                    continue
                print(f"Looking for matches to {obj_name} ({obj.position.to_string('hmsdms')})")
                # obj.load_output_file()
                plt.close()
                # Get nearest Source-Extractor object
                nearest, separation = img.find_object(obj.position, dual=dual)
                # Stick it in the table lists
                names.append(obj.name)
                rows.append(nearest)
                separations.append(separation.to(units.arcsec))
                ra_target.append(obj.position.ra)
                dec_target.append(obj.position.dec)

                if self.did_local_background_subtraction():
                    good_image_path = self.coadded_subtracted[fil].path
                else:
                    good_image_path = self.coadded_unprojected[fil].path
                # If the nearest object is outside tolerance, declare that no match was found and send a dummy entry to
                # the object's photometry table.
                if separation > tolerance_eff:
                    obj.add_photometry(
                        instrument=self.instrument_name,
                        fil=fil,
                        epoch_name=self.name,
                        mag=-999 * units.mag,
                        mag_err=-999 * units.mag,
                        snr=-999,
                        ellipse_a=-999 * units.arcsec,
                        ellipse_a_err=-999 * units.arcsec,
                        ellipse_b=-999 * units.arcsec,
                        ellipse_b_err=-999 * units.arcsec,
                        ellipse_theta=-999 * units.arcsec,
                        ellipse_theta_err=-999 * units.arcsec,
                        ra=-999 * units.deg,
                        ra_err=-999 * units.deg,
                        dec=-999 * units.deg,
                        dec_err=-999 * units.deg,
                        kron_radius=-999.,
                        separation_from_given=separation,
                        epoch_date=self.date_str(),
                        class_star=-999.,
                        spread_model=-999.,
                        spread_model_err=-999.,
                        class_flag=-999,
                        mag_psf=-999. * units.mag,
                        mag_psf_err=-999. * units.mag,
                        snr_psf=-999.,
                        image_depth=depth,
                        image_path=img.path,
                        good_image_path=good_image_path,
                        do_mask=img.mask_nearby()
                    )
                    print(
                        f"No object detected at position (nearest match at {nearest['RA']}, {nearest['DEC']}, separation {separation.to('arcsec')}).")
                    print()
                # Otherwise,  send the match's information to the object's photometry table (and make plots).
                else:
                    u.debug_print(2, "ImagingImage.get_photometry(): nearest.colnames ==", nearest.colnames)
                    err = nearest[f'MAGERR_AUTO_ZP_best']
                    if not self.quiet:
                        print(f"MAG_AUTO = {nearest['MAG_AUTO_ZP_best']} +/- {err}")
                        print(f"A = {nearest['A_WORLD'].to(units.arcsec)}; B = {nearest['B_WORLD'].to(units.arcsec)}")
                    img.plot_source_extractor_object(
                        nearest,
                        output=os.path.join(fil_output_path, f"{obj.name_filesys}.png"),
                        show=False,
                        title=f"{obj.name}, {fil}-band, {nearest['MAG_AUTO_ZP_best'].round(3).value} ± {err.round(3)}",
                        find=obj.position
                    )
                    obj.cat_row = nearest

                    if "MAG_PSF_ZP_best" in nearest.colnames:
                        mag_psf = nearest["MAG_PSF_ZP_best"]
                        mag_psf_err = nearest["MAGERR_PSF_ZP_best"]
                        snr_psf = nearest["FLUX_PSF"] / nearest["FLUXERR_PSF"]
                        spread_model = nearest["SPREAD_MODEL"]
                        spread_model_err = nearest["SPREADERR_MODEL"]
                        class_flag = nearest["CLASS_FLAG"]
                    else:
                        mag_psf = -999.0 * units.mag
                        mag_psf_err = -999.0 * units.mag
                        snr_psf = -999.0
                        spread_model = -999.0
                        spread_model_err = -999.0
                        class_flag = -999

                    obj.add_photometry(
                        instrument=self.instrument_name,
                        fil=fil,
                        epoch_name=self.name,
                        mag=nearest['MAG_AUTO_ZP_best'],
                        mag_err=err,
                        snr=nearest['SNR_AUTO'],
                        ellipse_a=nearest['A_WORLD'],
                        ellipse_a_err=nearest["ERRA_WORLD"],
                        ellipse_b=nearest['B_WORLD'],
                        ellipse_b_err=nearest["ERRB_WORLD"],
                        ellipse_theta=nearest['THETA_WORLD'],
                        ellipse_theta_err=nearest['ERRTHETA_WORLD'],
                        ra=nearest['RA'],
                        ra_err=np.sqrt(nearest["ERRX2_WORLD"]),
                        dec=nearest['DEC'],
                        dec_err=np.sqrt(nearest["ERRY2_WORLD"]),
                        kron_radius=nearest["KRON_RADIUS"],
                        separation_from_given=separation,
                        epoch_date=img.extract_date_obs(),
                        class_star=nearest["CLASS_STAR"],
                        spread_model=spread_model,
                        spread_model_err=spread_model_err,
                        class_flag=class_flag,
                        mag_psf=mag_psf,
                        mag_psf_err=mag_psf_err,
                        snr_psf=snr_psf,
                        image_depth=depth,
                        image_path=img.path,
                        good_image_path=good_image_path,
                        do_mask=img.mask_nearby()
                    )

                    # Do more plots with FRB ellipses if this is an FRBField
                    if isinstance(self.field, fld.FRBField) and not skip_plots:

                        frames = [
                            img.nice_frame(row=obj.cat_row),
                            10 * units.arcsec,
                            20 * units.arcsec,
                            40 * units.arcsec,
                        ]
                        if "frame" in obj.plotting_params and obj.plotting_params["frame"] is not None:
                            frames.append(obj.plotting_params["frame"])

                        normalize_kwargs = {}
                        if fil in obj.plotting_params:
                            if "normalize" in obj.plotting_params[fil]:
                                normalize_kwargs = obj.plotting_params[fil]["normalize"]
                        for frame in frames:
                            for stretch in ["log", "sqrt"]:
                                print(f"\nPlotting {frame=}, {stretch=}")
                                normalize_kwargs["stretch"] = stretch
                                centre = obj.position_from_cat_row()

                                fig = plt.figure(figsize=(6, 5))
                                fig, ax, other = self.field.plot_host(
                                    img=img,
                                    fig=fig,
                                    centre=centre,
                                    show_frb=True,
                                    frame=frame,
                                    imshow_kwargs={
                                        "cmap": "plasma"
                                    },
                                    frb_kwargs={
                                        "edgecolor": "black"
                                    },
                                    normalize_kwargs=normalize_kwargs
                                )
                                output_path = os.path.join(
                                    fil_output_path,
                                    f"{obj.name_filesys}_{fil}_{str(frame).replace(' ', '-')}_{stretch}")
                                name = obj.name
                                img.extract_filter()
                                if img.filter is None:
                                    f_name = fil
                                else:
                                    f_name = img.filter.nice_name()
                                ax.set_title(f"{name}, {f_name}")
                                fig.savefig(output_path + ".pdf")
                                fig.savefig(output_path + ".png")
                                ax.clear()
                                fig.clear()
                                plt.close("all")
                                pl.latex_off()
                                del fig, ax, other

            # Make the subset SE table
            if len(rows) > 0:
                tbl = table.vstack(rows)
                tbl.add_column(names, name="NAME")
                tbl.add_column(separations, name="OFFSET_FROM_TARGET")
                tbl.add_column(ra_target, name="RA_TARGET")
                tbl.add_column(dec_target, name="DEC_TARGET")

                tbl.write(
                    os.path.join(fil_output_path, f"{self.field.name}_{self.name}_{fil}.ecsv"),
                    format="ascii.ecsv",
                    overwrite=True
                )
                tbl.write(
                    os.path.join(fil_output_path, f"{self.field.name}_{self.name}_{fil}.csv"),
                    format="ascii.csv",
                    overwrite=True
                )

        for obj in self.field.objects.values():
            obj.update_output_file()
            # obj.push_to_table(select=True)
            # obj.write_plot_photometry()

    # def proc_get_photometry_all(self, output_dir: str, **kwargs):
    #     if "image_type" in kwargs and isinstance(kwargs["image_type"], str):
    #         image_type = kwargs["image_type"]
    #     else:
    #         image_type = "final"
    #     self.get_photometry_all(output_dir, image_type=image_type)

    # def get_photometry_all(
    #         self, path: str,
    #         image_type: str = "coadded_trimmed",
    #         dual: bool = False
    # ):
    #     obs.load_master_all_objects_table()
    #     image_dict = self._get_images(image_type=image_type)
    #     u.mkdir_check(path)
    #     # Loop through filters
    #     for fil in image_dict:
    #         fil_output_path = os.path.join(path, fil)
    #         u.mkdir_check(fil_output_path)
    #         img = image_dict[fil]
    #         img.push_source_cat(dual=dual)

    def astrometry_diagnostics(
            self,
            images: dict = None,
            reference_cat: table.QTable = None,
            offset_tolerance: units.Quantity = 0.5 * units.arcsec
    ):

        if images is None:
            images = self._get_images("final")
        elif isinstance(images, str):
            images = self._get_images(images)

        if reference_cat is None:
            reference_cat = self.epoch_gaia_catalogue()

        for fil, img in images.items():
            print("Attempting astrometry diagnostics for", img.name)
            img.source_cat.load_table()
            stats = -99.
            while not isinstance(stats, dict):
                stats = img.astrometry_diagnostics(
                    reference_cat=reference_cat,
                    local_coord=self.field.centre_coords,
                    offset_tolerance=offset_tolerance
                )
                offset_tolerance += 0.5 * units.arcsec
            stats["file_path"] = img.path
            self.astrometry_stats[fil] = stats

        self.add_log(
            "Ran astrometry diagnostics.",
            method=self.astrometry_diagnostics,
        )

        self.update_output_file()
        return self.astrometry_stats

    def psf_diagnostics(
            self,
            images: dict = None
    ):
        if images is None:
            images = self._get_images("final")

        for fil in images:
            img = images[fil]
            if not self.quiet:
                print(f"Performing PSF measurements on {img}...")
            self.psf_stats[fil], _ = img.psf_diagnostics()
            self.psf_stats[fil]["file_path"] = img.path

        self.update_output_file()
        return self.psf_stats

    def _get_images(self, image_type: str) -> Dict[str, image.CoaddedImage]:
        """
        A helper method for finding the desired coadded image dictionary.
        :param image_type: "trimmed", "coadded", "unprojected" or "astrometry"
        :return: dict with filter names as keys and CoaddedImage objects as values.
        """

        if image_type in ["final", "coadded_final"]:
            if self.coadded_final is not None:
                image_type = self.coadded_final
            else:
                raise ValueError("coadded_final has not been set.")

        if image_type in ["coadded_trimmed", "trimmed"]:
            image_dict = self.coadded_trimmed
        elif image_type == "coadded":
            image_dict = self.coadded
        elif image_type in ["coadded_unprojected", "unprojected"]:
            image_dict = self.coadded_unprojected
        elif image_type in ["coadded_subtracted", "subtracted"]:
            image_dict = self.coadded_subtracted
        elif image_type in ["coadded_subtracted_trimmed", "subtracted_trimmed"]:
            image_dict = self.coadded_subtracted_trimmed
        elif image_type in ["coadded_subtracted_patch", "subtracted_patch"]:
            image_dict = self.coadded_subtracted_patch
        elif image_type in ["coadded_astrometry", "astrometry"]:
            image_dict = self.coadded_astrometry
        else:
            raise ValueError(f"Images type '{image_type}' not recognised.")
        return image_dict

    def _get_frames(self, frame_type: str) -> Dict[str, List[image.ImagingImage]]:
        """
        A helper method for finding the desired frame dictionary
        :param frame_type: "science", "reduced", "trimmed", "normalised", "registered", "astrometry" or "diagnosed"
        :return: dictionary, with filter names as keys, and lists of frame Image objects as keys.
        """
        if frame_type == "final":
            if self.frames_final is not None:
                frame_type = self.frames_final
            else:
                raise ValueError("frames_final has not been set.")

        if frame_type in ("science", "frames_science"):
            image_dict = self.frames_science
        elif frame_type in ("reduced", "frames_reduced"):
            image_dict = self.frames_reduced
        elif frame_type in ("trimmed", "frames_trimmed"):
            image_dict = self.frames_trimmed
        elif frame_type in ("normalised", "frames_normalised"):
            image_dict = self.frames_normalised
        elif frame_type in ("subtracted", "frames_substracted"):
            image_dict = self.frames_subtracted
        elif frame_type in ("registered", "frames_registered"):
            image_dict = self.frames_registered
        elif frame_type in ("astrometry", "frames_astrometry"):
            image_dict = self.frames_astrometry
        elif frame_type == ("diagnosed", "frames_diagnosed"):
            image_dict = self.frames_diagnosed
        else:
            raise ValueError(f"Frame type '{frame_type}' not recognised.")

        return image_dict

    def guess_data_path(self):
        if self.data_path is None and self.field is not None and self.field.data_path is not None and \
                self.instrument_name is not None and self.date is not None:
            self.data_path = self.build_data_path_absolute(
                field=self.field,
                instrument_name=self.instrument_name,
                date=self.date,
                name=self.name
            )
        return self.data_path

    def _output_dict(self):
        output_dict = super()._output_dict()
        if self.deepest is not None:
            deepest = self.deepest.path
        else:
            deepest = None
        from ..epoch import _output_img_list, _output_img_dict_single, _output_img_dict_list
        output_dict.update({
            "airmass_mean": self.airmass_mean,
            "airmass_err": self.airmass_err,
            "astrometry_indices": self.astrometry_indices,
            "astrometry_successful": self.astrometry_successful,
            "astrometry_stats": self.astrometry_stats,
            "coadded": _output_img_dict_single(self.coadded),
            "coadded_final": self.coadded_final,
            "coadded_trimmed": _output_img_dict_single(self.coadded_trimmed),
            "coadded_unprojected": _output_img_dict_single(self.coadded_unprojected),
            "coadded_astrometry": _output_img_dict_single(self.coadded_astrometry),
            "coadded_subtracted": _output_img_dict_single(self.coadded_subtracted),
            "coadded_subtracted_trimmed": _output_img_dict_single(self.coadded_subtracted_trimmed),
            "coadded_subtracted_patch": _output_img_dict_single(self.coadded_subtracted_patch),
            "deepest": deepest,
            "deepest_filter": self.deepest_filter,
            "exp_time_mean": self.exp_time_mean,
            "exp_time_err": self.exp_time_err,
            "filters": self.filters,
            "frames_final": self.frames_final,
            "frames_raw": _output_img_list(self.frames_raw),
            "frames_reduced": _output_img_dict_list(self.frames_reduced),
            "frames_normalised": _output_img_dict_list(self.frames_normalised),
            "frames_subtracted": _output_img_dict_list(self.frames_subtracted),
            "frames_registered": _output_img_dict_list(self.frames_registered),
            "frames_astrometry": _output_img_dict_list(self.frames_astrometry),
            "frames_diagnosed": _output_img_dict_list(self.frames_diagnosed),
            "psf_stats": self.psf_stats,
            "std_pointings": self.std_pointings,
            "validation_catalogue_path": self.validation_catalogue_path
        })
        return output_dict

    def load_output_file(self, **kwargs):
        outputs = super().load_output_file(**kwargs)
        if isinstance(outputs, dict):
            frame_cls = image.ImagingImage.select_child_class(instrument_name=self.instrument_name, mode='imaging')
            coadd_class = image.CoaddedImage.select_child_class(instrument_name=self.instrument_name, mode='imaging')
            if self.date is None:
                if "date" in outputs:
                    self.set_date(outputs["date"])
            if "filters" in outputs:
                self.filters = outputs["filters"]
            if self._check_output_file_path("deepest", outputs):
                self.deepest = image.from_path(
                    path=outputs["deepest"],
                    cls=coadd_class
                )
            if "deepest_filter" in outputs:
                self.deepest_filter = outputs["deepest_filter"]
            if "exp_time_mean" in outputs:
                self.exp_time_mean = outputs["exp_time_mean"]
            if "exp_time_err" in outputs:
                self.exp_time_err = outputs["exp_time_err"]
            if "airmass_mean" in outputs:
                self.airmass_mean = outputs["airmass_mean"]
            if "airmass_err" in outputs:
                self.airmass_err = outputs["airmass_err"]
            if "psf_stats" in outputs:
                self.psf_stats = outputs["psf_stats"]
            if "astrometry_stats" in outputs:
                self.astrometry_stats = outputs["astrometry_stats"]
            if "astrometry_successful" in outputs:
                self.astrometry_successful = outputs["astrometry_successful"]
            if "astrometry_indices" in outputs:
                self.astrometry_indices = outputs["astrometry_indices"]
            if "validation_catalogue_path" in outputs:
                self.validation_catalogue_path = outputs["validation_catalogue_path"]
            if "frames_raw" in outputs:
                for frame in set(outputs["frames_raw"]):
                    if os.path.isfile(frame):
                        self.add_frame_raw(raw_frame=frame)
            if "frames_reduced" in outputs:
                for fil in outputs["frames_reduced"]:
                    if outputs["frames_reduced"][fil] is not None:
                        for frame in set(outputs["frames_reduced"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_reduced(frame=frame)
            if "frames_normalised" in outputs:
                for fil in outputs["frames_normalised"]:
                    if outputs["frames_normalised"][fil] is not None:
                        for frame in set(outputs["frames_normalised"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_normalised(frame=frame)
            if "frames_subtracted" in outputs:
                for fil in outputs["frames_subtracted"]:
                    if outputs["frames_subtracted"][fil] is not None:
                        for frame in set(outputs["frames_subtracted"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_subtracted(frame=frame)

            if "frames_registered" in outputs:
                for fil in outputs["frames_registered"]:
                    if outputs["frames_registered"][fil] is not None:
                        for frame in set(outputs["frames_registered"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_registered(frame=frame)
            if "frames_astrometry" in outputs:
                for fil in outputs["frames_astrometry"]:
                    if outputs["frames_astrometry"][fil] is not None:
                        for frame in set(outputs["frames_astrometry"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_astrometry(frame=frame)
            if "frames_diagnosed" in outputs:
                for fil in outputs["frames_diagnosed"]:
                    if outputs["frames_diagnosed"][fil] is not None:
                        for frame in set(outputs["frames_diagnosed"][fil]):
                            if os.path.isfile(frame):
                                self.add_frame_diagnosed(frame=frame)
            if "coadded" in outputs:
                for fil in outputs["coadded"]:
                    if outputs["coadded"][fil] is not None:
                        self.add_coadded_image(img=outputs["coadded"][fil], key=fil, **kwargs)
            if "coadded_subtracted" in outputs:
                for fil in outputs["coadded_subtracted"]:
                    if outputs["coadded_subtracted"][fil] is not None:
                        self.add_coadded_subtracted_image(img=outputs["coadded_subtracted"][fil], key=fil, **kwargs)
            if "coadded_subtracted_trimmed" in outputs:
                for fil in outputs["coadded_subtracted_trimmed"]:
                    if outputs["coadded_subtracted_trimmed"][fil] is not None:
                        self.add_coadded_subtracted_trimmed_image(
                            img=outputs["coadded_subtracted_trimmed"][fil],
                            key=fil,
                            **kwargs
                        )
            if "coadded_subtracted_patch" in outputs:
                for fil in outputs["coadded_subtracted_patch"]:
                    if outputs["coadded_subtracted_patch"][fil] is not None:
                        self.add_coadded_subtracted_patch_image(
                            img=outputs["coadded_subtracted_patch"][fil],
                            key=fil,
                            **kwargs
                        )
            if "coadded_trimmed" in outputs:
                for fil in outputs["coadded_trimmed"]:
                    if outputs["coadded_trimmed"][fil] is not None:
                        u.debug_print(1, f"Attempting to load coadded_trimmed[{fil}]")
                        self.add_coadded_trimmed_image(img=outputs["coadded_trimmed"][fil], key=fil, **kwargs)
            if "coadded_unprojected" in outputs:
                for fil in outputs["coadded_unprojected"]:
                    if outputs["coadded_unprojected"][fil] is not None:
                        u.debug_print(1, f"Attempting to load coadded_unprojected[{fil}]")
                        self.add_coadded_unprojected_image(img=outputs["coadded_unprojected"][fil], key=fil, **kwargs)
            if "coadded_astrometry" in outputs:
                for fil in outputs["coadded_astrometry"]:
                    if outputs["coadded_astrometry"][fil] is not None:
                        u.debug_print(1, f"Attempting to load coadded_astrometry[{fil}]")
                        self.add_coadded_astrometry_image(img=outputs["coadded_astrometry"][fil], key=fil, **kwargs)
            if "std_pointings" in outputs:
                self.std_pointings = outputs["std_pointings"]

        return outputs

    def generate_astrometry_indices(
            self,
            cat_name="gaia",
            correct_to_epoch: bool = True,
            force: bool = False,
            delete_others: bool = True
    ):
        """
        Generates astrometry indices using astrometry.net and the specified catalogue, unless they have been generated
        before; in which case it simply copies them to the main index directory (overwriting those of other epochs there).

        :param cat_name:
        :param correct_to_epoch:
        :param force:
        :return:
        """
        if not isinstance(self.field, fld.Field):
            raise ValueError("field has not been set for this observation.")

        do_indices = False
        if force or not self.astrometry_indices:
            do_indices = True
        else:
            for path in self.astrometry_indices:
                if not os.path.isdir(path):
                    do_indices = True
                    break

        if do_indices:
            epoch_index_path = os.path.join(self.data_path, "astrometry_indices")
            self.field.retrieve_catalogue(cat_name=cat_name)

            csv_path = self.field.get_path(f"cat_csv_{cat_name}")

            if cat_name == "gaia":
                cat = self.epoch_gaia_catalogue(correct_to_epoch=correct_to_epoch)
            else:
                cat = retrieve.load_catalogue(
                    cat_name=cat_name,
                    cat=csv_path
                )

            unique_id_prefix = int(
                f"{abs(int(self.field.centre_coords.ra.value))}{abs(int(self.field.centre_coords.dec.value))}")

            self.astrometry_indices = astm.generate_astrometry_indices(
                cat_name=cat_name,
                cat=cat,
                output_file_prefix=f"{cat_name}_index_{self.field.name}",
                index_output_dir=epoch_index_path,
                fits_cat_output=csv_path.replace(".csv", ".fits"),
                p_lower=-2,
                p_upper=2,
                unique_id_prefix=unique_id_prefix,
                add_path=False
            )
        index_path = os.path.join(p.config["top_data_dir"], "astrometry_index_files")
        u.mkdir_check(index_path)
        cat_index_path = os.path.join(index_path, cat_name)
        astm.astrometry_net.add_index_directory(cat_index_path)
        if delete_others:
            for file in os.listdir(cat_index_path):
                file_path = os.path.join(cat_index_path, file)
                u.rm_check(file_path)
        for index_path in self.astrometry_indices:
            shutil.copy(index_path, cat_index_path)
        self.update_output_file()
        return self.astrometry_indices

    def epoch_gaia_catalogue(
            self,
            correct_to_epoch: bool = True
    ):
        if correct_to_epoch:
            if self.date is None:
                raise ValueError(f"{self}.date not set; needed to correct Gaia cat to epoch.")
            self.gaia_catalogue = astm.correct_gaia_to_epoch(
                self.field.get_path(f"cat_csv_gaia"),
                new_epoch=self.date
            )
        else:
            self.gaia_catalogue = astm.load_catalogue(cat_name="gaia", cat=self.field.get_path(f"cat_csv_gaia"))
        return self.gaia_catalogue

    def _check_frame(self, frame: Union[image.ImagingImage, str], frame_type: str):
        if isinstance(frame, str):
            if os.path.isfile(frame):
                cls = image.ImagingImage.select_child_class(instrument_name=self.instrument_name)
                u.debug_print(2, f"{cls} {self.instrument_name}")
                frame = image.from_path(
                    path=frame,
                    frame_type=frame_type,
                    cls=cls
                )
            else:
                u.debug_print(2, f"File {frame} not found.")
                return None, None
        fil = frame.extract_filter()
        frame.epoch = self

        return frame, fil

    def _add_frame(self, frame: Union[image.ImagingImage, str], frames_dict: dict, frame_type: str):
        frame, fil = self._check_frame(frame=frame, frame_type=frame_type)
        if frame is None:
            return None
        if self.check_filter(fil=fil) and frame not in frames_dict[fil]:
            frames_dict[fil].append(frame)
        return frame

    def add_frame_raw(self, raw_frame: Union[image.ImagingImage, str]):
        raw_frame, fil = self._check_frame(frame=raw_frame, frame_type="raw")
        self.check_filter(fil)
        if raw_frame is None:
            return None
        if raw_frame not in self.frames_raw:
            self.frames_raw.append(raw_frame)
        self.sort_frame(raw_frame, sort_key=fil)
        return raw_frame

    def add_frame_reduced(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_reduced, frame_type="reduced")

    def add_frame_trimmed(self, frame: image.ImagingImage):
        self._add_frame(frame=frame, frames_dict=self.frames_trimmed, frame_type="reduced")

    def add_frame_subtracted(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_subtracted, frame_type="subtracted")

    def add_frame_registered(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_registered, frame_type="registered")

    def add_frame_astrometry(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_astrometry, frame_type="astrometry")

    def add_frame_diagnosed(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_diagnosed, frame_type="diagnosed")

    def add_frame_normalised(self, frame: Union[str, image.ImagingImage]):
        return self._add_frame(frame=frame, frames_dict=self.frames_normalised, frame_type="reduced")

    def add_coadded_trimmed_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_trimmed)

    def add_coadded_unprojected_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_unprojected)

    def add_coadded_subtracted_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_subtracted)

    def add_coadded_subtracted_trimmed_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_subtracted_trimmed)

    def add_coadded_subtracted_patch_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_subtracted_patch)

    def add_coadded_astrometry_image(self, img: Union[str, image.Image], key: str, **kwargs):
        return self._add_coadded(img=img, key=key, image_dict=self.coadded_astrometry)

    def check_filter(self, fil: str):
        """
        If a filter name is not present in the various lists and dictionaries that use it, adds it.
        :param fil:
        :return: False if None, True if not.
        """
        if fil not in (None, "", " "):
            if fil not in self.filters:
                if not self.quiet:
                    print(f"Adding {fil} to filter list")
                self.filters.append(fil)
            if fil not in self.astrometry_successful:
                self.astrometry_successful[fil] = {}
            if fil not in self.frames_standard:
                if isinstance(self.frames_standard, dict):
                    self.frames_standard[fil] = []
            if fil not in self.frames_flat:
                if isinstance(self.frames_flat, dict):
                    self.frames_flat[fil] = []
            if fil not in self.frames_science:
                if isinstance(self.frames_science, dict):
                    self.frames_science[fil] = []
            if fil not in self.frames_reduced:
                if isinstance(self.frames_reduced, dict):
                    self.frames_reduced[fil] = []
            if fil not in self.frames_normalised:
                if isinstance(self.frames_normalised, dict):
                    self.frames_normalised[fil] = []
            if fil not in self.frames_subtracted:
                if isinstance(self.frames_subtracted, dict):
                    self.frames_subtracted[fil] = []
            if fil not in self.frames_registered:
                if isinstance(self.frames_registered, dict):
                    self.frames_registered[fil] = []
            if fil not in self.frames_diagnosed:
                if isinstance(self.frames_diagnosed, dict):
                    self.frames_diagnosed[fil] = []
            if fil not in self.frames_astrometry:
                self.frames_astrometry[fil] = []
            if fil not in self.coadded:
                self.coadded[fil] = None
            if fil not in self.coadded_trimmed:
                self.coadded_trimmed[fil] = None
            if fil not in self.coadded_unprojected:
                self.coadded_unprojected[fil] = None
            if fil not in self.coadded_subtracted:
                self.coadded_subtracted[fil] = None
            if fil not in self.coadded_subtracted_patch:
                self.coadded_subtracted_patch[fil] = None
            if fil not in self.coadded_subtracted_trimmed:
                self.coadded_subtracted_trimmed[fil] = None
            if fil not in self.coadded_astrometry:
                self.coadded_astrometry[fil] = None
            if fil not in self.exp_time_mean:
                self.exp_time_mean[fil] = None
            if fil not in self.exp_time_err:
                self.exp_time_err[fil] = None
            if fil not in self.airmass_mean:
                self.airmass_mean[fil] = None
            if fil not in self.airmass_err:
                self.airmass_err[fil] = None
            if fil not in self.astrometry_stats:
                self.astrometry_stats[fil] = {}
            if fil not in self.frame_stats:
                self.frame_stats[fil] = {}
            if fil not in self.fringe_maps:
                self.fringe_maps[fil] = {}
            return True
        else:
            return False

    def plot_object(
            self, img: str,
            fil: str,
            fig: plt.Figure,
            centre: SkyCoord,
            frame: units.Quantity = 30 * units.pix,
            n: int = 1, n_x: int = 1, n_y: int = 1,
            cmap: str = 'viridis', show_cbar: bool = False,
            stretch: str = 'sqrt',
            vmin: float = None,
            vmax: float = None,
            show_grid: bool = False,
            ticks: int = None, interval: str = 'minmax',
            show_coords: bool = True,
            font_size: int = 12,
            reverse_y=False,
            **kwargs):
        if img == "coadded":
            u.debug_print(1, self.name, type(self))
            u.debug_print(1, self.coadded)
            to_plot = self.coadded[fil]
        else:
            raise ValueError(f"img type {img} not recognised.")

        u.debug_print(1, f"PIXEL SCALE: {to_plot.extract_pixel_scale()}")

        subplot, hdu_cut = to_plot.plot_subimage(
            fig=fig, frame=frame,
            centre=centre,
            n=n, n_x=n_x, n_y=n_y,
            cmap=cmap, show_cbar=show_cbar, stretch=stretch,
            vmin=vmin, vmax=vmax,
            show_grid=show_grid,
            ticks=ticks, interval=interval,
            show_coords=show_coords,
            font_size=font_size,
            reverse_y=reverse_y,
            **kwargs
        )
        return subplot, hdu_cut

    def push_to_table(self):

        from craftutils.observation.output.epoch import imaging_table
        imaging_table.load_table()

        # frames = self._get_frames("final")
        coadded = self._get_images("final")

        for fil in self.filters:
            img = coadded[fil]

            inttime = coadded[fil].extract_header_item("INTTIME") * units.second
            n_frames = self.n_frames(fil)
            if self.exp_time_mean[fil] is None:
                final_frames = self._get_frames("final")
                exp_times = list(map(lambda frame: frame.extract_exposure_time().value, final_frames[fil]))
                self.exp_time_mean[fil] = np.mean(exp_times) * units.s
            frame_exp_time = self.exp_time_mean[fil].round()

            depth = img.select_depth()

            entry = {
                "field_name": self.field.name,
                "epoch_name": self.name,
                "date_utc": self.date_str(),
                "mjd": self.mjd() * units.day,
                "instrument": self.instrument_name,
                "filter_name": fil,
                "filter_lambda_eff": self.instrument.filters[fil].lambda_eff.to(units.Angstrom).round(3),
                "n_frames": n_frames,
                "n_frames_included": coadded[fil].extract_ncombine(),
                "frame_exp_time": frame_exp_time,
                "total_exp_time": n_frames * frame_exp_time,
                "total_exp_time_included": inttime,
                "psf_fwhm": self.psf_stats[fil]["gauss"]["fwhm_median"],
                "program_id": str(self.program_id),
                "zeropoint": coadded[fil].zeropoint_best["zeropoint_img"],
                "zeropoint_err": coadded[fil].zeropoint_best["zeropoint_img_err"],
                "zeropoint_source": coadded[fil].zeropoint_best["catalogue"],
                "last_processed": Time.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "depth": depth
            }

            if isinstance(self.field, fld.FRBField) and self.field.frb.tns_name is not None:
                entry["transient_tns_name"] = self.field.frb.tns_name

            imaging_table.add_epoch(
                epoch_name=self.name,
                fil_name=fil,
                entry=entry
            )

        imaging_table.write_table()

    @classmethod
    def from_params(
            cls,
            name: str,
            instrument: str,
            field: Union['fld.Field', str] = None,
            quiet: bool = False
    ):
        if name in active_epochs:
            return active_epochs[name]
        instrument = instrument.lower()
        field_name, field = cls._from_params_setup(name=name, field=field)
        path = cls.build_param_path(
            instrument_name=instrument,
            field_name=field_name,
            epoch_name=name)
        return cls.from_file(param_file=path, field=field, quiet=quiet)

    @classmethod
    def build_param_path(cls, instrument_name: str, field_name: str, epoch_name: str):
        path = u.mkdir_check_args(p.param_dir, "fields", field_name, "imaging", instrument_name)
        return os.path.join(path, f"{epoch_name}.yaml")

    @classmethod
    def build_data_path_absolute(
            cls,
            field: 'fld.Field',
            instrument_name: str,
            name: str,
            date: Time = None
    ):
        date = u.check_time(date)
        if date is not None:
            name_str = f"{date.strftime('%Y-%m-%d')}-{name}"
        else:
            name_str = name

        return u.mkdir_check_args(field.data_path, "imaging", instrument_name, name_str)

    @classmethod
    def from_file(
            cls,
            param_file: Union[str, dict],
            field: 'fld.Field' = None,
            quiet: bool = False
    ):
        if not quiet:
            print("Initializing epoch...")

        name, param_file, param_dict = p.params_init(param_file)

        if param_dict is None:
            raise FileNotFoundError(f"There is no param file at {param_file}")

        pdict_backup = param_dict.copy()

        instrument = param_dict.pop("instrument").lower()

        fld_from_dict = param_dict.pop("field")
        if field is None:
            field = fld_from_dict
        # else:
        #     param_dict.pop("field")

        data_path = None
        if "data_path" in param_dict:
            data_path = param_dict.pop('data_path')
        if not data_path:
            data_path = cls.build_data_path_absolute(
                field=field,
                instrument_name=instrument,
                name=name,
                date=param_dict["date"]
            )

        p.join_data_dir(data_path)

        if "name" in param_dict:
            param_dict.pop("name")
        if "param_path" in param_dict:
            param_dict.pop("param_path")

        sub_cls = cls.select_child_class(instrument=instrument)
        u.debug_print(1, sub_cls)
        return sub_cls(
            name=name,
            field=field,
            param_path=param_file,
            data_path=data_path,
            instrument=instrument,
            date=param_dict.pop('date'),
            program_id=param_dict.pop("program_id"),
            target=param_dict.pop("target"),
            source_extractor_config=param_dict.pop('sextractor'),
            quiet=quiet,
            **param_dict
        )

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        default_params.update({
            "sextractor":
                {
                    "dual_mode": False,
                    "threshold": 1.5,
                    "kron_factor": 2.5,
                    "kron_radius_min": 3.5
                },
            # "background_subtraction":
            #     {"renormalise_centre": objects.position_dictionary.copy(),
            #      "test_synths":
            #          [{"position": objects.position_dictionary.copy(),
            #            "mags": {}
            #            }]
            #
            #      },
        })

        return default_params

    @classmethod
    def select_child_class(cls, instrument: str):
        instrument = instrument.lower()
        if instrument == "vlt-fors2":
            from .eso import FORS2ImagingEpoch
            child_class = FORS2ImagingEpoch
        elif instrument == "vlt-hawki":
            from .eso import HAWKIImagingEpoch
            child_class = HAWKIImagingEpoch
        elif instrument == "panstarrs1":
            from .survey import PanSTARRS1ImagingEpoch
            child_class = PanSTARRS1ImagingEpoch
        elif instrument == "gs-aoi":
            from .gemini import GSAOIImagingEpoch
            child_class = GSAOIImagingEpoch
        elif instrument in ["hst-wfc3_ir", "hst-wfc3_uvis2"]:
            from .hubble import HubbleImagingEpoch
            child_class = HubbleImagingEpoch
        elif instrument == "decam":
            from .survey import DESEpoch
            child_class = DESEpoch
        # elif instrument in p.instruments_imaging:
        else:
            child_class = ImagingEpoch
        # else:
        # raise ValueError(f"Unrecognised instrument {instrument}")
        u.debug_print(2, f"field.select_child_class(): instrument ==", instrument, "child_class ==", child_class)
        return child_class


def best_for_path(
        image_dict: Dict[str, image.ImagingImage],
        exclude: list = ()
):
    from craftutils.observation.filters import best_for_path
    filter_list = list(map(lambda k: image_dict[k].name(), image_dict))
    best_fil = best_for_path(filter_list, exclude=exclude)
    best_img = image_dict[best_fil.name]
    print(f"Best image for PATH is {best_img.name}")
    return best_img
