import os
import shutil
from typing import Union, List, Dict

import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
import astropy.table as table
import astropy.units as units
from astropy.modeling import models, fitting

import craftutils.utils as u
import craftutils.params as p
import craftutils.retrieve as retrieve
import craftutils.astrometry as astm
import craftutils.observation.image as image
from craftutils.observation.filters import FORS2Filter, Filter
from ..epoch import ESOImagingEpoch, ImagingEpoch
from .std import FORS2StandardEpoch

if p.data_dir:
    zeropoint_yaml = os.path.join(p.data_dir, f"zeropoints.yaml")


class FORS2ImagingEpoch(ESOImagingEpoch):
    instrument_name = "vlt-fors2"
    frame_class = image.FORS2Image
    coadded_class = image.FORS2CoaddedImage
    eso_name = "FORS2"

    def n_frames(self, fil: str):
        frame_pairs = self.pair_files(self.frames_reduced[fil])
        return len(frame_pairs)

    @classmethod
    def stages(cls):

        eso_stages = super().stages()
        ie_stages = ImagingEpoch.stages()

        stages = {
            "download": eso_stages["download"],
            "initial_setup": eso_stages["initial_setup"],
            "sort_reduced": eso_stages["sort_reduced"],
            "trim_reduced": eso_stages["trim_reduced"],
            "convert_to_cs": eso_stages["convert_to_cs"],
            # "defringe": ie_stages["defringe"],
            "register_frames": ie_stages["register_frames"],
            "correct_astrometry_frames": ie_stages["correct_astrometry_frames"],
            "frame_diagnostics": ie_stages["frame_diagnostics"],
            "insert_synthetic_frames": ie_stages["insert_synthetic_frames"],
            "subtract_background_frames": ie_stages["subtract_background_frames"],
            "coadd": ie_stages["coadd"],
            "correct_astrometry_coadded": ie_stages["correct_astrometry_coadded"],
            "trim_coadded": ie_stages["trim_coadded"],
            "source_extraction": ie_stages["source_extraction"],
            "photometric_calibration": ie_stages["photometric_calibration"],
            "dual_mode_source_extraction": ie_stages["dual_mode_source_extraction"],
            # "validate_photometry": ie_stages["check_validation"],
            "finalise": ie_stages["finalise"],
            "get_photometry": ie_stages["get_photometry"],
            # "get_photometry_all": ie_stages["get_photometry_all"]
        }

        # stages["defringe"]["default"] = True
        stages["photometric_calibration"]["keywords"]["skip_retrievable"] = True

        u.debug_print(2, f"FORS2ImagingEpoch.stages(): stages ==", stages)
        return stages

    def _pipeline_init(self, skip_cats: bool = False):
        super()._pipeline_init(skip_cats=skip_cats)
        self.frames_final = "astrometry"
        # If told not to correct astrometry on frames:
        if not self.combined_epoch and (
                "correct_astrometry_frames" in self.do_param and not self.do_param["correct_astrometry_frames"]):
            self.frames_final = "normalised"
            # If told to register frames
            if "register_frames" in self.do_param and self.do_param["register_frames"]:
                self.frames_final = "registered"
            if "frame_diagnostics" in self.do_param and self.do_param["frame_diagnostics"]:
                self.frames_final = "diagnosed"

        self.coadded_final = "coadded_trimmed"

    # def _register(self, frames: dict, fil: str, tmp: image.ImagingImage, n_template: int, output_dir: str, **kwargs):
    #     pairs = self.pair_files(images=frames[fil])
    #     if n_template >= 0:
    #         tmp = pairs[n_template]
    #
    #     for i, pair in enumerate(pairs):
    #         if not isinstance(pair, tuple):
    #             pair = [pair]
    #         if i != n_template:
    #             for j, frame in enumerate(pair):
    #                 if isinstance(tmp, tuple):
    #                     template = tmp[j]
    #                 else:
    #                     template = tmp
    #                 u.debug_print(2, frame.filename.replace("_norm.fits", "_registered.fits"))
    #                 registered = frame.register(
    #                     target=template,
    #                     output_path=os.path.join(
    #                         output_dir,
    #                         frame.filename.replace("_norm.fits", "_registered.fits"))
    #                 )
    #                 self.add_frame_registered(registered)
    #         else:
    #             for j, frame in enumerate(pair):
    #                 registered = frame.copy(
    #                     os.path.join(output_dir, frame.filename.replace("_norm.fits", "_registered.fits")))
    #                 self.add_frame_registered(registered)

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
        # List directories within 'reduction date' directories.
        # These should represent individual images reduced.

        _, subdirectory = os.path.split(subpath)

        # Get the files within the image directory.
        files = filter(
            lambda d: os.path.isfile(os.path.join(subpath, d)),
            os.listdir(subpath)
        )
        for file_name in files:
            # Retrieve the target object name from the fits file.
            file_path = os.path.join(subpath, file_name)
            inst_file = image.detect_instrument(file_path, fail_quietly=True)
            if inst_file != "vlt-fors2":
                continue
            file = image.from_path(
                path=file_path,
                cls=image.FORS2Image
            )
            file_obj = file.extract_object().lower()
            file_mjd = int(file.extract_header_item('MJD-OBS'))
            file_filter = file.extract_filter()
            # Check the object name and observation date against those of the epoch we're concerned with.
            if file_obj == obj and file_mjd == mjd:
                # Check which type of file we have.
                if file_name.endswith("PHOT_BACKGROUND_SCI_IMG.fits"):
                    file_destination = os.path.join(output_dir, "backgrounds")
                    suffix = "PHOT_BACKGROUND_SCI_IMG.fits"
                    file_type = "background"
                elif file_name.endswith("OBJECT_TABLE_SCI_IMG.fits"):
                    file_destination = os.path.join(output_dir, "obj_tbls")
                    suffix = "OBJECT_TABLE_SCI_IMG.fits"
                    file_type = "object_table"
                elif file_name.endswith("SCIENCE_REDUCED_IMG.fits"):
                    file_destination = os.path.join(output_dir, "science")
                    suffix = "SCIENCE_REDUCED_IMG.fits"
                    file_type = "science"
                else:
                    file_destination = os.path.join(output_dir, "sources")
                    suffix = "SOURCES_SCI_IMG.fits"
                    file_type = "sources"
                # Make this directory, if it doesn't already exist.
                u.mkdir_check(file_destination)
                # Make a subdirectory by filter.
                file_destination = os.path.join(file_destination, file_filter)
                u.mkdir_check(file_destination)
                # Title new file.
                file_destination = os.path.join(
                    file_destination,
                    f"{self.name}_{subdirectory}_{suffix}"
                )
                # Copy file to new location.
                if not self.quiet:
                    print(f"Copying: {file_path} to \n\t {file_destination}")
                file.copy(file_destination)
                if delete_output and os.path.isfile(file_destination):
                    os.remove(file_path)
                img = image.from_path(
                    path=file_destination,
                    cls=image.FORS2Image
                )
                u.debug_print(2, "ESOImagingEpoch._sort_after_esoreflex(): file_type ==", file_type)
                if file_type == "science":
                    self.add_frame_reduced(img)
                elif file_type == "background":
                    self.add_frame_background(img)
        # With the FORS2 substructure we want to search every subdirectory
        return False

    def generate_fringe_map(
            self,
            fil: str,
            output_dir: str = None,
            force: bool = False,
            frame_type: str = "frames_normalised"
    ):

        # self.retrieve_extra_standards(output_dir=os.path.join(output_dir, "standards"), fil=fil)
        # std_epochs = self.build_standard_epochs(
        #     image_dict=self.frames_standard_extra,
        #     pointings_dict={},
        #     epochs_dict={}
        # )
        return super().generate_fringe_map(
            fil=fil,
            output_dir=output_dir,
            force=force,
        )

    def retrieve_extra_standards(
            self,
            output_dir: str,
            fil: str
    ):
        self.frames_standard_extra[fil] = []
        u.mkdir_check_nested(output_dir, remove_last=False)
        instrument = self.instrument_name.split('-')[-1]
        mode = "imaging"
        r = retrieve.save_eso_raw_data_and_calibs(
            output=output_dir,
            date_obs=self.date,
            instrument=instrument,
            mode=mode,
            fil=fil,
            data_type="standard"
        )
        if r:
            os.system(f"uncompress {output_dir}/*.Z -f")
        for file in filter(lambda f: f.endswith(".fits"), os.listdir(output_dir)):
            img = image.FORS2Image(file, frame_type="standard")
            self.frames_standard_extra[fil].append(img)

        return

    def correct_astrometry_frames(
            self,
            output_dir: str,
            frames: dict = None,
            **kwargs
    ):
        """
        Uses `astrometry.net` with Gaia DR3 indices to correct the WCS of a set of individual exposures associated with
        this epoch.

        :param output_dir:
        :param frames: A "frames" dictionary ({"filter_name": [image_objects]}) containing the images to solve.
        :param kwargs:
            method: method with which to solve astrometry of epoch. Allowed values are:
                individual: each frame, including separate chips in the same exposure, will be passed to astrometry.net
                    individually. Of the options, this is the most likely to result in an error, especially if the FOV
                    is small; it will also slightly degrade the PSF of the stacked image, as although the accuracy of
                    the WCS of the individual frames is increased, slight errors will be introduced between frames.
                pairwise: the upper-chip image of each pair will first be passed to astrometry.net, and its solution
                    propagated to the bottom chip. If a solution is not found for the top chip, the reverse will be
                    attempted. This method is not recommended, as it will incorrectly capture distortions in the
                    unsolved chip.
                propagate_from_single: Each upper-chip image is passed to astrometry.net until a solution is found; this
                    solution is then propagated to all other upper-chip images. The same is repeated for the lower chip.
                    THIS CAN GIVE COARSE RESULTS ONLY.
        :return:
        """
        self.frames_astrometry = {}
        method = "individual"
        if "method" in kwargs:
            method = kwargs.pop("method")
        upper_only = False
        if "upper_only" in kwargs:
            upper_only = kwargs.pop("upper_only")
        if upper_only and method == "pairwise":
            method = "individual"
        if frames is None:
            frames = self.frames_normalised
        if not self.quiet:
            print()
            print(f"Solving astrometry using method '{method}'")
            print()

        if method == "individual":

            if upper_only:
                frames_upper = {}
                for fil in frames:
                    frames_upper[fil] = []
                    for img in frames[fil]:
                        if img.extract_chip_number() == 1:
                            frames_upper[fil].append(img)
                frames = frames_upper
            super().correct_astrometry_frames(output_dir=output_dir, frames=frames, **kwargs)

        else:
            for fil in frames:
                astrometry_fil_path = os.path.join(output_dir, fil)
                if method == "pairwise":
                    pairs = self.pair_files(frames[fil])
                    reverse_pair = False
                    for pair in pairs:
                        if isinstance(pair, tuple):
                            img_1, img_2 = pair
                            success = False
                            failed_first = False
                            while not success:  # The SystemError should stop this from looping indefinitely.
                                if not reverse_pair:
                                    new_img_1 = img_1.correct_astrometry(
                                        output_dir=astrometry_fil_path,
                                        **kwargs)
                                    # Check if the first astrometry run was successful.
                                    # If it wasn't, we need to be running on the second image of the pair.
                                    if new_img_1 is None:
                                        reverse_pair = True
                                        failed_first = True
                                        self.astrometry_successful[fil][img_1.name] = False
                                        if not self.quiet:
                                            print(
                                                f"Astrometry.net failed to solve {img_1}, trying on opposite chip {img_2}.")
                                    else:
                                        self.add_frame_astrometry(new_img_1)
                                        self.astrometry_successful[fil][img_1.name] = True
                                        new_img_2 = img_2.correct_astrometry_from_other(
                                            new_img_1,
                                            output_dir=astrometry_fil_path,
                                        )
                                        self.add_frame_astrometry(new_img_2)

                                        success = True
                                # We don't use an else statement here because reverse_pair can change within the above
                                # block, and if it does the block below needs to execute.
                                if reverse_pair:
                                    new_img_2 = img_2.correct_astrometry(
                                        output_dir=astrometry_fil_path,
                                        **kwargs)
                                    if new_img_2 is None:
                                        self.astrometry_successful[fil][img_2.name] = False
                                        if failed_first:
                                            raise SystemError(
                                                f"Astrometry.net failed to solve both chips of this pair ({img_1}, {img_2})")
                                        else:
                                            reverse_pair = False
                                    else:
                                        self.add_frame_astrometry(new_img_2)
                                        self.astrometry_successful[fil][img_2.name] = True
                                        new_img_1 = img_1.correct_astrometry_from_other(
                                            new_img_2,
                                            output_dir=astrometry_fil_path,
                                        )
                                        self.add_frame_astrometry(new_img_1)
                                        success = True
                        else:
                            new_img = pair.correct_astrometry(
                                output_dir=astrometry_fil_path,
                                **kwargs)
                            self.add_frame_astrometry(new_img)

                        self.update_output_file()

                elif method == "propagate_from_single":
                    # Sort frames by upper or lower chip.
                    chips = self.sort_by_chip(frames[fil])
                    upper = chips[1]
                    lower = chips[2]
                    if upper_only:
                        lower = []
                    for j, lst in enumerate((upper, lower)):
                        successful = None
                        i = 0
                        while successful is None and i < len(upper):
                            img = lst[i]
                            i += 1
                            new_img = img.correct_astrometry(output_dir=astrometry_fil_path,
                                                             **kwargs)
                            # Check if successful:
                            if new_img is not None:
                                lst.remove(img)
                                self.add_frame_astrometry(new_img)
                                successful = new_img
                                self.astrometry_successful[fil][img.name] = True
                            else:
                                self.astrometry_successful[fil][img.name] = False

                        # If we failed to find a solution on any frame in lst:
                        if successful is None and not self.quiet:
                            print(
                                f"Astrometry.net failed to solve any of the chip {j + 1} images. "
                                f"Chip 2 will not be included in the co-addition.")

                        # Now correct all of the other images in the list with the successful solution.
                        else:
                            for img in lst:
                                new_img = img.correct_astrometry_from_other(
                                    successful,
                                    output_dir=astrometry_fil_path
                                )

                                self.add_frame_astrometry(new_img)

                        self.update_output_file()

                else:
                    raise ValueError(
                        f"Astrometry method {method} not recognised. Must be individual, pairwise or propagate_from_single")

    def estimate_atmospheric_extinction(
            self,
            n: int = 10,
            output: str = None
    ):
        mjd = self.date.mjd
        fils_known = []
        tbls_known = {}

        fils_find = []

        for fil_name in FORS2Filter.qc1_retrievable:
            fil = Filter.from_params(fil_name, instrument_name="vlt-fors2")
            fil.retrieve_calibration_table()
            fils_known.append(fil)
            tbls_known[fil_name] = fil.get_nearest_calib_rows(mjd=mjd, n=n)

        fils_known.sort(key=lambda f: f.lambda_eff)

        lambdas_known = list(map(lambda f: f.lambda_eff.value, fils_known))

        results_tbl = {
            "mjd": [],
            "curve_err": [],
        }

        for fil_name in self.filters:
            if fil_name not in FORS2Filter.qc1_retrievable:
                fil = Filter.from_params(fil_name, instrument_name="vlt-fors2")
                fils_find.append(fil)
                results_tbl[f"ext_{fil_name}"] = []
                # results_tbl[f"ext_err_{fil_name}"] = []
                results_tbl[f"stat_err_{fil_name}"] = []

        fils_find.sort(key=lambda f: f.lambda_eff)
        lambdas_find = list(map(lambda f: f.lambda_eff.value, fils_find))

        if output is None:
            output = self.data_path

        for i in range(n):
            extinctions_known = []
            extinctions_known_err = []
            mjd = None
            mjds = []
            for fil in fils_known:
                tbl = tbls_known[fil.name]
                if mjd is None:
                    mjd = tbl[i]["mjd_obs"]
                mjds.append(tbl[i]["mjd_obs"])
                extinctions_known.append(tbl[i]["extinction"].value)
                extinctions_known_err.append(tbl[i]["extinction_err"].value)
            results_tbl["mjd"].append(mjd)
            extinctions_known_err = np.array(extinctions_known_err)
            model_init = models.PowerLaw1D()
            fitter = fitting.LevMarLSQFitter()

            try:
                model = fitter(model_init, np.array(lambdas_known), np.array(extinctions_known),
                               weights=1 / extinctions_known_err)
                curve_err = u.root_mean_squared_error(model_values=model(lambdas_known), obs_values=extinctions_known)
                results_tbl["curve_err"].append(curve_err)
                extinctions_find = model(lambdas_find)
                lambda_eff_fit = np.linspace(3000, 10000)
                plt.close()
                plt.plot(lambda_eff_fit, model(lambda_eff_fit))
                plt.scatter(lambdas_known, extinctions_known, label="Known")
                for j, m in enumerate(mjds):
                    plt.text(lambdas_known[j], extinctions_known[j], fils_known[j])
                plt.scatter(lambdas_find, extinctions_find, label="fitted")
                plt.xlabel("$\lambda_{eff}$ (Ang)")
                plt.ylabel("Extinction (mag)")
                try:
                    plt.savefig(os.path.join(output, f"extinction_fit_mjd_{mjd}.png"))
                except TypeError:
                    pass
                plt.close()

                for fil in fils_find:
                    results_tbl[f"ext_{fil.name}"].append(model(fil.lambda_eff.value))

            except fitting.NonFiniteValueError:
                print("Fitting failed for MJD", mjd)
                results_tbl["curve_err"].append(np.nan)
                for fil in fils_find:
                    results_tbl[f"ext_{fil.name}"].append(np.nan)

        for fil in fils_find:
            results_tbl[f"stat_err_{fil.name}"] = [np.std(results_tbl[f"ext_{fil.name}"])] * n

        results_tbl = table.QTable(results_tbl)
        for fil in fils_find:
            results_tbl[f"ext_err_{fil.name}"] = np.sqrt(
                results_tbl[f"stat_err_{fil.name}"] ** 2 + results_tbl[f"curve_err"] ** 2) * units.mag
            results_tbl[f"stat_err_{fil.name}"] *= units.mag
            results_tbl[f"ext_{fil.name}"] *= units.mag
        results_tbl[f"curve_err"] *= units.mag

        i, nrst = u.find_nearest(results_tbl["mjd"], self.date.mjd)

        results_tbl.write(os.path.join(output, "fitted_extinction.csv"), format="ascii.csv")

        return results_tbl[i], results_tbl

    def generate_master_biases(self, output_dir: str = None, force: bool = False):
        if output_dir is None:
            output_dir = self.processed_calib_dir()
        from craftutils.wrap import esorex
        # Split up bias images by chip
        print("Bias frames:")
        for frame in self.frames_bias:
            print(f"\t{frame.name}")
        if force:
            self.master_biases = {}
        bias_sets = self.sort_by_chip(self.frames_bias)
        for chip, bias_set in bias_sets.items():
            if chip not in self.master_biases:
                # For each chip, generate a master bias image
                try:
                    master_bias = esorex.fors_bias(
                        bias_frames=list(map(lambda b: b.path, bias_set)),
                        output_dir=output_dir,
                        output_filename=f"master_bias_{chip}.fits",
                        sof_name=f"bias_{chip}.sof"
                    )
                    self.master_biases[chip] = master_bias
                except SystemError:
                    continue
        return self.master_biases

    def generate_master_flats(
            self,
            output_dir: str = None,
            force: bool = False,
            skip_retrievable: bool = False
    ):
        from craftutils.wrap import esorex

        if not self.master_biases:
            self.generate_master_biases()

        if force:
            self.master_flats = {}

        if output_dir is None:
            output_dir = self.processed_calib_dir()
        for fil in self.filters:
            flat_chips = self.sort_by_chip(self.frames_flat[fil])
            for chip, flat_set in flat_chips.items():
                # For each chip, generate a master flat
                if chip not in self.master_flats:
                    self.master_flats[chip] = {}
                if fil not in self.master_flats[chip]:
                    master_bias = self.get_master_bias(chip)
                    try:
                        master_sky_flat_img = esorex.fors_img_sky_flat(
                            flat_frames=list(map(lambda b: b.path, flat_set)),
                            master_bias=master_bias,
                            output_dir=output_dir,
                            output_filename=f"master_sky_flat_img_{chip}_{fil}.fits",
                            sof_name=f"flat_chip_{chip}_{fil}"
                        )
                        self.master_flats[chip][fil] = master_sky_flat_img
                    except SystemError:
                        continue

        return self.master_flats

    def processed_calib_dir(self):
        path = os.path.join(self.data_path, "processed_calibs")
        u.mkdir_check_nested(path, False)
        return path

    def build_standard_epochs(
            self,
            image_dict: dict = None,
            pointings_dict: dict = None,
            epochs_dict: dict = None,
            skip_retrievable: bool = False,
            output_dir: str = None
    ):
        if image_dict is None:
            image_dict = self.frames_standard
        if pointings_dict is None:
            pointings_dict = self.std_pointings
        if epochs_dict is None:
            epochs_dict = self.std_epochs

        self.generate_master_biases()
        self.generate_master_flats(skip_retrievable=skip_retrievable)

        for fil, images in image_dict.items():
            if skip_retrievable and fil in FORS2Filter.qc1_retrievable:
                continue
            for std in images:
                print("\nProcessing std", std.name)
                # generate or load an appropriate StandardEpoch
                # (and StandardField in the background)
                pointing = std.extract_pointing()
                jname = astm.jname(pointing, 0, 0)
                if pointing not in pointings_dict:
                    pointings_dict.append(pointing)
                    print("\tAdding pointing to std_pointings:", pointing)
                if jname not in epochs_dict:
                    print(f"\t{jname=} not found in std_epochs; generating epoch.")
                    std_epoch = FORS2StandardEpoch(
                        centre_coords=pointing,
                        instrument=self.instrument,
                        frames_flat=self.frames_flat,
                        frames_bias=self.frames_bias,
                        date=self.date
                    )
                    epochs_dict[jname] = std_epoch
                    std_epoch.master_flats = self.master_flats.copy()
                    std_epoch.master_biases = self.master_biases.copy()
                else:
                    print(f"\t{jname=} found in std_epochs; retrieving.")
                    std_epoch = epochs_dict[jname]
                print(f"\tAdding raw standard to std_epoch")
                std_epoch.add_frame_raw(std)

        return epochs_dict

    def photometric_calibration_from_standards(
            self,
            image_dict: dict,
            output_path: str,
            skip_retrievable: bool = False
    ):

        import craftutils.wrap.esorex as esorex

        ext_row, ext_tbl = self.estimate_atmospheric_extinction(output=output_path)
        # image_dict = self._get_images(image_type=image_type)
        for fil, img in image_dict.items():
            if f"ext_{fil}" in ext_row.colnames:
                img.extinction_atmospheric = ext_row[f"ext_{fil}"]
                img.extinction_atmospheric_err = ext_row[f"ext_err_{fil}"]

        # Do esorex reduction of standard images, and attempt esorex zeropoints if there are enough different
        # observations
        # image_dict = self._get_images(image_type)
        self.build_standard_epochs(
            output_dir=output_path,
            skip_retrievable=skip_retrievable
        )

        for fil in image_dict:
            if skip_retrievable and fil in FORS2Filter.qc1_retrievable:
                continue
            std_chips = self.sort_by_chip(self.frames_standard[fil])
            img = image_dict[fil]
            if "calib_pipeline" in img.zeropoints:
                img.zeropoints.pop("calib_pipeline")
            fil_dir = os.path.join(output_path, fil)
            u.mkdir_check(fil_dir)

        std_dir = os.path.join(self.data_path, "reduced_standards")

        aligned_phots = {}

        for jname, std_epoch in self.std_epochs.items():
            aligned_phots_epoch = std_epoch.reduce()
            for chip in aligned_phots_epoch:
                if chip not in aligned_phots:
                    aligned_phots[chip] = {}
                for fil in aligned_phots_epoch[chip]:
                    if fil not in aligned_phots[chip]:
                        aligned_phots[chip][fil] = []
                    aligned_phots[chip][fil] += aligned_phots_epoch[chip][fil]

        for chip in aligned_phots:
            chip_dir = os.path.join(output_path, f"chip_{chip}")
            u.mkdir_check(chip_dir)
            for fil in aligned_phots[chip]:
                if skip_retrievable and fil in FORS2Filter.qc1_retrievable:
                    continue
                fil_dir = os.path.join(chip_dir, fil)
                u.mkdir_check(fil_dir)
                aligned_phots_this = list(set(aligned_phots[chip][fil]))
                if len(aligned_phots_this) > 1:
                    try:
                        master_sky_flat_img = self.get_master_flat(chip=chip, fil=fil)
                        phot_coeff_table = esorex.fors_photometry(
                            aligned_phot=aligned_phots_this,
                            master_sky_flat_img=master_sky_flat_img,
                            output_dir=fil_dir,
                            # output_filename=f"{}",
                            chip_num=chip,
                        )

                        phot_coeff_table = fits.open(phot_coeff_table)[1].data

                        # The intention here is that a chip 1 zeropoint override a chip 2 zeropoint, but
                        # if chip 1 doesn't work a chip 2 one will do.

                        if fil in image_dict:
                            img = image_dict[fil]
                            if f"ext_{fil}" in ext_row.colnames:
                                img.extinction_atmospheric = ext_row[f"ext_{fil}"]
                                img.extinction_atmospheric_err = ext_row[f"ext_err_{fil}"]

                            if chip == 1 or "calib_pipeline" not in img.zeropoints:
                                img.add_zeropoint(
                                    zeropoint=phot_coeff_table["ZPOINT"][0] * units.mag,
                                    zeropoint_err=phot_coeff_table["DZPOINT"][0] * units.mag,
                                    airmass=img.extract_airmass(),
                                    airmass_err=self.airmass_err[fil],
                                    extinction=phot_coeff_table["EXT"][0] * units.mag,
                                    extinction_err=phot_coeff_table["DEXT"][0] * units.mag,
                                    catalogue="calib_pipeline",
                                    n_matches=None,
                                )

                        # img.update_output_file()
                    except SystemError:
                        if not self.quiet:
                            print(
                                "System error encountered while doing esorex processing; possibly impossible value encountered. Skipping.")

                else:
                    print(f"Insufficient standard observations to calculate esorex zeropoint for {img}")

        if not self.quiet:
            print("Estimating zeropoints from standard observations...")
        for jname in self.std_epochs:
            std_epoch = self.std_epochs[jname]
            std_epoch.photometric_calibration(skip_retrievable=skip_retrievable)
            for fil in image_dict:
                img = image_dict[fil]
                # We save time by only bothering with non-qc1-obtainable zeropoints.
                if fil in std_epoch.frames_reduced and not (skip_retrievable and img.filter.calib_retrievable()):
                    for std in std_epoch.frames_reduced[fil]:
                        img.add_zeropoint_from_other(std)

    def photometric_calibration(
            self,
            output_path: str,
            image_dict: dict,
            **kwargs
    ):

        # if "image_type" in kwargs and kwargs["image_type"] is not None:
        #     image_type = kwargs["image_type"]
        # else:
        #     image_type = "final"

        suppress_select = True
        if "suppress_select" in kwargs and kwargs["suppress_select"] is not None:
            suppress_select = kwargs.pop("suppress_select")
        skip_retrievable = False
        if "skip_retrievable" in kwargs and kwargs["skip_retrievable"] is not None:
            skip_retrievable = kwargs.pop("skip_retrievable")
        skip_standards = False
        if "skip_standards" in kwargs and kwargs["skip_standards"] is not None:
            skip_standards = kwargs.pop("skip_standards")

        if not self.combined_epoch and not skip_standards:
            self.photometric_calibration_from_standards(
                image_dict=image_dict,
                output_path=output_path,
                skip_retrievable=skip_retrievable
            )

        zeropoints = p.load_params(zeropoint_yaml)
        if zeropoints is None:
            zeropoints = {}

        super().photometric_calibration(
            output_path=output_path,
            suppress_select=True,
            image_dict=image_dict,
            **kwargs
        )

        for fil in image_dict:
            if "preferred_zeropoint" in kwargs and fil in kwargs["preferred_zeropoint"]:
                preferred = kwargs["preferred_zeropoint"][fil]
            else:
                preferred = None
            img = image_dict[fil]

            img.select_zeropoint(suppress_select, preferred=preferred)

            if fil not in zeropoints:
                zeropoints[fil] = {}
            for cat in img.zeropoints:
                if cat not in zeropoints[fil]:
                    zeropoints[fil][cat] = {}
                zeropoints[fil][cat][self.date_str()] = img.zeropoints[cat]

            # Transfer derived zeropoints to other versions (if they exist and are distinct)
            for coadded_dict in self.coadded_derivatives:
                if isinstance(coadded_dict[fil], image.CoaddedImage) and coadded_dict[fil] is not img:
                    coadded_dict[fil].clone_zeropoints(img)

        p.save_params(zeropoint_yaml, zeropoints)

    @classmethod
    def pair_files(cls, images: list):
        pairs = []
        images.sort(key=lambda im: im.name)
        is_paired = True
        for i, img_1 in enumerate(images):
            # If the images are in pairs, it's sufficient to check only the even-numbered ones.
            # If not, is_paired=False should be triggered by the case below.
            if i % 2 == 0 or not is_paired:
                chip_this = img_1.extract_chip_number()
                # If we are at the end of the list and still checking, this must be unpaired.
                if i + 1 == len(images):
                    pair = img_1
                else:
                    # Get the next image in the list.
                    img_2 = images[i + 1]
                    chip_other = img_2.extract_chip_number()
                    # If we have chip
                    if (chip_this == 1 and chip_other == 2) or (chip_this == 2 and chip_other == 1):
                        img_1.other_chip = img_2
                        img_1.update_output_file()
                        img_2.other_chip = img_1
                        img_2.update_output_file()
                        if chip_this == 1:
                            pair = (img_1, img_2)
                        elif chip_this == 2:
                            pair = (img_2, img_1)
                        else:
                            raise ValueError("Image is missing chip.")
                        is_paired = True
                    else:
                        is_paired = False
                        pair = img_1
                if isinstance(pair, tuple):
                    u.debug_print(1, str(pair[0]), ",", str(pair[1]))
                else:
                    u.debug_print(1, pair)
                pairs.append(pair)

        return pairs

    @classmethod
    def from_file(
            cls,
            param_file: Union[str, dict],
            name: str = None,
            field: 'fld.Field' = None
    ):
        return super().from_file(param_file=param_file, name=name, field=field)
