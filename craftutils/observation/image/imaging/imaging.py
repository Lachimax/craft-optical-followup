import os
import shutil
from typing import Union, Tuple, List
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

import astropy.units as units
import astropy.wcs as wcs
import astropy.table as table
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from astropy.visualization import quantity_support
from astropy.stats import SigmaClip
from astropy.modeling import models, fitting
from astropy.visualization import (
    ImageNormalize,
    LogStretch,
    SqrtStretch,
    MinMaxInterval,
    ZScaleInterval
)

from astroalign import register

try:
    import photutils
except ModuleNotFoundError:
    print("photutils not installed; some photometry-related functionality will be unavailable.")
try:
    import sep
except ModuleNotFoundError:
    print("sep not installed; some photometry-related functionality will be unavailable.")

import craftutils.utils as u
import craftutils.params as p
import craftutils.photometry as ph
import craftutils.fits_files as ff
import craftutils.astrometry as astm
import craftutils.plotting as pl
from craftutils.retrieve import cat_columns, cat_instruments
from craftutils.stats import gaussian_distributed_point

import craftutils.observation.catalogue as catalog
import craftutils.observation.objects as objects
import craftutils.observation.instrument as inst

import craftutils.wrap.source_extractor as se
import craftutils.wrap.psfex as psfex
import craftutils.wrap.galfit as galfit
from craftutils.wrap.astrometry_net import solve_field

from ..image import Image


# __all__ = []

# @u.export
class ImagingImage(Image):

    def __init__(
            self,
            path: str,
            frame_type: str = None,
            instrument_name: str = None,
            load_outputs: bool = True,
            **kwargs
    ):
        super().__init__(
            path=path,
            frame_type=frame_type,
            instrument_name=instrument_name,
            **kwargs
        )

        self.wcs = []

        self.filter_name = None
        self.filter_short = None
        self.filter = None
        self.pixel_scale_x = None
        self.pixel_scale_y = None

        self.psfex_path = None
        self.psfex_output = None
        self.psfex_successful = None
        self.source_cat = None
        self.source_cat_dual = None
        self.dual_mode_template = None

        self.sep_background = None
        self.pu_background = None
        self.data_sub_bkg = None

        self.synth_cat_path = None
        self.synth_cat = None

        self.fwhm_pix_psfex = None
        self.fwhm_psfex = None

        self.fwhm_max_moffat = None
        self.fwhm_median_moffat = None
        self.fwhm_min_moffat = None
        self.fwhm_sigma_moffat = None
        self.fwhm_rms_moffat = None

        self.psf_stats = {}

        self.fwhm_max_gauss = None
        self.fwhm_median_gauss = None
        self.fwhm_min_gauss = None
        self.fwhm_sigma_gauss = None
        self.fwhm_rms_gauss = None

        self.fwhm_max_sextractor = None
        self.fwhm_median_sextractor = None
        self.fwhm_min_sextractor = None
        self.fwhm_sigma_sextractor = None
        self.fwhm_rms_sextractor = None

        self.sky_background = None

        self.zeropoints = {}
        self.zeropoint_output_paths = {}
        self.zeropoint_best = None

        self.extinction_atmospheric = None
        self.extinction_atmospheric_err = None

        self.depth = {}

        self.astrometry_err = None
        self.ra_err = None
        self.dec_err = None
        self.astrometry_corrected_path = None
        self.astrometry_stats = {}

        self.extract_filter()

        if load_outputs:
            self.load_output_file()

    def source_extraction(
            self,
            configuration_file: str,
            output_dir: str,
            parameters_file: str = None,
            catalog_name: str = None,
            template: 'ImagingImage' = None,
            **configs
    ) -> str:
        if template is not None:
            template = template.path
            self.dual_mode_template = template
        self.extract_gain()
        u.debug_print(2, f"ImagingImage.source_extraction(): template ==", template)
        if not self.do_subtract_background():
            configs["BACK_TYPE"] = "MANUAL"
        output_path = se.source_extractor(
            image_path=self.path,
            output_dir=output_dir,
            configuration_file=configuration_file,
            parameters_file=parameters_file,
            catalog_name=catalog_name,
            template_image_path=template,
            gain=self.gain.value,
            **configs
        )
        self.add_log(
            action="Sources extracted using Source Extractor.",
            method=self.source_extraction,
            output_path=output_dir,
            packages=["source-extractor"]
        )
        self.update_output_file()
        return output_path

    def psfex(
            self,
            output_dir: str,
            force: bool = False,
            set_attributes: bool = True,
            se_kwargs: dict = {},
            **kwargs
    ):
        """
        Run PSFEx on this image to obtain a PSF model.
        :param output_dir: path to directory to write PSFEx outputs to.
        :param force: If False, and this object already has a PSF model, we just return the one that already exists.
        :param se_kwargs: arguments to pass to Source Extractor.
        :param kwargs: arguments to pass to PSFEx.
        :param set_attributes: If True, this Image's psfex_path, psfex_output, fwhm_pix_psfex and fwhm_psfex will be set
            according to the PSFEx output.
        :return: HDUList representing the PSF model FITS file.
        """
        psfex_output = None

        if force or self.psfex_path is None or not os.path.isfile(self.psfex_path):
            # Set up a list of photometric apertures to pass to SE as a string.
            _, scale = self.extract_pixel_scale()
            aper_arcsec = [
                              # 50,
                              4.87,
                              3.9,
                              2.92
                          ] * units.arcsec
            phot_aper = aper_arcsec.to(units.pix, scale).value
            phot_aper_str = ""
            for a in phot_aper:
                phot_aper_str += f"{a},"
            phot_aper_str = phot_aper_str[:-1]
            se_kwargs["PHOT_APERTURES"] = phot_aper_str
            kwargs["PHOTFLUX_KEY"] = '"FLUX_APER(1)"'
            kwargs["PHOTFLUXERR_KEY"] = '"FLUXERR_APER(1)"'

            config = p.path_to_config_sextractor_config_pre_psfex()
            output_params = p.path_to_config_sextractor_param_pre_psfex()
            catalogue = self.source_extraction(
                configuration_file=config,
                output_dir=output_dir,
                parameters_file=output_params,
                catalog_name=f"{self.name}_psfex.fits",
                **se_kwargs
            )

            psfex_path = psfex.psfex(
                catalog=catalogue,
                output_dir=output_dir,
                **kwargs
            )
            psfex_output = fits.open(psfex_path)

            if not psfex.check_successful(psfex_output):
                print(f"PSFEx did not converge. Retrying with PHOTFLUX_KEY==FLUX_AUTO")

                kwargs["PHOTFLUX_KEY"] = "FLUX_AUTO"
                kwargs["PHOTFLUXERR_KEY"] = "FLUXERR_AUTO"

                psfex_path = psfex.psfex(
                    catalog=catalogue,
                    output_dir=output_dir,
                    **kwargs
                )
                psfex_output = fits.open(psfex_path)

            i = 1
            while not psfex.check_successful(psfex_output) and i < len(aper_arcsec):
                print(f"PSFEx did not converge. Retrying with smaller PHOTFLUX apertures.")
                kwargs["PHOTFLUX_KEY"] = f'"FLUX_APER({i + 1})"'
                kwargs["PHOTFLUXERR_KEY"] = f'"FLUXERR_APER({i + 1})"'

                catalogue = self.source_extraction(
                    configuration_file=config,
                    output_dir=output_dir,
                    parameters_file=output_params,
                    catalog_name=f"{self.name}_psfex.fits",
                    **se_kwargs
                )

                psfex_path = psfex.psfex(
                    catalog=catalogue,
                    output_dir=output_dir,
                    **kwargs
                )

                i += 1

            if set_attributes:
                self.psfex_path = psfex_path
                self.extract_pixel_scale()
                pix_scale = self.pixel_scale_y
                self.fwhm_pix_psfex = psfex_output[1].header['PSF_FWHM'] * units.pixel
                self.fwhm_psfex = self.fwhm_pix_psfex.to(units.arcsec, pix_scale)

            self.add_log(
                action="PSF modelled using psfex.",
                method=self.psfex,
                output_path=output_dir,
                packages=["psfex"]
            )
            self.update_output_file()

        if set_attributes:
            return self.load_psfex_output()
        else:
            return psfex_output

    # def _psfex(self):

    def load_psfex_output(self, force: bool = False):
        if force or self.psfex_output is None:
            self.psfex_output = fits.open(self.psfex_path)
        return self.psfex_output

    def psf_image(self, x: float, y: float, match_pixel_scale: bool = True):
        if match_pixel_scale:
            return psfex.load_psfex(model_path=self.psfex_path, x=x, y=y)
        else:
            return psfex.load_psfex_oversampled(model=self.psfex_path, x=x, y=y)

    def clone_diagnostics(
            self,
            other: 'ImagingImage',
            ext: int = 0
    ):
        """
        The intent behind this function is to use it when it is a derived image that should have the same
        characteristics (eg PSF, astrometry, zeropoint) as the original or other derivative, and to transfer
        those properties across.

        :param other:
        :param ext:
        :return:
        """

        self.load_headers()
        other.load_headers()
        self.clone_psf(other=other, ext=ext)
        self.clone_zeropoints(other=other, ext=ext)
        self.clone_astrometry_info(other=other, ext=ext)

        self.update_output_file()
        self.write_fits_file()

    def clone_astrometry_info(
            self,
            other: 'ImagingImage',
            ext: int = 0
    ):
        self.astrometry_stats = other.astrometry_stats
        self.astrometry_err = other.astrometry_err
        self.ra_err = other.ra_err
        self.dec_err = other.dec_err
        astm_rms = other.extract_header_item(key="ASTM_RMS", accept_absent=True)
        if astm_rms:
            self.set_header_item(
                key="ASTM_RMS",
                value=astm_rms,
                ext=ext
            )
        ra_rms = other.extract_header_item(key="RA_RMS", accept_absent=True)
        if ra_rms:
            self.set_header_item(
                key="RA_RMS",
                value=ra_rms,
                ext=ext
            )
        dec_rms = other.extract_header_item(key="DEC_RMS", accept_absent=True)
        if dec_rms:
            self.set_header_item(
                key="DEC_RMS",
                value=dec_rms,
                ext=ext
            )

    def clone_zeropoints(
            self,
            other: 'ImagingImage',
            ext: int = 0
    ):
        self.zeropoints = other.zeropoints.copy()
        self.zeropoint_best = other.zeropoint_best
        zp = other.extract_header_item(key="ZP", accept_absent=True)
        if zp:
            self.set_header_item(
                key="ZP",
                value=zp,
                ext=ext
            )
        zp_err = other.extract_header_item(key="ZP_ERR", accept_absent=True)
        if zp_err:
            self.set_header_item(
                key="ZP_ERR",
                value=zp_err,
                ext=ext
            )
        zp_cat = other.extract_header_item(key="ZPCAT", accept_absent=True)
        if zp_cat:
            self.set_header_item(
                key="ZPCAT",
                value=zp_cat,
                ext=ext
            )

    def clone_psf(
            self,
            other: 'ImagingImage',
            ext: int = 0
    ):
        self.load_headers()

        self.psfex_path = other.psfex_path
        self.fwhm_pix_psfex = other.fwhm_pix_psfex
        self.fwhm_psfex = other.fwhm_psfex
        self.psfex_successful = other.psfex_successful
        self.psfex_output = other.psfex_output
        self.psf_stats = other.psf_stats

        self.fwhm_median_gauss = other.fwhm_median_gauss
        self.fwhm_max_gauss = other.fwhm_max_gauss
        self.fwhm_min_gauss = other.fwhm_min_gauss
        self.fwhm_sigma_gauss = other.fwhm_sigma_gauss
        self.fwhm_rms_gauss = other.fwhm_rms_gauss

        self.fwhm_median_moffat = other.fwhm_median_moffat
        self.fwhm_max_moffat = other.fwhm_max_moffat
        self.fwhm_min_moffat = other.fwhm_min_moffat
        self.fwhm_sigma_moffat = other.fwhm_sigma_moffat
        self.fwhm_rms_moffat = other.fwhm_rms_moffat

        psf_fwhm = other.extract_header_item(key="PSF_FWHM", ext=ext, accept_absent=True)
        if psf_fwhm:
            self.set_header_item(
                key="PSF_FWHM",
                value=psf_fwhm,
                ext=ext
            )
        psf_fwhm_err = other.extract_header_item(key="PSF_FWHM_ERR", ext=ext, accept_absent=True)
        if psf_fwhm_err:
            self.set_header_item(
                key="PSF_FWHM_ERR",
                value=psf_fwhm_err,
                ext=ext
            )

    def source_extraction_psf(
            self,
            output_dir: str,
            template: 'ImagingImage' = None,
            force: bool = False,
            **configs
    ):
        """
        Uses a PSFEx-generated PSF model in conjunction with Source Extractor to generate a source catalog. The key
        difference with source_extraction is that source_extraction uses only Source Extractor, and does not therefore
        use PSF-fitting (ie, no CLASS_STAR, MAG_PSF or FLUX_PSF columns are written).
        :param output_dir: The directory in which to write the PSFEx and Source Extractor output files.
        :param template: The path to the file to use as template, if dual mode is to be used.
        :param force: If True, performs all functions regardless of whether source_catalogues already exist, and
            overwrites them; if False, checks whether they exist first and skips some steps if so.
        :param configs: A dictionary of Source Extractor arguments to pass to command line.
        :return:
        """

        psf = self.psfex(
            output_dir=output_dir,
            force=force,
        )

        if psfex.check_successful(psf):
            cat_path = self.source_extraction(
                configuration_file=p.path_to_config_sextractor_config(),
                output_dir=output_dir,
                parameters_file=p.path_to_config_sextractor_param(),
                catalog_name=f"{self.name}_psf-fit.cat",
                psf_name=self.psfex_path,
                seeing_fwhm=self.fwhm_psfex.value,
                template=template,
                **configs
            )
        else:
            cat_path = self.source_extraction(
                configuration_file=p.path_to_config_sextractor_failed_psfex_config(),
                parameters_file=p.path_to_config_sextractor_failed_psfex_param(),
                output_dir=output_dir,
                catalog_name=f"{self.name}_failed-psf-fit.cat",
                template=template,
                **configs
            )
        dual = False
        if template is not None:
            dual = True

        cat = catalog.SECatalogue(se_path=cat_path, image=self)

        if len(cat) == 0:
            print()
            print("PSF source extraction was unsuccessful, probably due to lack of viable sources. Trying again without"
                  " PSFEx.")
            print()
            self.psfex_successful = False
            cat_path = self.source_extraction(
                configuration_file=p.path_to_config_sextractor_failed_psfex_config(),
                output_dir=output_dir,
                parameters_file=p.path_to_config_sextractor_failed_psfex_param(),
                catalog_name=f"{self.name}.cat",
                template=template,
                **configs
            )
            cat.set_se_path(cat_path, load=True)
        else:
            self.psfex_successful = True

        u.debug_print(2, "dual, template:", dual, template)

        if dual:
            self.source_cat_dual = cat
        else:
            self.source_cat = cat

        self.plot_apertures()
        self.add_log(
            action="Sources extracted using Source Extractor with PSFEx PSF modelling.",
            method=self.source_extraction_psf,
            output_path=output_dir,
            packages=["psfex", "source-extractor"]
        )
        self.signal_to_noise_measure(dual=dual)
        print()
        self.update_output_file()

    def world_to_pixel(
            self,
            coord: SkyCoord,
            origin: int = 0,
            ext: int = 0
    ) -> Tuple[np.ndarray]:
        """
        Turns a sky coordinate into image pixel coordinates;
        :param coord: SkyCoord object to convert to pixel coordinates; essentially a wrapper for SkyCoord.to_pixel()
        :param origin: Do you want pixel indexing that starts at 1 (FITS convention) or 0 (numpy convention)?
        :return: xp, yp: numpy.ndarray, the pixel coordinates.
        """
        self.load_wcs()
        return coord.to_pixel(self.wcs[ext], origin=origin)

    def pixel_to_world(
            self,
            x: Union[float, np.ndarray, units.Quantity],
            y: Union[float, np.ndarray, units.Quantity],
            origin: int = 0,
            ext: int = 0
    ) -> SkyCoord:
        """
        Uses the image's wcs to turn pixel coordinates into sky; essentially a wrapper for SkyCoord.from_pixel().
        :param x: Pixel x-coordinate. Can be provided as an astropy Quantity with units pix, or as a raw number.
        :param y: Pixel y-coordinate. Can be provided as an astropy Quantity with units pix, or as a raw number.
        :param origin: Do you want pixel indexing that starts at 1 (FITS convention) or 0 (numpy convention)?
        :return coord: SkyCoord reflecting the sky coordinates.
        """
        self.load_wcs()
        x = u.dequantify(x, unit=units.pix)
        y = u.dequantify(y, unit=units.pix)
        return SkyCoord.from_pixel(x, y, wcs=self.wcs[ext], origin=origin)

    def plot_ellipse(
            self,
            ax: plt.Axes,
            coord: SkyCoord,
            a: units.Quantity,
            b: units.Quantity = None,
            theta: units.Quantity = None,
            plot_centre: bool = False,
            centre_kwargs: dict = {},
            **kwargs
    ):
        if b is None:
            b = a
        if theta is None:
            theta = 0 * units.deg
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = "white"
        if "facecolor" not in kwargs:
            kwargs["facecolor"] = "none"
        x, y = self.world_to_pixel(coord, 0)
        e = Ellipse(
            xy=(x, y),
            width=2 * a.to(units.pix, self.pixel_scale_y).value,
            height=2 * b.to(units.pix, self.pixel_scale_y).value,
            angle=theta.value,
            **kwargs
        )
        # e.set_edgecolor(color)
        ax.add_artist(e)
        if plot_centre:
            if "c" not in centre_kwargs:
                if "edgecolor" in kwargs:
                    centre_kwargs["c"] = kwargs["edgecolor"]
                else:
                    centre_kwargs["c"] = "white"
            if "marker" not in centre_kwargs:
                centre_kwargs["marker"] = "x"
            ax.scatter(x, y, **centre_kwargs)
        return ax

    def load_data(self, force: bool = False):
        reset = force or not self.data
        super().load_data(force=force)
        if reset:
            self.data_sub_bkg = [None] * len(self.data)
            self.sep_background = [None] * len(self.data)
            self.pu_background = [None] * len(self.data)
        return self.data

    def get_source_cat(self, dual: bool, force: bool = False):
        if dual:
            source_cat = self.source_cat_dual
        else:
            source_cat = self.source_cat
        return source_cat

    def push_source_cat(self, dual: bool = True):
        source_cat = self.get_source_cat(dual=dual)
        for i, row in enumerate(source_cat):
            print(f"Pushing row {i} of {len(source_cat)}")
            obj = objects.Object(row=row, field=self.epoch.field)
            if "SNR_PSF" in self.depth["secure"]:
                depth = self.depth["secure"]["SNR_PSF"][f"5-sigma"]
            else:
                depth = self.depth["secure"]["SNR_AUTO"][f"5-sigma"]
            obj.add_photometry(
                instrument=self.instrument_name,
                fil=self.filter_name,
                epoch_name=self.epoch.name,
                mag=row['MAG_AUTO_ZP_best'],
                mag_err=row[f'MAGERR_AUTO_ZP_best'],
                snr=row[f'SNR_AUTO'],
                ellipse_a=row['A_WORLD'],
                ellipse_a_err=row["ERRA_WORLD"],
                ellipse_b=row['B_WORLD'],
                ellipse_b_err=row["ERRB_WORLD"],
                ellipse_theta=row['THETA_J2000'],
                ellipse_theta_err=row['ERRTHETA_J2000'],
                ra=row['RA'],
                ra_err=np.sqrt(row["ERRX2_WORLD"]),
                dec=row['DEC'],
                dec_err=np.sqrt(row["ERRY2_WORLD"]),
                kron_radius=row["KRON_RADIUS"],
                separation_from_given=None,
                epoch_date=str(self.epoch.date.isot),
                class_star=row["CLASS_STAR"],
                spread_model=row["SPREAD_MODEL"],
                spread_model_err=row["SPREADERR_MODEL"],
                class_flag=row["CLASS_FLAG"],
                mag_psf=row["MAG_PSF_ZP_best"],
                mag_psf_err=row["MAGERR_PSF_ZP_best"],
                snr_psf=row["FLUX_PSF"] / row["FLUXERR_PSF"],
                image_depth=depth,
                image_path=self.path,
                do_mask=self.mask_nearby(),
                zeropoint=row["ZP_best_ATM_CORR"]
            )
            obj.push_to_table(select=False)

    def load_synth_cat(self, force: bool = False):
        if force or self.synth_cat is None:
            if self.synth_cat_path is not None:
                u.debug_print(2, f"ImagingImage.load_synth_cat(): {self}.synth_cat_path ==", self.synth_cat_path)
                self.synth_cat = table.QTable.read(self.synth_cat_path, format="ascii.ecsv")
            else:
                u.debug_print(1, "No valid synth_cat_path found. Could not load synth_cat.")
            return self.synth_cat

    def write_synth_cat(self):

        if self.synth_cat is None:
            u.debug_print(1, "synth_cat not yet loaded.")
        else:
            if self.synth_cat_path is None:
                self.synth_cat_path = self.path.replace(".fits", "_synth_cat.ecsv")
            u.debug_print(1, "Writing source catalogue to", self.synth_cat_path)
            self.synth_cat.write(self.synth_cat_path, format="ascii.ecsv", overwrite=True)

    def load_wcs(self) -> List[wcs.WCS]:
        if not self.wcs:
            self.load_headers()
            self.wcs = []
            for ext in range(len(self.headers)):
                self.wcs.append(wcs.WCS(header=self.headers[ext]))
        return self.wcs

    def extract_astrometry_err(self):
        key = self.header_keys()["astrometry_err"]
        self.astrometry_err = self.extract_header_item(key)
        key = self.header_keys()["ra_err"]
        self.ra_err = self.extract_header_item(key)
        key = self.header_keys()["dec_err"]
        self.dec_err = self.extract_header_item(key)
        if self.astrometry_err is not None:
            self.astrometry_err *= units.arcsec
        if self.ra_err is not None:
            self.ra_err *= units.arcsec
        if self.dec_err is not None:
            self.dec_err *= units.arcsec
        return self.astrometry_err

    def extract_psf_fwhm(self):
        key = self.header_keys()["psf_fwhm"]
        return self.extract_header_item(key) * units.arcsec

    def extract_rotation_angle(self, ext: int = 0):
        self.load_wcs()
        matrix = self.wcs[ext].pixel_scale_matrix
        theta = np.arctan2(matrix[1, 1], matrix[1, 0]) * 180 / np.pi - 90
        return theta * units.deg

    def extract_y_sense(self, ext: int = 0):
        self.load_wcs()
        matrix = self.wcs[ext].pixel_scale_matrix
        return np.sign(matrix[1, 1])

    def extract_wcs_footprint(self, ext: int = 0):
        """
        Returns the RA & Dec of the corners of the image.
        :return: tuple of SkyCoords, (top_left, top_right, bottom_left, bottom_right)
        """
        self.load_wcs()
        print("Calculating footprint.")
        return self.wcs[ext].calc_footprint()

    def _pixel_scale(self, ext: int = 0):
        self.load_wcs()
        print("Getting pixel scale.")
        return wcs.utils.proj_plane_pixel_scales(
            self.wcs[ext]
        ) * units.deg

    def extract_pixel_scale(self, ext: int = 0, force: bool = False):
        if force or self.pixel_scale_x is None or self.pixel_scale_y is None:
            x, y = self._pixel_scale(ext=ext)
            self.pixel_scale_x = units.pixel_scale(x / units.pix)
            self.pixel_scale_y = units.pixel_scale(y / units.pix)
        else:
            u.debug_print(2, "Pixel scale already set.")

        return self.pixel_scale_x, self.pixel_scale_y

    def extract_world_scale(self, ext: int = 0, force: bool = False):
        x, y = self._pixel_scale(ext=ext)
        dec = self.extract_pointing().dec.to(units.rad)
        ra_scale = units.pixel_scale((x / np.cos(dec)) / units.pix)
        dec_scale = units.pixel_scale(y / units.pix)
        return ra_scale, dec_scale

    def extract_filter(self):
        key = self.header_keys()["filter"]
        self.filter_name = self.extract_header_item(key)
        if self.filter_name is not None:
            self.filter_short = self.filter_name[0]

        self._filter_from_name()

        return self.filter_name

    def _filter_from_name(self):
        if self.filter_name is not None and self.instrument is not None and self.filter_name in self.instrument.filters:
            self.filter = self.instrument.filters[self.filter_name]

    def extract_airmass(self):
        key = self.header_keys()["airmass"]
        self.airmass = self.extract_header_item(key)
        key = self.header_keys()["airmass_err"]
        self.airmass_err = self.extract_header_item(key)
        if self.airmass_err is None:
            self.airmass_err = 0.0
        return self.airmass

    def extract_pointing(self):
        key = self.header_keys()["ra"]
        ra = self.extract_header_item(key)
        key = self.header_keys()["dec"]
        dec = self.extract_header_item(key)
        self.pointing = SkyCoord(ra, dec, unit=units.deg)
        return self.pointing

    def extract_ref_pixel(self) -> Tuple[float, float]:
        """
        Retrieve the coordinates of the "reference pixel" from the header.
        :return: Tuple containing the reference pixel coordinates as (x, y).
        """
        key = self.header_keys()["ref_pix_x"]
        x = self.extract_header_item(key)
        key = self.header_keys()["ref_pix_y"]
        y = self.extract_header_item(key)
        return x, y

    def extract_old_pointing(self):
        key = self.header_keys()["ra_old"]
        ra = self.extract_header_item(key)
        key = self.header_keys()["dec_old"]
        dec = self.extract_header_item(key)
        return SkyCoord(ra, dec, unit=units.deg)

    def _output_dict(self):
        outputs = super()._output_dict()
        outputs.update(
            {
                "astrometry_stats": self.astrometry_stats,
                "extinction_atmospheric": self.extinction_atmospheric,
                "extinction_atmospheric_err": self.extinction_atmospheric_err,
                "filter": self.filter_name,
                "psfex_path": self.psfex_path,
                "synth_cat_path": self.synth_cat_path,
                "psf_stats": self.psf_stats,
                "fwhm_pix_psfex": self.fwhm_pix_psfex,
                "fwhm_psfex": self.fwhm_psfex,
                "psfex_succesful": self.psfex_successful,
                "zeropoints": self.zeropoints,
                "zeropoint_output_paths": self.zeropoint_output_paths,
                "zeropoint_best": self.zeropoint_best,
                "depth": self.depth,
                "dual_mode_template": self.dual_mode_template,
            }
        )
        if self.source_cat is not None:
            outputs["source_cat_path"] = self.source_cat.path
        if self.source_cat_dual is not None:
            outputs["source_cat_dual_path"] = self.source_cat_dual.path,
        return outputs

    def update_output_file(self):
        p.update_output_file(self)
        if self.source_cat is not None:
            self.source_cat.update_output_file()
        if self.source_cat_dual is not None:
            self.source_cat_dual.update_output_file()
        self.write_synth_cat()

    def load_output_file(self):
        outputs = super().load_output_file()
        if outputs is not None:
            if "astrometry_stats" in outputs:
                self.astrometry_stats = outputs["astrometry_stats"]
            if "extinction_atmospheric" in outputs:
                self.extinction_atmospheric = outputs["extinction_atmospheric"]
            if "extinction_atmospheric_err" in outputs:
                self.extinction_atmospheric_err = outputs["extinction_atmospheric_err"]
            if "filter" in outputs:
                self.filter_name = outputs["filter"]
            if "psfex_path" in outputs:
                self.psfex_path = outputs["psfex_path"]
            if "source_cat_path" in outputs and outputs["source_cat_path"] is not None and os.path.exists(
                    outputs["source_cat_path"]):
                self.source_cat = catalog.SECatalogue(path=outputs["source_cat_path"], image=self)
            if "synth_cat_path" in outputs:
                self.synth_cat_path = outputs["synth_cat_path"]
            if "source_cat_dual_path" in outputs and outputs["source_cat_dual_path"] is not None and os.path.exists(
                    outputs["source_cat_dual_path"]):
                self.source_cat_dual = catalog.SECatalogue(path=outputs["source_cat_dual_path"], image=self)
            if "fwhm_psfex" in outputs:
                self.fwhm_psfex = outputs["fwhm_psfex"]
            if "fwhm_psfex" in outputs:
                self.fwhm_pix_psfex = outputs["fwhm_pix_psfex"]
            if "psf_stats" in outputs:
                self.psf_stats = outputs["psf_stats"]
            if "psfex_successful" in outputs:
                self.psfex_successful = outputs["psfex_successful"]
            if "zeropoints" in outputs:
                self.zeropoints = outputs["zeropoints"]
            if "zeropoint_output_paths" in outputs:
                self.zeropoint_output_paths = outputs["zeropoint_output_paths"]
            if "zeropoint_best" in outputs:
                self.zeropoint_best = outputs["zeropoint_best"]
            if "depth" in outputs and outputs["depth"] is not None:
                self.depth = outputs["depth"]
            if "dual_mode_template" in outputs and outputs["dual_mode_template"] is not None:
                self.dual_mode_template = outputs["dual_mode_template"]
        return outputs

    def select_zeropoint(self, no_user_input: bool = True, preferred: str = None):

        if not self.zeropoints:
            print(f"No zeropoints set ({self}.zeropoints is None); try loading output file.")
            return None, None

        ranking, diff = self.rank_photometric_cat(cats=self.zeropoints)
        if preferred is not None:
            ranking.insert(0, preferred)

        zps = []
        for i, cat in enumerate(ranking):
            if cat in self.zeropoints:
                zps_cat = []
                for img_name in self.zeropoints[cat]:
                    zp = self.zeropoints[cat][img_name]
                    zps_cat.append(zp)
                zps_cat.sort(key=lambda z: z["zeropoint_img_err"])
                zps.extend(zps_cat)

        zp_tbl = table.QTable(zps)
        if len(zp_tbl) > 0:
            # zp_tbl.sort("zeropoint_img_err")
            zp_tbl.write(
                os.path.join(self.data_path, f"{self.name}_zeropoints.ecsv"), format="ascii.ecsv",
                overwrite=True
            )
            best_row = zp_tbl[0]
            best_cat = best_row["catalogue"]
            best_img = best_row["image_name"]

            if best_cat is None:
                raise ValueError("No zeropoints are present to select from.")

            zeropoint_best = self.zeropoints[best_cat][best_img]
            print(
                f"For {self.name}, we have selected a zeropoint of {zeropoint_best['zeropoint_img']} "
                f"+/- {zeropoint_best['zeropoint_img_err']}, "
                f"from {zeropoint_best['catalogue']} on {zeropoint_best['image_name']}.")
            if not no_user_input:
                select_own = u.select_yn(message="Would you like to select another?", default=False)
                if select_own:
                    zps = {}
                    for i, row in enumerate(zp_tbl):
                        pick_str = f"{row['catalogue']} {row['zeropoint_img']} +/- {row['zeropoint_img_err']}, " \
                                   f"{row['n_matches']} stars, " \
                                   f"from {row['image_name']}"
                        zps[pick_str] = self.zeropoints[row['catalogue']][row['image_name']]
                    _, zeropoint_best = u.select_option(message="Select best zeropoint:", options=zps)
                    best_cat = zeropoint_best["catalogue"]
            self.zeropoint_best = zeropoint_best

            self.set_header_items(
                items={
                    "ZP": zeropoint_best["zeropoint_img"],
                    "ZP_ERR": zeropoint_best["zeropoint_img_err"],
                    "ZPCAT": str(zeropoint_best["catalogue"]),
                },
                ext=0,
                write=False
            )

            self.add_log(
                action=f"Selected best zeropoint as {zeropoint_best['zeropoint']} +/- {zeropoint_best['zeropoint_err']}, from {zeropoint_best['catalogue']}",
                method=self.select_zeropoint
            )
        else:
            best_cat = None

        self.update_output_file()
        self.write_fits_file()
        return self.zeropoint_best, best_cat

    def zeropoint(
            self,
            cat_path: str,
            output_path: str,
            cat_name: str,
            cat_zeropoint: units.Quantity = 0.0 * units.mag,
            cat_zeropoint_err: units.Quantity = 0.0 * units.mag,
            image_name: str = None,
            show: bool = False,
            phot_type: str = "PSF",
            sex_ra_col: str = "RA",
            sex_dec_col: str = "DEC",
            stars_only: bool = True,
            star_class_tol: float = 0.95,
            mag_range_sex_lower: units.Quantity = -100. * units.mag,
            mag_range_sex_upper: units.Quantity = 100. * units.mag,
            dist_tol: units.Quantity = None,
            snr_cut=3.,
            iterate_uncertainty: bool = True,
            do_x_shift: bool = True,
            vega: bool = False,
            **kwargs
    ):
        print(f"\nEstimating photometric zeropoint for {self.name}, {type(self)}\n")

        sex_flux_col = f"FLUX_{phot_type}"

        if phot_type == "PSF":
            sex_x_col = "XPSF_IMAGE"
            sex_y_col = "YPSF_IMAGE"
        else:
            sex_x_col = "X_IMAGE"
            sex_y_col = "Y_IMAGE"

        self.signal_to_noise_measure()
        if image_name is None:
            image_name = self.name
        self.extract_filter()
        column_names = cat_columns(cat=cat_name, f=self.filter_short)
        cat_ra_col = column_names['ra']
        cat_dec_col = column_names['dec']
        cat_mag_col = column_names['mag_psf']
        cat_mag_col_err = column_names['mag_psf_err']
        cat_type = "csv"

        if dist_tol is None:
            self.load_headers()
            self.extract_astrometry_err()
            if self.astrometry_err is not None:
                dist_tol = 10 * self.astrometry_err
            else:
                dist_tol = 2 * units.arcsec

        zp_dict = ph.determine_zeropoint_sextractor(
            sextractor_cat=self.source_cat.table,
            image=self.path,
            cat_path=cat_path,
            cat_name=cat_name,
            output_path=output_path,
            image_name=image_name,
            show=show,
            cat_ra_col=cat_ra_col,
            cat_dec_col=cat_dec_col,
            cat_mag_col=cat_mag_col,
            cat_mag_col_err=cat_mag_col_err,
            sex_ra_col=sex_ra_col,
            sex_dec_col=sex_dec_col,
            sex_x_col=sex_x_col,
            sex_y_col=sex_y_col,
            dist_tol=dist_tol,
            flux_column=sex_flux_col,
            mag_range_sex_upper=mag_range_sex_upper,
            mag_range_sex_lower=mag_range_sex_lower,
            stars_only=stars_only,
            star_class_tol=star_class_tol,
            exp_time=self.extract_exposure_time(),
            cat_type=cat_type,
            cat_zeropoint=cat_zeropoint,
            cat_zeropoint_err=cat_zeropoint_err,
            snr_col=f'SNR_{phot_type}',
            snr_cut=snr_cut,
            iterate_uncertainty=iterate_uncertainty,
            do_x_shift=do_x_shift
        )

        if zp_dict is None:
            return None

        if vega:
            offset = self.filter.vega_magnitude_offset()
            zp_dict["zeropoint"] += offset
            zp_dict["ab_correction"] = offset

        zp_dict = self.add_zeropoint(
            # catalogue=cat_name,
            # zeropoint=zp_dict["zeropoint"],
            # zeropoint_err=zp_dict["zeropoint_err"],
            extinction=0.0 * units.mag,
            extinction_err=0.0 * units.mag,
            # airmass=0.0,
            airmass_err=0.0,
            # n_matches=zp_dict["n_matches"],
            image_name="self",
            **zp_dict
        )
        self.zeropoint_output_paths[cat_name.lower()] = output_path
        self.add_log(
            action=f"Calculated zeropoint as {zp_dict['zeropoint_img']} +/- {zp_dict['zeropoint_img_err']}, from {zp_dict['catalogue']}.",
            method=self.zeropoint,
            output_path=output_path
        )
        self.update_output_file()
        return zp_dict

    def get_zeropoint(
            self,
            cat_name: str,
            img_name: str = 'self'
    ):
        if cat_name == "best":
            zp_dict = self.zeropoint_best
        elif cat_name not in self.zeropoints:
            raise KeyError(f"Zeropoint {cat_name} does not exist.")
        elif img_name not in self.zeropoints[cat_name]:
            raise KeyError(f"Zeropoint from {img_name} does not exist")
        else:
            zp_dict = self.zeropoints[cat_name][img_name]

        return zp_dict

    def add_zeropoint(
            self,
            catalogue: str,
            zeropoint: Union[float, units.Quantity],
            zeropoint_err: Union[float, units.Quantity],
            extinction: Union[float, units.Quantity],
            extinction_err: Union[float, units.Quantity],
            airmass: float,
            airmass_err: float,
            n_matches: int = None,
            image_name: str = "self",
            **kwargs
    ):
        zp_dict = kwargs.copy()
        if extinction is None:
            extinction = extinction_err = 0. * units.mag
        elif extinction_err is None:
            extinction_err = 0. * units.mag

        if airmass is None:
            airmass = airmass_err = 0.
        elif airmass_err is None:
            airmass_err = 0.
        zp_dict.update({
            "zeropoint": u.check_quantity(zeropoint, units.mag),
            "zeropoint_err": u.check_quantity(zeropoint_err, units.mag),
            "extinction": u.check_quantity(extinction, units.mag),
            "extinction_err": u.check_quantity(extinction_err, units.mag),
            "airmass": airmass,
            "airmass_err": airmass_err,
            "catalogue": catalogue.lower(),
            "n_matches": n_matches,
            "image_name": image_name
        })
        print(f"Adding zeropoint to image {self.path}"
              f"\nfrom {catalogue} "
              f"\non {image_name}:")
        print(f"\t {zeropoint=} +/- {zeropoint_err}")
        print(f"\t {airmass=} +/- {airmass_err}")
        print(f"\t {extinction=} +/- {extinction_err}")
        zp_dict["zeropoint_img"] = zp_dict["zeropoint"] - zp_dict["extinction"] * zp_dict["airmass"]
        zp_dict['zeropoint_img_err'] = np.sqrt(
            zp_dict["zeropoint_err"] ** 2 + u.uncertainty_product(
                zp_dict["extinction"] * zp_dict["airmass"],
                (zp_dict["extinction"], zp_dict["extinction_err"]),
                (zp_dict["airmass"], zp_dict["airmass_err"])
            ) ** 2
        )

        img_key = image_name

        cat_key = catalogue.lower()
        if cat_key not in self.zeropoints:
            self.zeropoints[cat_key] = {}
        self.zeropoints[cat_key][img_key] = zp_dict
        self.update_output_file()
        return zp_dict

    def add_zeropoint_from_other(self, other: 'ImagingImage'):
        if other.filter_name != self.filter_name:
            raise ValueError(
                f"Zeropoints must come from images with the same filter; other filter {other.filter_name} does not match this filter {self.filter_name}.")
        if other.instrument_name != self.instrument_name:
            raise ValueError(
                f"Zeropoints must come from images with the same instrument; other instrument {other.instrument_name} does not match this instrument {self.instrument_name}.")

        airmass = self.extract_airmass()
        airmass_err = self.airmass_err

        airmass_other = other.extract_airmass()
        airmass_other_err = other.airmass_err

        delta_airmass = airmass - airmass_other
        delta_airmass_err = u.uncertainty_sum(airmass_err, airmass_other_err)

        for source in other.zeropoints:
            if 'self' in other.zeropoints[source]:
                zeropoint = other.zeropoints[source]['self']
                zeropoint.update({
                    "airmass": delta_airmass,
                    "airmass_err": delta_airmass_err,
                    "image_name": other.path,
                    "extinction": self.extinction_atmospheric,
                    "extinction_err": self.extinction_atmospheric_err
                })
                self.add_zeropoint(
                    **zeropoint
                )
        self.update_output_file()

    def aperture_areas(self):
        self.extract_pixel_scale()
        self.source_cat.aperture_areas(pixel_scale=self.pixel_scale_y)
        self.source_cat_dual.aperture_areas(pixel_scale=self.pixel_scale_y)

        self.add_log(
            action=f"Calculated area of FLUX_AUTO apertures.",
            method=self.aperture_areas,
        )
        self.update_output_file()

    def calibrate_magnitudes(
            self,
            zeropoint_name: str = "best",
            force: bool = True,
            dual: bool = False
    ):
        cat = self.get_source_cat(dual=dual, force=True)
        if cat is None:
            raise ValueError(f"Catalogue (dual={dual}) could not be loaded.")

        self.extract_exposure_time()
        zp_dict = self.get_zeropoint(cat_name=zeropoint_name)
        if zp_dict is None:
            raise ValueError(f"No zeropoint found for {self.name}.")
        cat.calibrate_magnitudes(
            zeropoint_dict=zp_dict,
            mag_name=f"ZP_{zeropoint_name}",
            force=force
        )
        self.add_log(
            action=f"Calibrated source self.tablealogue magnitudes using zeropoint {zeropoint_name}.",
            method=self.calibrate_magnitudes,
        )

    def magnitude(
            self,
            flux: units.Quantity,
            flux_err: units.Quantity = None,
            cat_name: str = 'best',
            img_name: str = 'self',
            **kwargs
    ):

        zp_dict = self.get_zeropoint(cat_name=cat_name, img_name=img_name)

        if flux_err is None:
            if isinstance(flux, units.Quantity):
                flux_err = 0 * flux.unit
            else:
                flux_err = 0.

        if zp_dict is None:
            raise ValueError(f"The {cat_name} zeropoint on {img_name}, for {self.name}, does not appear to exist.")

        if "exp_time" not in kwargs:
            kwargs["exp_time"] = self.extract_exposure_time()
        if "exp_time_err" not in kwargs:
            kwargs["exp_time_err"] = 0.0 * units.second
        if "colour" not in kwargs:
            kwargs["colour"] = 0.0 * units.mag
        if "colour_term" not in kwargs:
            kwargs["colour_term"] = 0.0

        mag, mag_err = ph.magnitude_instrumental(
            flux=flux,
            flux_err=flux_err,
            zeropoint=zp_dict['zeropoint'],
            zeropoint_err=zp_dict['zeropoint_err'],
            airmass=zp_dict['airmass'],
            airmass_err=zp_dict['airmass_err'],
            ext=zp_dict['extinction'],
            ext_err=zp_dict['extinction_err'],
            **kwargs
        )

        mag_no_ext_corr, mag_no_ext_corr_err = ph.magnitude_instrumental(
            flux=flux,
            flux_err=flux_err,
            exp_time=self.extract_exposure_time(),
            exp_time_err=0.0 * units.second,
            zeropoint=zp_dict['zeropoint'],
            zeropoint_err=zp_dict['zeropoint_err'],
            airmass=0.0,
            airmass_err=0.0,
            ext=0.0 * units.mag,
            ext_err=0.0 * units.mag,
            colour_term=0.0,
            colour=0.0 * units.mag,
        )

        return mag, mag_err, mag_no_ext_corr, mag_no_ext_corr_err

    def estimate_depth(
            self,
            zeropoint_name: str = "best",
            dual: bool = False,
            stars_only: bool = False,
            star_tolerance: float = 0.9,
            do_magnitude_calibration: bool = True
    ):
        """
        Use various measures of S/N to estimate image depth at a range of sigmas.
        :param zeropoint_name:
        :param dual:
        :return:
        """

        # self.signal_to_noise_ccd(dual=dual)

        self.signal_to_noise_measure(dual=dual)
        if do_magnitude_calibration:
            self.calibrate_magnitudes(zeropoint_name=zeropoint_name, dual=dual)

        source_cat = self.get_source_cat(dual=dual).table

        # "max" stores the magnitude of the faintest object with S/N > x sigma
        self.depth = {"max": {}, "secure": {}}
        # "secure" finds the brightest object with S/N < x sigma, then increments to the
        # overall; thus giving the faintest magnitude at which we can be confident of a detection

        if stars_only:
            source_cat = source_cat[source_cat["CLASS_STAR"] >= star_tolerance]

        for snr_key in ["PSF", "AUTO"]:  # ["SNR_CCD", "SNR_MEASURED", "SNR_SE"]:
            # We do this to ensure that, in the "secure" step, object i+1 is the next-brightest in the catalogue
            if f"FLUX_{snr_key}" not in source_cat.colnames:
                continue
            source_cat_key = source_cat.copy()
            source_cat_key.sort(f"FLUX_{snr_key}")
            self.depth["max"][f"SNR_{snr_key}"] = {}
            self.depth["secure"][f"SNR_{snr_key}"] = {}
            # Dispose of the infinite SNRs and mags
            source_cat_key = source_cat_key[np.invert(np.isinf(source_cat_key[f"MAG_{snr_key}"]))]
            source_cat_key = source_cat_key[np.invert(np.isinf(source_cat_key[f"SNR_{snr_key}"]))]
            source_cat_key = source_cat_key[source_cat_key[f"MAG_{snr_key}"] < 100 * units.mag]
            source_cat_key.sort(f"FLUX_{snr_key}")
            for sigma in range(1, 6):
                source_cat_sigma = source_cat_key.copy()
                u.debug_print(1, "ImagingImage.estimate_depth(): snr_key, sigma ==", snr_key, sigma)
                # Faintest source at x-sigma:
                u.debug_print(
                    1, f"ImagingImage.estimate_depth(): source_cat[SNR_{snr_key}].unit ==",
                    source_cat_sigma[f"SNR_{snr_key}"].unit)
                cat_more_xsigma = source_cat_sigma[source_cat_sigma[f"SNR_{snr_key}"] > sigma]
                self.depth["max"][f"SNR_{snr_key}"][f"{sigma}-sigma"] = np.max(
                    cat_more_xsigma[f"MAG_{snr_key}_ZP_{zeropoint_name}"])

                # Brightest source less than x-sigma (kind of)
                # Get the sources with SNR less than x-sigma
                source_less_sigma = source_cat_sigma[source_cat_sigma[f"SNR_{snr_key}"] < sigma]
                if len(source_less_sigma) > 0:
                    # Get the source with the greatest flux
                    i = np.argmax(source_less_sigma[f"FLUX_{snr_key}"])
                    # Find its counterpart in the full catalogue
                    i, _ = u.find_nearest(source_cat["NUMBER"], source_less_sigma[i]["NUMBER"])
                    # Get the source that is next up in brightness (being brighter)
                    i += 1
                    src_lim = source_cat[i]

                else:
                    src_lim = source_cat_sigma[source_cat_sigma[f"SNR_{snr_key}"].argmin()]

                self.depth["secure"][f"SNR_{snr_key}"][f"{sigma}-sigma"] = src_lim[f"MAG_{snr_key}_ZP_{zeropoint_name}"]

                self.update_output_file()

        source_cat.sort("NUMBER")
        self.add_log(
            action=f"Estimated image depth.",
            method=self.estimate_depth,
            stars_only=stars_only,
            star_tolerance=star_tolerance
        )
        self.update_output_file()

        return self.depth

    def send_column_to_source_cat(
            self,
            column_name: str,
            subset_table: table.Table,
            dual: bool = False
    ):
        """
        Takes a column from a table that is a subset of source_cat, and adds the column values to the appropriate
        entries in source_cat using the `NUMBER` column. Rows not in the subset table will be assigned `-99.` in that
        column.
        Trust me, it comes in handy.
        Assumes that NO entries have been removed from or added to the main source_cat since being produced by Source Extractor,
        as it requires that the relationship source_cat["NUMBER"] = i - 1 holds true.

        :param column_name: Name of column to send.
        :param subset_table: Subset table
        :param dual:
        :return:
        """
        cat = self.get_source_cat(dual=dual)
        cat.add_partial_column(
            column_name=column_name,
            subset_table=subset_table,
        )

    def register(
            self,
            target: 'ImagingImage',
            output_path: str,
            ext: int = 0,
            ext_target: int = 0,
            trim: bool = True,
            **kwargs
    ):
        self.load_data()
        target.load_data()

        data_source = self.data[ext]
        data_source = u.sanitise_endianness(data_source)
        data_target = target.data[ext_target]
        data_target = u.sanitise_endianness(data_target)
        u.debug_print(0,
                      f"Attempting registration of {self.name} (Chip {self.extract_chip_number()}) against {target.name} (Chip {target.extract_chip_number()})")
        registered, footprint = register(data_source, data_target, **kwargs)

        self.copy(output_path)
        with fits.open(output_path, mode="update") as new_file:
            new_file[0].data = registered
            u.debug_print(1, "Writing registered image to", output_path)
            new_file.writeto(output_path, overwrite=True)

        new_image = self.new_image(path=output_path)
        new_image.transfer_wcs(target)

        if trim:
            frame_value = new_image.detect_frame_value(ext=ext)
            if frame_value is not None:
                left, right, bottom, top = new_image.detect_edges(frame_value=frame_value)
                trimmed = new_image.trim(left=left, right=right, bottom=bottom, top=top)
                new_image = trimmed

        new_image.add_log(
            action=f"Registered and reprojected to footprint of {target} using astroalign.",
            method=self.register,
        )
        new_image.update_output_file()
        return new_image

    def detect_frame_value(self, ext: int = 0):
        self.open()
        frame_value = ff.detect_frame_value(file=self.hdu_list, ext=ext)
        self.close()
        return frame_value

    def detect_edges(self, frame_value: float, ext: int = 0):
        self.open()
        left, right, bottom, top = ff.detect_edges(file=self.hdu_list, value=frame_value, ext=ext)
        self.close()
        return left, right, bottom, top

    # TODO: Add option to run Astrometry.net via API / astroquery:
    # https://astroquery.readthedocs.io/en/latest/astrometry_net/astrometry_net.html

    def correct_astrometry(
            self,
            output_dir: str = None,
            tweak: bool = True,
            time_limit: int = None,
            am_flags: list = (),
            am_params: dict = None,
            **kwargs
    ):
        """
        Uses astrometry.net to solve the astrometry of the image. Solved image is output as a separate file.
        :param output_dir:
        :param tweak:
        :param time_limit:
        :param am_flags:
        :param am_params:
        :param kwargs:
        :return:
        """
        self.extract_pointing()
        u.debug_print(1, "image.correct_astrometry(): tweak ==", tweak)
        if output_dir is not None:
            u.mkdir_check(output_dir)
        base_filename = f"{self.name}_astrometry"
        if "search_radius" not in kwargs:
            kwargs["search_radius"] = 4.0 * units.arcmin
        success = solve_field(
            image_files=self.path,
            base_filename=base_filename,
            overwrite=True,
            tweak=tweak,
            guess_scale=True,
            centre=self.pointing,
            time_limit=time_limit,
            am_flags=am_flags,
            am_params=am_params,
            **kwargs
        )
        if not success:
            return None
        new_path = os.path.join(self.data_path, f"{base_filename}.new")
        new_new_path = os.path.join(self.data_path, f"{base_filename}.fits")
        os.rename(new_path, new_new_path)

        if output_dir is not None and not os.path.samefile(self.data_path, output_dir):
            if not os.path.isdir(output_dir):
                raise ValueError(f"Invalid output directory {output_dir}")
            for astrometry_product in filter(lambda f: f.startswith(base_filename), os.listdir(self.data_path)):
                path = os.path.join(self.data_path, astrometry_product)
                shutil.copy(path, output_dir)
                os.remove(path)
        else:
            output_dir = self.data_path
        final_file = os.path.join(output_dir, f"{base_filename}.fits")
        self.astrometry_corrected_path = final_file
        new_image = self.new_image(final_file)
        new_image.set_header_item("GAIA", True)
        new_image.add_log(
            "Astrometry corrected using Astrometry.net.",
            method=self.correct_astrometry,
            packages=["astrometry.net"])
        new_image.update_output_file()
        new_image.write_fits_file()
        return new_image

    def correct_astrometry_coarse(
            self,
            output_dir: str = None,
            cat: table.Table = None,
            ext: int = 0,
            cat_name: str = None,
            **diag_kwargs
    ):
        if self.source_cat.table is None:
            self.source_extraction_psf(output_dir=output_dir)

        if cat is None:
            if self.epoch is not None:
                cat = self.epoch.epoch_gaia_catalogue()
            else:
                raise ValueError(f"If image epoch is not assigned, cat must be provided.")
        diagnostics = self.astrometry_diagnostics(
            reference_cat=cat,
            **diag_kwargs
        )
        new_path = os.path.join(output_dir, self.filename.replace(".fits", "_astrometry.fits"))
        new = self.copy(new_path)

        ra_scale, dec_scale = self.extract_world_scale(ext=ext)

        new.load_headers()

        if not np.isnan(diagnostics["median_offset_x"].value) and not np.isnan(diagnostics["median_offset_y"].value):

            delta_ra = diagnostics["median_offset_x"].to(units.deg, ra_scale).value
            delta_dec = -diagnostics["median_offset_y"].to(units.deg, dec_scale).value

            new.shift_wcs(
                delta_ra=delta_ra,
                delta_dec=delta_dec
            )

            new.add_log(
                "Astrometry corrected using median offsets from reference catalogue.",
                method=self.correct_astrometry_coarse,
                input_path=self.path,
                output_path=new_path,
                ext=ext
            )
            if cat_name.lower() == 'gaia':
                new.set_header_item("GAIA", True)
            new.write_fits_file()
            return new, {"delta_ra": delta_ra, "delta_dec": delta_dec}
        else:
            u.rm_check(new_path)
            return None

    def shift_wcs(self, delta_ra: units.Quantity, delta_dec: units.Quantity, ext: int = 0):
        delta_ra = u.dequantify(delta_ra, unit=units.deg)
        delta_dec = u.dequantify(delta_dec, unit=units.deg)
        self.headers[ext]["CRVAL1"] += delta_ra
        self.headers[ext]["CRVAL2"] += delta_dec
        self.add_log(
            f"Shifted WCS coordinate reference value by RA+={delta_ra}, DEC+={delta_dec}.",
            method=self.shift_wcs,
            ext=ext
        )

    def transfer_wcs(self, other_image: 'ImagingImage', ext: int = 0):
        other_image.load_headers()
        self.load_headers()
        self.headers[ext] = ff.wcs_transfer(
            header_template=other_image.headers[ext],
            header_update=self.headers[ext]
        )
        self.add_log(
            f"Changed WCS information to match {other_image}.",
            method=self.transfer_wcs
        )
        self.update_output_file()
        self.write_fits_file()

    def correct_astrometry_from_other(self, other_image: 'ImagingImage', output_dir: str = None) -> 'ImagingImage':
        """
        Uses the header information from an image that has already been corrected by the Astrometry.net code to apply
        the same tweak to this image.
        This assumes that both images had the same astrometry to begin with, eg if one is a derived version of the other.
        :param other_image: Header must contain both _RVAL and CRVAL keywords.
        :param output_dir: Path to write new fits file to.
        :return:
        """
        if not isinstance(other_image, ImagingImage):
            raise ValueError("other_image is not a valid ImagingImage")
        other_header = other_image.load_headers()[0]

        u.mkdir_check_nested(output_dir, remove_last=False)
        output_path = os.path.join(output_dir, f"{self.name}_astrometry.fits")
        shutil.copyfile(self.path, output_path)

        # TODO: This method works, but does not preserve the order of header keys in the new file.
        # In fact, it makes rather a mess of them. Work out how to do this properly.

        # Take old astrometry info from other header
        end_index = start_index = other_header.index("_RVAL1") - 1
        keys = list(other_header.keys())
        while keys[end_index].startswith("_") or keys[end_index] == "COMMENT":
            end_index += 1
        insert = other_header[start_index:end_index]

        # Take new astrometry info from other header
        start_index = other_header.index("WCSAXES") - 5
        end_index = start_index + 269
        insert.update(other_header[start_index:end_index])

        # Calculate offset, in other image and in world coordinates, of the new reference pixel from its old one.
        other_pointing = other_image.extract_pointing()
        other_old_pointing = other_image.extract_old_pointing()
        offset_ra = other_pointing.ra - other_old_pointing.ra
        offset_dec = other_pointing.dec - other_old_pointing.dec

        # Calculate the offset, in the other image and in pixels, of the new reference frame from its old one.
        offset_crpix1 = other_header["CRPIX1"] - other_header["_RPIX1"]
        offset_crpix2 = other_header["CRPIX2"] - other_header["_RPIX2"]

        if "GAIA" in other_header:
            insert["GAIA"] = other_header["GAIA"]

        with fits.open(output_path, "update") as file:
            # Apply the same offsets to this image, while keeping the old values as "_" keys
            insert["_RVAL1"] = file[0].header["CRVAL1"]
            insert["_RVAL2"] = file[0].header["CRVAL2"]
            insert["CRVAL1"] = insert["_RVAL1"] + offset_ra.value
            insert["CRVAL2"] = insert["_RVAL2"] + offset_dec.value

            insert["_RPIX1"] = file[0].header["CRPIX1"]
            insert["_RPIX2"] = file[0].header["CRPIX2"]
            insert["CRPIX1"] = insert["_RPIX1"] + offset_crpix1
            insert["CRPIX2"] = insert["_RPIX2"] + offset_crpix2

            # Insert all other astrometry info as previously extracted.
            file[0].header.update(insert)

        cls = type(self)  # ImagingImage.select_child_class(instrument=self.instrument_name)
        new_image = cls(path=output_path)

        new_image.add_log(
            f"Used WCS info from {other_image} to correct this image.",
            method=self.correct_astrometry_from_other
        )

        new_image.update_output_file()

        return new_image

    def astrometry_diagnostics(
            self,
            reference_cat: Union[str, table.QTable],
            ra_col: str = "ra", dec_col: str = "dec", mag_col: str = "phot_g_mean_mag",
            offset_tolerance: units.Quantity = 0.5 * units.arcsec,
            # star_tolerance: float = 1,
            local_coord: SkyCoord = None,
            local_radius: units.Quantity = 0.5 * units.arcmin,
            show_plots: bool = False,
            output_path: str = None,
            min_matches: int = 10,
            ext: int = 0
    ):
        """
        Perform diagnostics of astrometric offset of stars in image from catalogue.
        :param reference_cat: Path to reference catalogue.
        :param ra_col:
        :param dec_col:
        :param mag_col:
        :param offset_tolerance: Maximum offset to be matched.
        :param star_tolerance: Maximum CLASS_FLAG for object to be considered.
        :param local_coord:
        :param local_radius:
        :param show_plots:
        :param output_path:
        :return:
        """

        # quantity_support()

        if local_coord is None:
            local_coord = self.extract_pointing()

        if output_path is None:
            output_path = self.data_path

        self.source_cat.load_table()

        if isinstance(reference_cat, str):
            reference_cat = table.QTable.read(reference_cat)

        u.debug_print(2, "ImagingImage.astrometry_diagnostics(): reference_cat ==", reference_cat)

        plt.close()

        with quantity_support():
            plt.scatter(self.source_cat["RA"].value, self.source_cat["DEC"].value, marker='x')
            plt.xlabel("Right Ascension (Catalogue)")
            plt.ylabel("Declination (Catalogue)")
            # plt.colorbar(label="Offset of measured position from catalogue (\")")
            if show_plots:
                plt.show()
            plt.savefig(os.path.join(output_path, f"{self.name}_sourcecat_sky.pdf"))
            plt.close()

            plt.scatter(reference_cat[ra_col].value, reference_cat[dec_col].value, marker='x')
            plt.xlabel("Right Ascension (Catalogue)")
            plt.ylabel("Declination (Catalogue)")
            # plt.colorbar(label="Offset of measured position from catalogue (\")")
            if show_plots:
                plt.show()
            plt.savefig(os.path.join(output_path, f"{self.name}_referencecat_sky.pdf"))
            plt.close()

            self.load_wcs()
            ref_cat_coords = SkyCoord(reference_cat[ra_col], reference_cat[dec_col])
            in_footprint = self.wcs[ext].footprint_contains(ref_cat_coords)

            plt.scatter(
                self.source_cat["RA"],
                self.source_cat["DEC"],
                marker='x'
            )
            plt.scatter(
                reference_cat[ra_col][in_footprint],
                reference_cat[dec_col][in_footprint],
                marker='x'
            )
            plt.xlabel("Right Ascension (Catalogue)")
            plt.ylabel("Declination (Catalogue)")
            # plt.colorbar(label="Offset of measured position from catalogue (\")")
            if show_plots:
                plt.show()
            plt.savefig(os.path.join(output_path, f"{self.name}_bothcats_sky.pdf"))
            plt.close()

            matches_source_cat, matches_ext_cat, distance = self.match_to_cat(
                cat=reference_cat,
                ra_col=ra_col,
                dec_col=dec_col,
                offset_tolerance=offset_tolerance,
                # star_tolerance=star_tolerance
            )
            if len(matches_source_cat) < min_matches:
                self.astrometry_err = -99 * units.arcsec
                self.ra_err = -99 * units.arcsec
                self.dec_err = -99 * units.arcsec
                self.headers[0]["ASTM_RMS"] = self.astrometry_err.value
                self.headers[0]["RA_RMS"] = self.ra_err.value
                self.headers[0]["DEC_RMS"] = self.dec_err.value
                self.write_fits_file()
                self.update_output_file()
                return -99.0

            matches_coord = SkyCoord(matches_source_cat["RA"], matches_source_cat["DEC"])

            sigma_clip = SigmaClip(sigma=3.)
            distance_clipped = sigma_clip(distance, masked=False)
            distance_clipped_masked = sigma_clip(distance, masked=True)
            mask = ~distance_clipped_masked.mask

            offset_ra = matches_source_cat["RA"][mask] - matches_ext_cat[ra_col][mask]
            offset_dec = matches_source_cat["DEC"][mask] - matches_ext_cat[dec_col][mask]

            mean_offset = np.mean(distance_clipped)
            median_offset = np.median(distance_clipped)
            rms_offset = np.sqrt(np.mean(distance_clipped ** 2))
            rms_offset_ra = np.sqrt(np.mean(offset_ra ** 2))
            rms_offset_dec = np.sqrt(np.mean(offset_dec ** 2))

            ref = self.extract_pointing()
            ref_distance = ref.separation(matches_coord)

            local_distance = local_coord.separation(matches_coord)
            distance_local = distance[local_distance <= local_radius]
            u.debug_print(2, distance_local)
            mean_offset_local = np.mean(distance_local)
            median_offset_local = np.median(distance_local)
            rms_offset_local = np.sqrt(np.mean(distance_local ** 2))

            plt.scatter(ref_distance.to(units.arcsec), distance.to(units.arcsec))
            plt.xlabel("Distance from reference pixel (\")")
            plt.ylabel("Offset (\")")
            if show_plots:
                plt.show()
            plt.savefig(os.path.join(output_path, f"{self.name}_astrometry_offset_v_ref.pdf"))
            plt.close()

            plt.hist(
                distance.to(units.arcsec).value,
                bins=int(np.sqrt(len(distance))),
                label="Full sample"
            )
            plt.hist(
                distance_clipped.to(units.arcsec).value,
                edgecolor='black',
                linewidth=1.2,
                label="Sigma-clipped",
                fc=(0, 0, 0, 0),
                bins=int(np.sqrt(len(distance_clipped)))
            )
            plt.xlabel("Offset (\")")
            plt.legend()
            if show_plots:
                plt.show()
            plt.savefig(os.path.join(output_path, f"{self.name}_astrometry_offset_hist.pdf"))
            plt.close()

            plt.scatter(matches_ext_cat[ra_col], matches_ext_cat[dec_col], c=distance.to(units.arcsec), marker='x')
            plt.xlabel("Right Ascension (Catalogue)")
            plt.ylabel("Declination (Catalogue)")
            plt.colorbar(label="Offset of measured position from catalogue (\")")
            if show_plots:
                plt.show()
            plt.savefig(os.path.join(output_path, f"{self.name}_astrometry_offset_sky.pdf"))
            plt.close()

            fig = plt.figure(figsize=(12, 12), dpi=1000)
            self.plot_catalogue(
                cat=reference_cat[in_footprint],
                ra_col=ra_col, dec_col=dec_col,
                fig=fig,
                colour_column=mag_col,
                cbar_label=mag_col)
        # fig.savefig(os.path.join(output_path, f"{self.name}_cat_overplot.pdf"))

        self.astrometry_stats["mean_offset"] = mean_offset.to(units.arcsec)
        self.astrometry_stats["median_offset"] = median_offset.to(units.arcsec)
        self.astrometry_stats["rms_offset"] = rms_offset.to(units.arcsec)
        self.astrometry_stats["rms_offset_ra"] = rms_offset_ra.to(units.arcsec)
        self.astrometry_stats["rms_offset_dec"] = rms_offset_dec.to(units.arcsec)
        self.astrometry_stats["median_offset_x"] = np.nanmedian(matches_source_cat["X_OFFSET_FROM_REF"])
        self.astrometry_stats["median_offset_y"] = np.nanmedian(matches_source_cat["Y_OFFSET_FROM_REF"])
        self.astrometry_stats["mean_offset_x"] = np.nanmean(matches_source_cat["X_OFFSET_FROM_REF"])
        self.astrometry_stats["mean_offset_y"] = np.nanmean(matches_source_cat["Y_OFFSET_FROM_REF"])

        self.astrometry_stats["mean_offset_local"] = mean_offset_local.to(units.arcsec)
        self.astrometry_stats["median_offset_local"] = median_offset_local.to(units.arcsec)
        self.astrometry_stats["rms_offset_local"] = rms_offset_local.to(units.arcsec)

        self.astrometry_stats["n_matches"] = len(matches_source_cat)
        self.astrometry_stats["n_cat"] = sum(in_footprint)
        self.astrometry_stats["n_local"] = len(distance_local)
        self.astrometry_stats["local_coord"] = local_coord
        self.astrometry_stats["local_tolerance"] = local_radius
        # self.astrometry_stats["star_tolerance"] = star_tolerance
        self.astrometry_stats["offset_tolerance"] = offset_tolerance

        self.send_column_to_source_cat(column_name="OFFSET_FROM_REF", subset_table=matches_source_cat)
        self.send_column_to_source_cat(column_name="RA_OFFSET_FROM_REF", subset_table=matches_source_cat)
        self.send_column_to_source_cat(column_name="DEC_OFFSET_FROM_REF", subset_table=matches_source_cat)
        self.send_column_to_source_cat(column_name="PIX_OFFSET_FROM_REF", subset_table=matches_source_cat)
        self.send_column_to_source_cat(column_name="X_OFFSET_FROM_REF", subset_table=matches_source_cat)
        self.send_column_to_source_cat(column_name="Y_OFFSET_FROM_REF", subset_table=matches_source_cat)

        self.add_log(
            action=f"Calculated astrometry offset statistics.",
            method=self.astrometry_diagnostics,
            output_path=output_path
        )
        self.astrometry_err = self.astrometry_stats["rms_offset"]
        self.ra_err = self.astrometry_stats["rms_offset_ra"]
        self.dec_err = self.astrometry_stats["rms_offset_dec"]

        self.source_cat["RA_ERR"] = np.sqrt(
            self.source_cat["ERRX2_WORLD"].to(units.arcsec ** 2) + self.ra_err ** 2)
        self.source_cat["DEC_ERR"] = np.sqrt(
            self.source_cat["ERRY2_WORLD"].to(units.arcsec ** 2) + self.dec_err ** 2)

        if not np.isnan(self.astrometry_err.value):
            self.headers[0]["ASTM_RMS"] = self.astrometry_err.value
            self.headers[0]["RA_RMS"] = self.ra_err.value
            self.headers[0]["DEC_RMS"] = self.dec_err.value

        self.write_fits_file()
        self.update_output_file()

        return self.astrometry_stats

    def psf_diagnostics(
            self,
            mag_max: float = 0.0 * units.mag,
            mag_min: float = -50. * units.mag,
            match_to: table.Table = None,
            star_class_tol: int = 0,
            frame: float = None,
            ext: int = 0,
            target: SkyCoord = None,
            near_radius: units.Quantity = 1 * units.arcmin,
            output_path: str = None
    ):
        self.open()
        # self.source_cat.load_table()
        if frame is None:
            _, scale = self.extract_pixel_scale()
            frame = (4 * units.arcsec).to(units.pix, scale).value
        if output_path is None:
            output_path = self.data_path
        star_dict = ph.image_psf_diagnostics(
            hdu=self.hdu_list,
            cat=self.source_cat.table,
            mag_max=mag_max,
            mag_min=mag_min,
            match_to=match_to,
            frame=frame,
            near_centre=target,
            near_radius=near_radius,
            output=output_path,
            plot_file_prefix=self.name,
            ext=ext,
            star_class_tol=star_class_tol,
        )

        stars_gauss = star_dict["GAUSSIAN_FWHM_FITTED"]
        fwhm_gauss = stars_gauss["GAUSSIAN_FWHM_FITTED"]
        self.fwhm_median_gauss = np.nanmedian(fwhm_gauss)
        self.fwhm_max_gauss = np.nanmax(fwhm_gauss)
        self.fwhm_min_gauss = np.nanmin(fwhm_gauss)
        self.fwhm_sigma_gauss = np.nanstd(fwhm_gauss)
        self.fwhm_rms_gauss = np.sqrt(np.mean(fwhm_gauss ** 2))
        self.send_column_to_source_cat("GAUSSIAN_FWHM_FITTED", stars_gauss)

        stars_moffat = star_dict["MOFFAT_FWHM_FITTED"]
        fwhm_moffat = stars_moffat["MOFFAT_FWHM_FITTED"]
        self.fwhm_median_moffat = np.nanmedian(fwhm_moffat)
        self.fwhm_max_moffat = np.nanmax(fwhm_moffat)
        self.fwhm_min_moffat = np.nanmin(fwhm_moffat)
        self.fwhm_sigma_moffat = np.nanstd(fwhm_moffat)
        self.fwhm_rms_moffat = np.sqrt(np.mean(fwhm_moffat ** 2))
        self.send_column_to_source_cat("MOFFAT_FWHM_FITTED", stars_moffat)

        self.close()

        results = {
            "target": target,
            "radius": near_radius,
            "n_stars": len(stars_gauss),
            "fwhm_psfex": self.fwhm_psfex.to(units.arcsec),
            "gauss": {
                "fwhm_median": self.fwhm_median_gauss.to(units.arcsec),
                "fwhm_mean": np.nanmean(fwhm_gauss).to(units.arcsec),
                "fwhm_max": self.fwhm_max_gauss.to(units.arcsec),
                "fwhm_min": self.fwhm_min_gauss.to(units.arcsec),
                "fwhm_sigma": self.fwhm_sigma_gauss.to(units.arcsec),
                "fwhm_rms": self.fwhm_rms_gauss.to(units.arcsec),
                "n_stars": len(fwhm_gauss)
            },
            "moffat": {
                "fwhm_median": self.fwhm_median_moffat.to(units.arcsec),
                "fwhm_mean": np.nanmean(fwhm_moffat).to(units.arcsec),
                "fwhm_max": self.fwhm_max_moffat.to(units.arcsec),
                "fwhm_min": self.fwhm_min_moffat.to(units.arcsec),
                "fwhm_sigma": self.fwhm_sigma_moffat.to(units.arcsec),
                "fwhm_rms": self.fwhm_rms_moffat.to(units.arcsec)
            },

        }

        if "FWHM_WORLD" in star_dict:
            stars_se = star_dict["FWHM_WORLD"]
            fwhm_sextractor = stars_se["FWHM_WORLD"].to(units.arcsec)
            self.fwhm_median_sextractor = np.nanmedian(fwhm_sextractor)
            self.fwhm_max_sextractor = np.nanmax(fwhm_sextractor)
            self.fwhm_min_sextractor = np.nanmin(fwhm_sextractor)
            self.fwhm_sigma_sextractor = np.nanstd(fwhm_sextractor)
            self.fwhm_rms_sextractor = np.sqrt(np.mean(fwhm_sextractor ** 2))
            results["sextractor"] = {
                "fwhm_median": self.fwhm_median_sextractor.to(units.arcsec),
                "fwhm_mean": np.nanmean(fwhm_sextractor).to(units.arcsec),
                "fwhm_max": self.fwhm_max_sextractor.to(units.arcsec),
                "fwhm_min": self.fwhm_min_sextractor.to(units.arcsec),
                "fwhm_sigma": self.fwhm_sigma_sextractor.to(units.arcsec),
                "fwhm_rms": self.fwhm_rms_sextractor.to(units.arcsec)
            }

        self.headers[ext]["PSF_FWHM"] = self.fwhm_median_gauss.to(units.arcsec).value
        self.headers[ext]["PSF_FWHM_ERR"] = self.fwhm_sigma_gauss.to(units.arcsec).value
        self.add_log(
            action=f"Calculated PSF FWHM statistics.",
            method=self.psf_diagnostics,
            packages=["source-extractor", "psfex"]
        )
        self.psf_stats = results
        self.update_output_file()
        self.write_fits_file()
        return results, star_dict

    def trim(
            self,
            left: Union[int, units.Quantity] = None,
            right: Union[int, units.Quantity] = None,
            bottom: Union[int, units.Quantity] = None,
            top: Union[int, units.Quantity] = None,
            output_path: str = None,
            ext: int = 0
    ):

        left = u.dequantify(left, unit=units.pix)
        right = u.dequantify(right, unit=units.pix)
        bottom = u.dequantify(bottom, unit=units.pix)
        top = u.dequantify(top, unit=units.pix)

        if output_path is None:
            output_path = self.path.replace(".fits", "_trimmed.fits")
        image = self.copy_with_outputs(output_path)

        image.load_headers()
        image.load_data()

        header = image.headers[ext]

        trimmed_data, margins = u.trim_image(
            data=image.data[ext],
            left=left, right=right, bottom=bottom, top=top,
            return_margins=True
        )

        left, right, bottom, top = margins

        crpix1 = header['CRPIX1'] - left
        crpix2 = header['CRPIX2'] - bottom

        # Move reference pixel to account for trim; this should keep the same sky coordinate at the ref pix
        image.set_header_items(
            items={
                'CRPIX1': crpix1,
                'CRPIX2': crpix2
            },
            ext=ext,
            write=False
        )

        image.data[ext] = trimmed_data

        image.add_log(
            action=f"Trimmed image to margins left={left}, right={right}, bottom={bottom}, top={top}",
            method=self.trim,
            output_path=output_path
        )
        image.update_output_file()
        image.write_fits_file()

        return image

    # def convert_from_cs(self, output_path: str, ext: int = 0):
    #     """
    #     NOT IMPLEMENTED.
    #     Assuming units of counts / second, converts the image back to total counts.
    #     :param output_path: Path to write converted file to.
    #     :param ext: FITS extension to modify.
    #     :return new: ImagingImage object representing the modified file.
    #     """
    #     pass

    def convert_to_cs(self, output_path: str, ext: int = 0):
        """
        Converts the image to flux (units of counts per second) and writes to a new file.
        :param output_path: Path to write converted file to.
        :param ext: FITS extension to modify.
        :return new: ImagingImage object representing the modified file.
        """
        new = self.copy(output_path)
        gain = self.extract_gain()
        exp_time = self.extract_exposure_time()
        saturate = self.extract_saturate()
        read_noise = self.extract_noise_read()

        new.load_data()
        new_data = new.data[ext]
        # new_data *= gain
        new_data /= exp_time
        new.data[ext] = new_data

        u.debug_print(1, "Image.concert_to_cs() 2: new_data.unit ==", new_data.unit)

        header_keys = self.header_keys()
        new.set_header_items(
            items={
                header_keys["noise_read"]: str(new_data.unit),
                header_keys["gain"]: gain * exp_time,
                header_keys["gain_old"]: gain,
                header_keys["exposure_time"]: 1.0,
                header_keys["exposure_time_old"]: exp_time.value,
                header_keys["saturate_old"]: saturate,
                header_keys["saturate"]: saturate / exp_time.value,
                header_keys["noise_read"]: read_noise / exp_time.value,
                header_keys["noise_read_old"]: read_noise,
                header_keys["integration_time"]: exp_time.value
            },
            ext=ext,
            write=False
        )

        new.add_log(
            action=f"Converted image data on ext {ext} to cts / s, using exptime of {exp_time}.",
            method=self.convert_to_cs,
            output_path=output_path
        )

        new.write_fits_file()
        new.update_output_file()
        return new

    def clean_cosmic_rays(self, output_path: str, ext: int = 0):
        from ccdproc import cosmicray_lacosmic
        cleaned = self.copy(output_path)
        cleaned.load_data()
        data = cleaned.data[ext]
        gain = cleaned.extract_gain().value

        cleaned_data, mask = cosmicray_lacosmic(
            ccd=data,
            gain_apply=False,
            gain=gain,
            readnoise=cleaned.extract_noise_read().value,
            satlevel=cleaned.extract_saturate().value,
            verbose=True
        )

        cleaned.data[ext] = cleaned_data
        cleaned.write_fits_file()
        cleaned.add_log(
            action="Cleaned cosmic rays using LA cosmic algorithm.",
            method=self.clean_cosmic_rays,
            output_path=output_path,
            input_path=self.path,
        )
        cleaned.update_output_file()

    def scale_to_jansky(
            self,
            ext: int = 0,
            *args
    ):
        self.load_data()
        self.load_output_file()
        data = self.data[ext].value
        zp = self.zeropoint_best["zeropoint_img"].value
        exptime = self.extract_exposure_time().value
        data[data <= 0.] = np.min(data[data > 0.])
        data_scaled = 3631 * units.Jansky * (data / exptime) * 10 ** (zp / -2.5)
        extra_vals = []
        for v in args:
            if v is not None:
                v = u.dequantify(v)
                extra_vals.append(3631 * units.Jansky * (v / exptime) * 10 ** (zp / -2.5))
            else:
                extra_vals.append(v)
        if extra_vals:
            return data_scaled, extra_vals
        else:
            return data_scaled

    def pixel_magnitudes(
            self,
            ext: int = 0,
            sub_back: bool = False,
            back_kwargs: dict = {},
            **kwargs
    ):
        self.load_data()
        self.load_output_file()
        data = self.data[ext]
        if sub_back:
            _, bkg = self.model_background_photometry(**back_kwargs)
            data -= bkg
        data[data <= 0] = 0.
        return self.magnitude(
            data.value
        )

    def surface_brightness(
            self,
            ext: int = 0
    ):
        self.load_data()
        self.load_output_file()
        data = self.data[ext].value
        pix_mags = self.pixel_magnitudes()

    def reproject(
            self,
            other_image: 'ImagingImage',
            ext: int = 0,
            output_path: str = None,
            include_footprint: bool = False,
            write_footprint: bool = True,
            method: str = 'exact',
            mask_mode: bool = False
    ):
        import reproject as rp
        if output_path is None:
            output_path = self.path.replace(".fits", "_reprojected.fits")
        other_image.load_headers(force=True)
        print(f"Reprojecting {self.filename} into the pixel space of {other_image.filename}")
        if method == 'exact':
            reprojected, footprint = rp.reproject_exact(self.path, other_image.headers[ext], parallel=True)
        elif method == 'adaptive':
            reprojected, footprint = rp.reproject_adaptive(self.path, other_image.headers[ext])
        elif method in ['interp', 'interpolate', 'interpolation']:
            reprojected, footprint = rp.reproject_interp(self.path, other_image.headers[ext])
        else:
            raise ValueError(f"Reprojection method {method} not recognised.")

        # if not mask_mode:
        reprojected *= other_image.extract_unit(astropy=True)
        footprint *= units.pix
        if mask_mode:
            reprojected = np.round(reprojected)

        if output_path == self.path:
            reprojected_image = self
        else:
            reprojected_image = self.copy(output_path)
        reprojected_image.load_data(force=True)
        reprojected_image.data[ext] = reprojected

        if include_footprint:
            new_hdu = fits.ImageHDU()
            reprojected_image.headers.append(new_hdu.header)
            reprojected_image.data.append(footprint)

        if write_footprint:
            footprint_file = self.copy_with_outputs(output_path.replace(".fits", "_footprint.fits"))
            footprint_file.data[0] = footprint
            footprint_file.write_fits_file()

        reprojected_image.add_log(
            action=f"Reprojected into pixel space of {other_image}.",
            method=self.reproject,
            output_path=output_path
        )
        reprojected_image.update_output_file()
        reprojected_image.transfer_wcs(other_image=other_image)
        # reprojected_image.write_fits_file()

        return reprojected_image

    def trim_to_wcs(
            self,
            bottom_left: SkyCoord,
            top_right: SkyCoord,
            output_path: str = None, ext: int = 0
    ) -> 'ImagingImage':
        """
        Trims the image to a footprint defined by two RA/DEC coordinates
        :param bottom_left:
        :param top_right:
        :param output_path:
        :return:
        """
        self.load_wcs()
        left, bottom = bottom_left.to_pixel(wcs=self.wcs[ext], origin=0)
        right, top = top_right.to_pixel(wcs=self.wcs[ext], origin=0)
        return self.trim(left=left, right=right, bottom=bottom, top=top, output_path=output_path, ext=ext)

    def match_to_cat(
            self,
            cat: Union[str, table.QTable],
            ra_col: str = "ra",
            dec_col: str = "dec",
            offset_tolerance: units.Quantity = 1 * units.arcsec,
            star_tolerance: float = None,
            dual: bool = False,
            ext: int = 0
    ):

        source_cat = self.get_source_cat(dual=dual).table

        _, scale = self.extract_pixel_scale()

        if star_tolerance is not None:
            source_cat = u.trim_to_class(
                cat=source_cat,
                modify=True,
                allowed=np.arange(0, star_tolerance + 1)
            )

        u.debug_print(2, "len(source_cat) match_catalogs:", len(source_cat))
        matches_source_cat, matches_ext_cat, distance = astm.match_catalogs(
            cat_1=source_cat,
            cat_2=cat,
            ra_col_1="RA",
            dec_col_1="DEC",
            ra_col_2=ra_col,
            dec_col_2=dec_col,
            tolerance=offset_tolerance)

        self.load_wcs()
        x_cat, y_cat = self.wcs[ext].all_world2pix(matches_ext_cat[ra_col], matches_ext_cat[dec_col], 0)
        matches_ext_cat["x_image"] = x_cat
        matches_ext_cat["y_image"] = y_cat

        matches_source_cat["OFFSET_FROM_REF"] = distance.to(units.arcsec)
        matches_source_cat["RA_OFFSET_FROM_REF"] = matches_source_cat["RA"] - matches_ext_cat[ra_col]
        matches_source_cat["DEC_OFFSET_FROM_REF"] = matches_source_cat["DEC"] - matches_ext_cat[dec_col]

        matches_source_cat["PIX_OFFSET_FROM_REF"] = distance.to(units.pix, scale)

        matches_source_cat["X_OFFSET_FROM_REF"] = matches_source_cat["X_IMAGE"] - x_cat * units.pix
        matches_source_cat["Y_OFFSET_FROM_REF"] = matches_source_cat["Y_IMAGE"] - y_cat * units.pix

        return matches_source_cat, matches_ext_cat, distance

    def signal_to_noise_ccd(self, dual: bool = False):
        self.extract_exposure_time()
        self.extract_gain()
        self.aperture_areas()
        source_cat = self.get_source_cat(dual=dual)

        flux_target = source_cat['FLUX_AUTO']
        rate_target = flux_target / self.exposure_time
        rate_sky = source_cat['BACKGROUND'] / (self.exposure_time * units.pix)
        rate_read = self.extract_noise_read()
        n_pix = source_cat['KRON_AREA_IMAGE'] / units.pixel

        source_cat["SNR_CCD"] = ph.signal_to_noise_ccd_equ(
            rate_target=rate_target,
            rate_sky=rate_sky,
            rate_read=rate_read,
            exp_time=self.exposure_time,
            gain=self.gain,
            n_pix=n_pix
        ).value

        self.update_output_file()

        self.add_log(
            action=f"Estimated SNR using CCD Equation.",
            method=self.signal_to_noise_ccd,
        )
        self.update_output_file()

        return source_cat["SNR_CCD"]

    def signal_to_noise_measure(self, dual: bool = False):
        print("Measuring signal-to-noise of sources...")
        print(self.path)
        source_cat = self.get_source_cat(dual=dual)
        source_cat["SNR_AUTO"] = source_cat["FLUX_AUTO"] / source_cat["FLUXERR_AUTO"]
        if "FLUX_PSF" in source_cat.table.colnames:
            source_cat["SNR_PSF"] = source_cat["FLUX_PSF"] / source_cat["FLUXERR_PSF"]

        # self.load_data()
        # _, scale = self.extract_pixel_scale()
        # mask = self.generate_mask(method='sep')
        # mask = mask.astype(bool)
        # bkg, bkg_data = self.calculate_background(method='sep', mask=mask)
        # rms = bkg.rms()
        #
        # gain = self.extract_gain() / units.electron
        #
        # snrs = []
        # snrs_se = []
        # sigma_fluxes = []
        #
        # for cat_obj in source_cat:
        #     x = cat_obj["X_IMAGE"].value - 1
        #     y = cat_obj["Y_IMAGE"].value - 1
        #
        #     a = cat_obj["A_WORLD"].to(units.pix, scale).value
        #     b = cat_obj["B_WORLD"].to(units.pix, scale).value
        #
        #     theta = u.world_angle_se_to_pu(
        #         cat_obj["THETA_WORLD"],
        #         rot_angle=self.extract_rotation_angle()
        #     )
        #
        #     ap = photutils.aperture.EllipticalAperture(
        #         [x, y],
        #         a=a,
        #         b=b,
        #         theta=theta
        #     )
        #
        #     ap_mask = ap.to_mask(method='center')
        #
        #     flux = cat_obj["FLUX_AUTO"]
        #
        #     ap_rms = ap_mask.multiply(rms)
        #     sigma_flux = np.sqrt(ap_rms.sum()) * units.ct
        #     snr = flux / np.sqrt(sigma_flux ** 2 + flux / gain)
        #
        #     snr_se = flux / cat_obj["FLUXERR_AUTO"].value
        #
        #     snrs.append(snr.value)
        #     sigma_fluxes.append(sigma_flux.value)
        #     snrs_se.append(snr_se.value)
        #
        # source_cat["SNR_MEASURED"] = snrs
        # source_cat["NOISE_MEASURED"] = sigma_fluxes
        # source_cat["SNR_PSF"] = snrs_se

        self.add_log(
            action=f"Estimated SNR using SEP RMS map and Source Extractor uncertainty.",
            method=self.signal_to_noise_measure,
            packages=["source-extractor"]
        )
        self.update_output_file()

    def object_axes(self):
        self.extract_pixel_scale()
        self.source_cat.object_axes(pixel_scale=self.pixel_scale_y)
        self.source_cat_dual.object_axes(pixel_scale=self.pixel_scale_y)
        self.add_log(
            action=f"Created axis columns A_IMAGE, B_IMAGE in pixel units from A_WORLD, B_WORLD.",
            method=self.object_axes,
        )
        self.update_output_file()

    def estimate_sky_background(self, ext: int = 0, force: bool = False):
        """
        Estimates background as a global median. VERY loose estimate.
        :param ext:
        :param force:
        :return:
        """
        if force or self.sky_background is None:
            self.load_data()
            self.sky_background = np.nanmedian(self.data[ext]) * units.ct / units.pixel
        else:
            print("Sky background already estimated.")
        return self.sky_background

    def find_object(self, coord: SkyCoord, dual: bool = True):
        cat = self.get_source_cat(dual=dual)
        u.debug_print(2, f"{self}.find_object(): dual ==", dual)
        coord_cat = cat.to_skycoord()
        separation = coord.separation(coord_cat)
        i = np.argmin(separation)
        nearest = cat[i]
        return nearest, separation[i]

    def find_object_index(self, index: int, dual: bool = True):
        """
        Using NUMBER column, finds the row referred to.
        :param index:
        :param dual:
        :return:
        """
        source_cat = self.get_source_cat(dual=dual)
        i, _ = u.find_nearest(source_cat["NUMBER"], index)
        return source_cat[i], i

    def pixel(
            self,
            value: Union[float, int, units.Quantity],
            z: float = None,
            obj: objects.Extragalactic = None,
            ext: int = 0
    ):
        value = u.check_quantity(
            number=value,
            unit=units.pix,
            allow_mismatch=True,
            enforce_equivalency=False
        )

        if not value.unit.is_equivalent(units.pix):
            if value.unit.is_equivalent(units.m):
                if obj is None:
                    if z is None:
                        raise ValueError("If `value` is in units of physical size, then `obj` or `z` must be provided.")
                    else:
                        obj = objects.Extragalactic(z=z)
                value = obj.angular_size(
                    distance=value
                )
            self.extract_pixel_scale(ext)
            value = value.to(units.pix, self.pixel_scale_y)

        return value

    def frame_from_coord(
            self,
            frame: units.Quantity,
            centre: SkyCoord,
            ext: int = 0,

    ):
        frame = self.pixel(frame, ext=ext)
        self.load_data()
        x, y = self.world_to_pixel(centre, 0)
        return u.frame_from_centre(frame=frame.value, x=x, y=y, data=self.data[ext])

    def prep_for_colour(
            self,
            output_path: str,
            frame: units.Quantity,
            centre: SkyCoord = None,
            vmax: float = None,
            vmin: float = None,
            ext: int = 0,
            scale_to_jansky: bool = False
    ):
        left, right, bottom, top = self.frame_from_coord(
            frame=frame,
            centre=centre,
            ext=ext
        )
        trimmed = self.trim(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            output_path=output_path
        )
        trimmed.load_wcs()

        if scale_to_jansky:
            data, vs = trimmed.scale_to_jansky(ext, vmax, vmin)
            vmax = u.dequantify(vs[0])
            vmin = u.dequantify(vs[1])
            data = data.value
        else:
            data = trimmed.data[0].value

        if vmax is not None:
            data[data > vmax] = vmax
        if vmin is not None:
            data[data < vmin] = vmin

        median = np.nanmedian(data)
        data_subbed = data - median
        data_subbed[np.isnan(data_subbed)] = median
        # data_scaled = data_subbed * 255 / np.max(data_subbed)
        return data_subbed, trimmed

    def nice_frame(
            self,
            row: Union[table.Row, dict],
            frame: units.Quantity = 10 * units.pix,
    ):
        self.extract_pixel_scale()
        u.debug_print(1, "ImagingImage.nice_frame(): row['KRON_RADIUS'], row['A_WORLD'] ==", row['KRON_RADIUS'],
                      row['A_WORLD'].to(units.arcsec))
        kron_a = row['KRON_RADIUS'] * row['A_WORLD']
        u.debug_print(1, "ImagingImage.nice_frame(): kron_a ==", kron_a)
        pix_scale = self.pixel_scale_y
        u.debug_print(1, "ImagingImage.nice_frame(): self.pixel_scale_dec ==", self.pixel_scale_y)
        this_frame = max(
            kron_a.to(units.pixel, pix_scale), frame)  # + 5 * units.pix,
        u.debug_print(1, "ImagingImage.nice_frame(): this_frame ==", this_frame)
        return this_frame

    def plot_apertures(self, dual=False, output: str = None, show: bool = False):
        cat = self.get_source_cat(dual=dual).table

        if cat is not None:
            pl.plot_all_params(image=self.path, cat=cat, kron=True, show=False)
            plt.title(self.filter_name)
            if output is None:
                output = os.path.join(self.data_path, f"{self.name}_source_cat_dual-{dual}.pdf")
            plt.savefig(output)
            if show:
                plt.show()

    def plot_subimage(
            self,
            centre: SkyCoord = None,
            frame: units.Quantity = None,
            corners: Tuple[SkyCoord] = None,
            ext: int = 0,
            fig: plt.Figure = None,
            ax: plt.Axes = None,
            n: int = 1, n_x: int = 1, n_y: int = 1,
            show_grid: bool = False,
            show_coords: bool = True,
            imshow_kwargs: dict = None,  # Can include cmap
            normalize_kwargs: dict = None,  # Can include vmin, vmax
            output_path: str = None,
            mask: np.ndarray = None,
            scale_bar_object: objects.Extragalactic = None,
            scale_bar_kwargs: dict = None,
            data: str = "image",
            clip_data: bool = False,
            **kwargs,
    ) -> Tuple[plt.Axes, plt.Figure, dict]:

        if data == "image":
            self.load_data()
            data = self.data[ext].value
        elif data == "background":
            _, data = self.model_background_photometry(**kwargs)
        elif data == "background_subtracted_image":
            self.model_background_photometry(**kwargs)
            data = self.data_sub_bkg[ext]
        elif data == "pixel_magnitudes":
            data, _, _, _ = self.pixel_magnitudes(**kwargs)
        elif isinstance(data, np.ndarray):
            data = data
        else:
            raise ValueError(
                f"data_type {data} not recognised; this can be 'image', 'background', or 'background_subtracted_image'")

        _, scale = self.extract_pixel_scale()

        other_args = {}
        if centre is not None and frame is not None:
            x, y = self.world_to_pixel(centre, 0)
            if "z" in kwargs:
                z = kwargs["z"]
            else:
                z = None
            if "obj" in kwargs:
                obj = kwargs["obj"]
            else:
                obj = None
            frame = self.pixel(
                value=frame,
                z=z,
                obj=obj
            )
            other_args["x"] = x
            other_args["y"] = y
            left, right, bottom, top = u.frame_from_centre(frame.value, x, y, data)
        elif corners is not None:
            x_0, y_0 = self.world_to_pixel(corners[0], 0)
            x_1, y_1 = self.world_to_pixel(corners[1], 0)
            xs = x_1, x_0
            left = int(min(xs))
            right = int(max(xs))
            ys = y_1, y_0
            bottom = int(min(ys))
            top = int(max(ys))

        else:
            left = 0
            right = data.shape[1]
            bottom = 0
            top = data.shape[0]

        # print(type(data), data[bottom:top, left:right])
        if mask is not None:
            data_masked = data * np.invert(mask.astype(bool)).astype(int)
            data_masked += mask * np.nanmedian(data[bottom:top, left:right])
            data = data_masked
        u.debug_print(1, "ImagingImage.plot_subimage(): frame ==", frame)

        if fig is None:
            fig = plt.figure()
        if normalize_kwargs is None:
            normalize_kwargs = {}
        if imshow_kwargs is None:
            imshow_kwargs = {}

        if "stretch" not in normalize_kwargs:
            normalize_kwargs["stretch"] = SqrtStretch()
        elif normalize_kwargs["stretch"] == "sqrt":
            normalize_kwargs["stretch"] = SqrtStretch()
        elif normalize_kwargs["stretch"] == "log":
            normalize_kwargs["stretch"] = LogStretch()

        if "interval" not in normalize_kwargs:
            normalize_kwargs["interval"] = MinMaxInterval()
        elif normalize_kwargs["interval"] == "minmax":
            normalize_kwargs["interval"] = MinMaxInterval()
        elif normalize_kwargs["interval"] == "zscale":
            normalize_kwargs["interval"] = ZScaleInterval()

        if "origin" not in imshow_kwargs:
            imshow_kwargs["origin"] = "lower"

        if ax is None:
            if show_coords:
                projection = self.wcs[ext]
            else:
                projection = None

            ax = fig.add_subplot(n_y, n_x, n, projection=projection)

        if not show_coords:
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.set_yticks([])
            frame1.axes.invert_yaxis()

        if "data" not in normalize_kwargs:
            scaling_data = data[bottom:top, left:right]
            if clip_data:
                sigma_clip = SigmaClip(sigma_lower=2., sigma_upper=np.inf)
                scaling_data = sigma_clip(scaling_data, masked=False)
        else:
            scaling_data = normalize_kwargs.pop("data")

        if "cmap" not in imshow_kwargs:
            if self.filter and self.filter.cmap:
                imshow_kwargs["cmap"] = self.filter.cmap

        # if "vmin" not in normalize_kwargs:
        #     normalize_kwargs["vmin"] = np.min(data_clipped)

        mapping = ax.imshow(
            data,
            norm=ImageNormalize(
                scaling_data,
                **normalize_kwargs
            ),
            interpolation="none",
            **imshow_kwargs
        )

        other_args["mapping"] = mapping
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        ax.set_xlabel("Right Ascension (J2000)", size=16)
        ax.set_ylabel("Declination (J2000)", size=16, rotation=-90)
        # ax.tick_params(labelsize=14)
        # ax.yaxis.set_label_position("right")

        # plt.tight_layout()

        if scale_bar_object is not None:
            if scale_bar_kwargs is None:
                scale_bar_kwargs = {}
            print(scale_bar_kwargs)
            self.scale_bar(
                obj=scale_bar_object,
                ax=ax,
                fig=fig,
                **scale_bar_kwargs
            )
        if output_path is not None:
            fig.savefig(output_path)

        del data

        return ax, fig, other_args

    def scale_bar(
            self,
            obj: objects.Extragalactic,
            ax: plt.Axes,
            fig: plt.Figure,
            size: units.Quantity = 1 * units.arcsec,
            spread_factor: float = 0.5,
            x_ax: float = 0.1,
            y_ax: float = 0.1,
            line_kwargs: dict = None,
            text_kwargs: dict = None,
            ext: int = 0,
            extra_height_top_factor: float = 2.
    ):
        self.extract_pixel_scale(ext=ext)
        if line_kwargs is None:
            line_kwargs = {}
        if text_kwargs is None:
            text_kwargs = {}

        if "fontsize" not in text_kwargs:
            text_kwargs["fontsize"] = 12
        if "color" not in text_kwargs:
            text_kwargs["color"] = "white"

        if "color" not in line_kwargs:
            line_kwargs["color"] = "white"
        if "lw" not in line_kwargs:
            line_kwargs["lw"] = 3

        # if isinstance(x, units.Quantity):
        #     if x.decompose().unit == units.rad:
        #         x = x.to(units.pix, self.pixel_scale_x)
        # if isinstance(x, units.Quantity):
        #     if x.decompose().unit == units.rad:
        #         x = x.to(units.pix, self.pixel_scale_x)

        if not isinstance(size, units.Quantity):
            size = size * units.pix
        if size.decompose().unit == units.pix:
            size_pix = size
            size_ang = size_pix.to(units.arcsec, self.pixel_scale_x)
            size_proj = obj.projected_size(size_ang)
        elif size.decompose().unit == units.meter:
            size_proj = size
            size_ang = obj.angular_size(distance=size)
            size_pix = size_ang.to(units.pix, self.pixel_scale_x)
        elif size.decompose().unit == units.rad:
            size_ang = size
            size_pix = size_ang.to(units.pix, self.pixel_scale_x)
            size_proj = obj.projected_size(size_ang)
        else:
            raise ValueError(f"The units of provided size, {size.unit}, cannot be parsed as a pixel, angular or "
                             f"physical distance.")

        if "solid_capstyle" not in line_kwargs:
            line_kwargs["solid_capstyle"] = "butt"

        # Draw angular size text in axes coordinates
        text_ang = ax.text(
            x_ax,
            y_ax,
            size_ang.round(1),
            transform=ax.transAxes,
            **text_kwargs
        )
        # The below seems complicated, but is made necessary by the fact that you only seem to be able to get the text
        # width out of matplotlib in Display coordinates (ie, rendered pixels), and the zero point (0, 0) of this
        # differs from both the Axes coordinates (0, 0) and the Data coordinates (0, 0), but in different ways.

        # Get the size of the text on the canvas
        r = fig.canvas.get_renderer()
        bb = text_ang.get_window_extent(r)
        # Transform the x and y axis coordinates to Display coordinates
        x_disp, y_ang_disp = ax.transAxes.transform((x_ax, y_ax))
        # Get the rightmost point of the text by adding the bounding box width (we don't actually use this right now,
        # but I'm leaving it here in case I make changes later and forget how this works)
        x_ang_disp_right = x_disp + bb.width
        # Get the topmost point of the text by adding bounding box height
        y_ang_disp_up = y_ang_disp + bb.height
        # Space the bar upwards by half the height of the text.
        y_bar_disp = y_ang_disp_up + bb.height * spread_factor
        # Transform the left bar coordinates back to Axes coordinates.
        x_ax, y_bar_ax = ax.transAxes.inverted().transform((x_disp, y_bar_disp))
        # Now, to get the bar's right points, we need to work in Data coordinates, because that's what the size is in
        # (that is, in DATA pixels).
        # Transform our left point from Display coordinates to Data coordinates.
        x_data, y_bar_data = ax.transData.inverted().transform((x_disp, y_bar_disp))
        # Add the width of the bar.
        x_bar_data_right = x_data + size_pix.value
        # Transform right point back to Display coordinates.
        x_bar_disp_right, y_bar_disp = ax.transData.transform((x_bar_data_right, y_bar_data))
        # Transform right point to Axes coordinates.
        x_bar_ax_right, y_bar_ax = ax.transAxes.inverted().transform((x_bar_disp_right, y_bar_disp))
        # Draw the bar.
        ax.plot(
            (x_ax, x_bar_ax_right),
            (y_bar_ax, y_bar_ax),
            transform=ax.transAxes,
            **line_kwargs
        )
        # Add a text height to get where we draw the projected distance text (since the font size is the same, no
        # need to do any more nasty conversions)
        y_proj_disp = y_bar_disp + extra_height_top_factor * bb.height * spread_factor
        # Except for this one, where we transform the final text coordinates back to Axes coordinates
        x_ax, y_proj_ax = ax.transAxes.inverted().transform((x_disp, y_proj_disp))
        # Draw the projected size text.
        ax.text(
            x_ax,
            y_proj_ax,
            size_proj.round(1),
            transform=ax.transAxes,
            **text_kwargs
        )

        # I am a matplotlib god.

        return ax

    def plot_source_extractor_object(
            self,
            row: table.Row,
            ext: int = 0,
            frame: units.Quantity = 10 * units.pix,
            output: str = None,
            show: bool = False,
            title: str = None,
            find: SkyCoord = None
    ):

        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot()

        self.load_headers()
        kron_a = row['KRON_RADIUS'] * row['A_WORLD']
        kron_b = row['KRON_RADIUS'] * row['B_WORLD']
        # kron_theta = -kron_theta + ff.get_rotation_angle(
        #     header=self.headers[ext],
        #     astropy_units=True)
        this_frame = self.nice_frame(row=row, frame=frame)
        mid_x = row["X_IMAGE"]
        mid_y = row["Y_IMAGE"]
        self.open()
        left, right, bottom, top = u.frame_from_centre(frame=this_frame, x=mid_x, y=mid_y, data=self.hdu_list[ext].data)
        image_cut = ff.trim(hdu=self.hdu_list, left=left, right=right, bottom=bottom, top=top)
        norm = pl.nice_norm(image=image_cut[ext].data)
        ax.imshow(image_cut[ext].data, origin='lower', norm=norm)
        # theta =
        pl.plot_gal_params(
            hdu=image_cut,
            ras=[row["RA"].value],
            decs=[row["DEC"].value],
            a=[row["A_WORLD"].value],
            b=[row["B_WORLD"].value],
            theta=[row["THETA_IMAGE"].value],
            world=True,
            show_centre=True
        )
        pl.plot_gal_params(
            hdu=image_cut,
            ras=[row["RA"].value],
            decs=[row["DEC"].value],
            a=[kron_a.value],
            b=[kron_b.value],
            theta=[row["THETA_IMAGE"].value],
            world=True,
            show_centre=True
        )
        if title is None:
            title = self.name
        title = u.latex_sanitise(title)
        if find is not None:
            x_find, y_find = self.world_to_pixel(find)
            x_find -= left
            y_find -= bottom
            ax.scatter(x_find, y_find, c="red", marker="x")
        ax.set_title(title)
        fig.savefig(os.path.join(output))
        if show:
            fig.show()
        self.close()
        plt.close(fig)
        return

    def plot(
            self,
            ax: plt.Axes = None,
            fig: plt.Figure = None,
            ext: int = 0,
            **kwargs
    ):

        if fig is None:
            fig = plt.figure(figsize=(12, 12), dpi=1000)
        if ax is None:
            ax, fig = self.wcs_axes(fig=fig, ext=ext)
        self.load_data()
        data = u.dequantify(self.data[ext])
        ax.imshow(
            data, **kwargs,
            norm=ImageNormalize(
                data,
                interval=MinMaxInterval(),
                stretch=SqrtStretch(),
                vmin=np.nanmedian(data),
            ),
            origin='lower',
        )
        return ax, fig

    def wcs_axes(self, fig: plt.Figure = None, ext: int = 0):
        if fig is None:
            fig = plt.figure(figsize=(12, 12), dpi=1000)
        ax = fig.add_subplot(
            projection=self.load_wcs()[ext]
        )
        return ax, fig

    def plot_catalogue(
            self,
            cat: table.QTable,
            ra_col: str = "ra",
            dec_col: str = "dec",
            colour_column: str = None,
            fig: plt.Figure = None,
            ext: int = 0,
            cbar_label: str = None,
            **kwargs
    ):
        if fig is None:
            fig = plt.figure()
        if colour_column is not None:
            c = u.dequantify(cat[colour_column])
        else:
            c = "red"

        ax, fig = self.plot(fig=fig, ext=ext, zorder=0, **kwargs)
        x, y = self.wcs[ext].all_world2pix(cat[ra_col], cat[dec_col], 0)
        pcm = plt.scatter(x, y, c=c, cmap="plasma", marker="x", zorder=10)
        if colour_column is not None:
            fig.colorbar(pcm, ax=ax, label=cbar_label)

        u.debug_print(2, f"{self}.plot_catalogue(): len(cat):", len(cat))

        return ax, fig

    def plot_coords(
            self,
            ax: plt.Axes,
            coord: Union[SkyCoord, List[SkyCoord]],
            **kwargs
    ):
        coord = u.check_iterable(coord)
        for i, c in enumerate(coord):
            coord[i] = astm.attempt_skycoord(c)
        coord = SkyCoord(coord)
        x, y = self.world_to_pixel(coord=coord)
        if "marker" not in kwargs:
            kwargs["marker"] = "x"
        if "c" not in kwargs:
            kwargs["c"] = "white"
        ax.scatter(
            x, y, **kwargs
        )
        return ax

    def plot_slit(
            self,
            ax: plt.Axes,
            centre: SkyCoord,
            width: units.Quantity,
            length: units.Quantity,
            position_angle: units.Quantity,
            **kwargs
    ):
        # Multiplying by the "y-sense" (the 1,1 entry on the wcs pixel scale matrix) accounts for images where the
        # y-axis is reversed wrt north (eg certain GMOS images)
        position_angle = u.check_quantity(
            position_angle,
            units.deg) * self.extract_y_sense() + self.extract_rotation_angle()
        centre_x, centre_y = self.world_to_pixel(centre)
        slit_width = self.pixel(width).value
        slit_length = self.pixel(length).value
        # Do some trigonometry to determine pixel coordinates for the Rectangle badge (which uses the corner as its origin. Thanks, matplotlib.)
        rec_x = centre_x + np.sin(position_angle) * slit_length / 2 - np.cos(position_angle) * slit_width / 2
        rec_y = centre_y - np.cos(position_angle) * slit_length / 2 - np.sin(position_angle) * slit_width / 2

        default_kwargs = dict(
            linewidth=2,
            edgecolor='white',
            facecolor='none'
        )
        default_kwargs.update(kwargs)

        rect = Rectangle(
            (rec_x, rec_y),
            slit_width,
            slit_length,
            angle=position_angle.value,
            **default_kwargs
        )
        ax.add_artist(rect)
        return ax

    def insert_synthetic_sources(
            self,
            x: np.float64, y: np.float64,
            mag: np.float64,
            output: str,
            overwrite: bool = True,
            world_coordinates: bool = False,
            extra_values: table.Table = None,
            model: str = "psfex"
    ):
        if self.psfex_path is None:
            raise ValueError(f"{self.name}.psfex_path has not been set.")
        if self.zeropoint_best is None:
            raise ValueError(f"{self.name}.zeropoint_best has not been set.")
        output_cat = output.replace('.fits', '_synth_cat.ecsv')

        self.extract_pixel_scale()

        # TODO: Fix treatment of zeropoint here
        if model == "gaussian":
            file, sources = ph.insert_point_sources_to_file(
                file=self.path,
                x=x, y=y, mag=mag,
                zeropoint=self.zeropoint_best["zeropoint"],
                airmass=self.extract_airmass(),
                extinction=self.zeropoint_best["extinction"],
                exp_time=self.extract_exposure_time(),
                world_coordinates=world_coordinates,
                extra_values=extra_values,
                output=output,
                output_cat=output_cat,
                overwrite=overwrite,
                fwhm=self.fwhm_psfex.to(units.pix, self.pixel_scale_y)
            )
        elif model == "psfex":
            file, sources = ph.insert_point_sources_to_file(
                file=self.path,
                x=x, y=y, mag=mag,
                psf_model=self.psfex_path,
                zeropoint=self.zeropoint_best["zeropoint"],
                airmass=self.extract_airmass(),
                extinction=self.zeropoint_best["extinction"],
                exp_time=self.extract_exposure_time(),
                world_coordinates=world_coordinates,
                extra_values=extra_values,
                output=output,
                output_cat=output_cat,
                overwrite=overwrite
            )
        else:
            raise ValueError(f"Model {model} not recognised.")

        inserted = self.new_image(output)
        u.debug_print(1, "ImagingImage.insert_synthetic_sources: output_cat", output_cat)
        inserted.synth_cat_path = output_cat
        inserted.add_log(
            action=f"Injected synthetic point-sources, defined in output catalogue at {output_cat}.",
            method=self.insert_synthetic_sources,
            output_path=output
        )
        inserted.update_output_file()
        return inserted, sources

    def insert_synthetic_range(
            self,
            x: float = None, y: float = None,
            mag_min: units.Quantity = 20.0 * units.mag,
            mag_max: units.Quantity = 30.0 * units.mag,
            interval: units.Quantity = 0.1 * units.mag,
            output_dir: str = None,
            filename: str = None,
            model: str = "psfex",
            positioning: str = "inplace",
            scale: units.Quantity = 10 * units.arcsec
    ):
        x = float(x)
        y = float(y)

        x_ref, y_ref = self.extract_ref_pixel()
        if x is None:
            x = x_ref
        if y is None:
            y = y_ref

        inserted = []
        cats = []
        if filename is None:
            filename = f"{self.name}_insert"
        for mag in np.linspace(mag_min, mag_max, int((mag_max - mag_min) / interval + 1)):
            u.debug_print(1, f"INSERTING SOURCE {mag}")

            if positioning == 'inplace':
                x_synth = x
                y_synth = y
            elif positioning == 'gaussian':
                self.extract_pixel_scale()
                scale.to(units.pix, self.pixel_scale_y)
                x_synth = -1
                y_synth = -1
                self.extract_n_pix()
                x_max, y_max = self.n_x, self.n_y
                while not (0 < x_synth < x_max) or not (0 < y_synth < y_max):
                    x_synth, y_synth = gaussian_distributed_point(x_0=x, y_0=y, sigma=scale.value)
            else:
                raise ValueError(f"positioning {positioning} not recognised.")

            file, sources = self.insert_synthetic_sources(
                x=x_synth, y=y_synth, mag=mag.value,
                output=os.path.join(output_dir, filename + f"_mag_{np.round(mag.value, 1)}.fits"),
                model=model
            )
            file.source_extraction_psf(output_dir=output_dir)
            file.zeropoint_best = self.zeropoint_best
            file.calibrate_magnitudes()
            file.signal_to_noise_ccd()
            inserted.append(file)
            cat = file.check_synthetic_sources()
            cats.append(cat)

        cat_all = table.vstack(cats)
        cat_all["distance_from_ref"] = np.sqrt(
            (cat_all["x_inserted"] - x) ** 2 + (cat_all["y_inserted"] - y) ** 2) * units.pix
        return cat_all

    def check_synthetic_sources(self):
        """
        Checks on the fidelity of inserted sources against catalogue.
        :return:
        """

        self.signal_to_noise_measure()
        self.signal_to_noise_ccd()

        self.load_synth_cat()
        if self.synth_cat is None:
            raise ValueError("No synth_cat present.")

        matches_source_cat, matches_synth_cat, distance = self.match_to_cat(
            cat=self.synth_cat,
            ra_col='ra_inserted',
            dec_col='dec_inserted',
            offset_tolerance=1.0 * units.arcsec,
            # star_tolerance=0.7,
        )

        aperture_radius = 2 * self.fwhm_pix_psfex

        self.synth_cat["flux_sep"], self.synth_cat["flux_sep_err"], _ = self.sep_aperture_photometry(
            aperture_radius=aperture_radius,
            x=self.synth_cat["x_inserted"],
            y=self.synth_cat["y_inserted"]
        )

        self.synth_cat["mag_sep"], self.synth_cat["mag_sep_err"], _, _ = self.magnitude(
            flux=self.synth_cat["flux_sep"],
            flux_err=self.synth_cat["flux_sep_err"]
        )

        self.synth_cat["delta_mag_sep"] = self.synth_cat["mag_sep"] - self.synth_cat["mag_inserted"]
        self.synth_cat["fraction_flux_recovered_sep"] = self.synth_cat["flux_sep"] / self.synth_cat["flux_inserted"]
        self.synth_cat["snr_sep"] = self.synth_cat["flux_sep"] / self.synth_cat["flux_sep_err"]
        self.synth_cat["aperture_radius"] = aperture_radius

        matches_source_cat["matching_dist"] = distance.to(units.arcsec)
        matches_source_cat["fraction_flux_recovered_auto"] = matches_source_cat["FLUX_AUTO"] / matches_synth_cat[
            "flux_inserted"]
        matches_source_cat["fraction_flux_recovered_psf"] = matches_source_cat["FLUX_PSF"] / matches_synth_cat[
            "flux_inserted"]
        matches_source_cat["delta_mag_auto"] = matches_source_cat["MAG_AUTO_ZP_best"] - matches_synth_cat[
            "mag_inserted"]
        matches_source_cat["delta_mag_psf"] = matches_source_cat["MAG_PSF_ZP_best"] - matches_synth_cat["mag_inserted"]

        if len(matches_source_cat) > 0:
            self.synth_cat = table.hstack([self.synth_cat, matches_source_cat])

        self.add_log(
            action=f"Created catalogue of synthetic sources and their measurements.",
            method=self.check_synthetic_sources,
            output_path=self.synth_cat_path
        )
        self.update_output_file()

        return self.synth_cat

    def test_limit_location(
            self,
            coord: SkyCoord,
            ap_radius: units.Quantity = None,
            ext: int = 0,
            sigma_min: int = 1,
            sigma_max: int = 10,
            **kwargs
    ):

        if ap_radius is None:
            psf = self.extract_header_item("PSF_FWHM", ext=ext) * units.arcsec
            if not psf:
                ap_radius = 2 * units.arcsec
            else:
                ap_radius = 2 * psf

        self.load_wcs()
        _, pix_scale = self.extract_pixel_scale()
        x, y = self.wcs[ext].all_world2pix(coord.ra, coord.dec, 0)
        ap_radius_pix = ap_radius.to(units.pix, pix_scale).value

        self.model_background_photometry(method="sep", do_mask=True, ext=ext, **kwargs)
        rms = self.sep_background[ext].rms()

        flux, _, _ = sep.sum_circle(rms ** 2, [x], [y], ap_radius_pix)
        sigma_flux = np.sqrt(flux)

        limits = []
        for i in range(sigma_min, sigma_max + 1):
            n_sigma_flux = sigma_flux * i
            limit, _, _, _ = self.magnitude(flux=n_sigma_flux)
            limits.append({
                "sigma": i,
                "flux": n_sigma_flux[0],
                "mag": limit[0]
            })
        return table.QTable(limits)

    def test_limit_synthetic(
            self,
            coord: SkyCoord = None,
            output_dir: str = None,
            positioning: str = "inplace",
            mag_min: units.Quantity = 20.0 * units.mag,
            mag_max: units.Quantity = 30.0 * units.mag,
            interval: units.Quantity = 0.1 * units.mag,
            ext: int = 0
    ):

        if output_dir is None:
            output_dir = os.path.join(self.data_path, f"{self.name}_lim_test")
        u.mkdir_check(output_dir)
        if SkyCoord is None:
            coord = self.extract_pointing()
        self.load_wcs()
        x, y = self.wcs[ext].all_world2pix(coord.ra, coord.dec, 0)
        sources = self.insert_synthetic_range(
            x=x, y=y,
            mag_min=mag_min,
            mag_max=mag_max,
            interval=interval,
            output_dir=output_dir,
            positioning=positioning
        )

        plt.scatter(sources["mag_inserted"], sources["fraction_flux_recovered_psf"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Fraction of flux recovered")
        plt.savefig(os.path.join(output_dir, "flux_recovered_psf.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["fraction_flux_recovered_auto"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Fraction of flux recovered")
        plt.savefig(os.path.join(output_dir, "flux_recovered_auto.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["fraction_flux_recovered_sep"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Fraction of flux recovered")
        plt.savefig(os.path.join(output_dir, "flux_recovered_sep.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["delta_mag_psf"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Mag psf - mag inserted")
        plt.savefig(os.path.join(output_dir, "delta_mag_psf.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["delta_mag_auto"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Mag auto - mag inserted")
        plt.savefig(os.path.join(output_dir, "delta_mag_auto.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["delta_mag_sep"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Mag auto - mag inserted")
        plt.savefig(os.path.join(output_dir, "delta_mag_sep.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["CLASS_STAR"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Class star")
        plt.savefig(os.path.join(output_dir, "class_star.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["SPREAD_MODEL"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Spread Model")
        plt.savefig(os.path.join(output_dir, "spread_model.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["matching_dist"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("Matching distance (arcsec)")
        plt.savefig(os.path.join(output_dir, "matching_dist.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["snr_sep"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("S/N, measured by SEP")
        plt.savefig(os.path.join(output_dir, "matching_dist.png"))
        plt.close()

        plt.scatter(sources["mag_inserted"], sources["SNR_PSF"])
        plt.xlabel("Inserted magnitude")
        plt.ylabel("S/N, measured by SEP")
        plt.savefig(os.path.join(output_dir, "matching_dist.png"))
        plt.close()

        # TODO: S/N measure and plot

        ax, fig = self.plot_catalogue(cat=sources, ra_col="ra_inserted", dec_col="dec_inserted")
        fig.savefig(os.path.join(output_dir, "inserted_overplot.png"))

        sources.write(os.path.join(output_dir, "synth_cat_all.ecsv"), format="ascii.ecsv")

        self.add_log(
            action=f"Created catalogue of synthetic sources from range of insertions and their measurements.",
            method=self.test_limit_synthetic,
            output_path=output_dir
        )
        self.update_output_file()

        return sources

    def model_background_local(
            self,
            centre: SkyCoord,
            frame: units.Quantity,
            ext: int = 0,
            model_type: models = models.Polynomial2D,
            fitter_type: fitting.Fitter = fitting.LevMarLSQFitter,
            init_params: dict = {"degree": 3},
            write: str = None,
            write_subbed: str = None,
            generate_mask: bool = True,
            mask_ellipses: List[dict] = None,
            mask_kwargs: dict = {},
            saturate_factor: float = 0.5,
    ):
        """
        Models and subtracts the background of a specified part of the image.
        A mask can be generated using either `sep` or `photutils`, automatically excluding objects from the background
        fit. Negative and saturated or near-saturated pixels, above `saturate_factor * header["SATURATE"]`, will also be masked
        regardless of `generate_mask`.

        :param centre: the centre of the mask region, as a SkyCoord.
        :param frame: the size of the region to fit, from `centre` to the edge, in pixels or units of on-sky angle.
            Currently only a square region is supported.
        :param ext: FITS extension of image to model.
        :param model_type: an `astropy` model to use for the background.
        :param fitter_type: an `astropy` Fitter to use.
        :param init_params: the initial parameters to give to the model.
        :param write: path to write the modelled background to disk, as a FITS file. If None, no file is written.
        :param write_subbed: path to write the subtracted image to, as a FITS file. If None, no file is written.
        :param generate_mask: whether to generate a mask or not.
        :param mask_ellipses: a set of ellipses to mask. If a mask is generated, will be added to it.
        :param mask_kwargs: keyword arguments to pass to `generate_mask()`.
        :param saturate_factor: pixels with values greater than `saturate_factor * header["SATURATE"] will be masked.
        :return: Tuple of:
            `model`: the fitted model.
            `model_eval`: the pixel values of the model.
            `data`: the original data.
            `subbed_data`: the full image with the model subtracted from the window.
            `mask`: the final mask used for fitting.
            `weights`: the weights used for fitting, the inverse of the image error.
        """
        margins = left, right, bottom, top = self.frame_from_coord(
            frame=frame,
            centre=centre,
            ext=ext
        )

        print("")
        print(self.filename)

        data = self.data[ext] * 1  # [bottom:top, left:right]

        if generate_mask:
            mask = self.generate_mask(
                margins=margins,
                method="sep",
                **mask_kwargs,
            )
            mask = mask  # [bottom:top, left:right]
            mask = mask.astype(bool)
        else:
            mask = np.zeros(data.shape, dtype=bool)

        mask += data < 0
        mask += data > self.extract_saturate() * saturate_factor

        if mask_ellipses:
            for ellipse_dict in mask_ellipses:
                j, i = np.mgrid[:data.shape[0], :data.shape[1]]
                x, y = self.world_to_pixel(ellipse_dict["centre"])
                a = self.pixel(ellipse_dict["a"]).value * 2
                b = self.pixel(ellipse_dict["b"]).value * 2
                cos_angle = np.cos(180. * units.deg - ellipse_dict["theta"])
                sin_angle = np.sin(180. * units.deg - ellipse_dict["theta"])

                xc = i - x
                yc = j - y

                xct = xc * cos_angle - yc * sin_angle
                yct = xc * sin_angle + yc * cos_angle

                rad_cc = (xct ** 2 / (a / 2.) ** 2) + (yct ** 2 / (b / 2.) ** 2)
                mask += rad_cc <= 1

        median = np.median(self.data[ext])
        data -= median

        mask[:, :left] = True
        mask[:, right:] = True
        mask[:bottom, :] = True
        mask[top:, :] = True

        weights = 1. / self.sep_background[ext].rms()
        where_mask = np.where(mask)

        for n, i in enumerate(where_mask[0]):
            j = where_mask[1][n]
            weights[i, j] = 0.  # np.invert(mask).astype(float)

        model_init = model_type(**init_params)
        fitter = fitter_type(True)
        y, x = np.mgrid[:data.shape[0], :data.shape[1]]
        model = fitter(
            model_init,
            x, y,
            data.value,
            weights=weights
        )
        model_eval = model(x, y)
        subbed_all = data.value - model_eval
        subbed = data.copy() + median
        subbed[bottom:top, left:right] = subbed_all[bottom:top, left:right] * data.unit + median

        if isinstance(write, str):
            back_file = self.copy(write)
            back_file.load_data()
            back_file.load_headers()
            back_file.data[ext] = model_eval * data.unit
            back_file.add_log(
                action=f"Background modelled.",
                method=self.model_background_photometry,
                input_path=self.path,
                output_path=write,
                ext=ext,
            )
            back_file.write_fits_file()

            weights_file = self.copy(write.replace(".fits", "_weights.fits"))
            weights_file.load_data()
            weights_file.load_headers()
            weights_file.data[ext] = weights * units.ct
            weights_file.add_log(
                action=f"Background modelled.",
                method=self.model_background_photometry,
                input_path=self.path,
                output_path=write,
                ext=ext,
            )
            weights_file.write_fits_file()

            mask_file = self.copy(write.replace(".fits", "_mask.fits"))
            mask_file.load_data()
            mask_file.load_headers()
            mask_file.data[ext] = mask.astype(float) * units.ct
            mask_file.add_log(
                action=f"Background modelled.",
                method=self.model_background_photometry,
                input_path=self.path,
                output_path=write,
                ext=ext,
            )
            mask_file.write_fits_file()

        if isinstance(write_subbed, str):
            subbed_file = self.copy(write_subbed)
            subbed_file.load_data()
            subbed_file.load_headers()
            subbed_file.data[ext] = subbed
            subbed_file.add_log(
                action=f"Background modelled and subtracted.",
                method=self.model_background_photometry,
                input_path=self.path,
                output_path=write_subbed,
                ext=ext,
            )
            subbed_file.write_fits_file()

        return model, model_eval, data, subbed, mask, weights

    def model_background_photometry(
            self, ext: int = 0,
            box_size: int = 64,
            filter_size: int = 3,
            method: str = "sep",
            write: str = None,
            write_subbed: str = None,
            do_mask: bool = False,
            **back_kwargs
    ):
        self.load_data()

        mask = None
        if do_mask:
            mask = self.generate_mask(method=method)
            mask = mask.astype(bool)

        if method == "sep":
            data = u.sanitise_endianness(self.data[ext])
            bkg = self.sep_background[ext] = sep.Background(
                data,
                bw=box_size, bh=box_size,
                fw=filter_size, fh=filter_size,
                mask=mask,
                **back_kwargs
            )
            if isinstance(data, units.Quantity):
                bkg_data = bkg.back() * data.unit
            else:
                bkg_data = bkg.back()
            self.data_sub_bkg[ext] = (data - bkg_data)

        elif method == "photutils":
            data = self.data[ext]
            sigma_clip = SigmaClip(sigma=3.)
            bkg_estimator = photutils.MedianBackground()
            bkg = self.pu_background[ext] = photutils.Background2D(
                data, box_size,
                filter_size=filter_size,
                sigma_clip=sigma_clip,
                bkg_estimator=bkg_estimator,
                mask=mask,
                **back_kwargs
            )
            bkg_data = bkg.background
            self.data_sub_bkg[ext] = (data - bkg_data)

        else:
            raise ValueError(f"Unrecognised method {method}.")

        if isinstance(write, str):
            back_file = self.copy(write)
            back_file.load_data()
            back_file.load_headers()
            back_file.data[ext] = bkg_data
            back_file.write_fits_file()

        if isinstance(write_subbed, str):
            subbed_file = self.copy(write_subbed)
            subbed_file.load_data()
            subbed_file.load_headers()
            subbed_file.data[ext] = self.data[ext] - bkg_data
            subbed_file.write_fits_file()

        return bkg, bkg_data

    def generate_segmap(
            self,
            ext: int = 0,
            threshold: float = 4,
            method="sep",
            margins: tuple = (None, None, None, None),
            min_area: int = 5,
            **background_kwargs
    ):
        """
        Generate a segmentation map of the image in which the image is broken into segments according to detected sources.
        Each source is assigned an integer, and the segmap has the same spatial dimensions as the input image.
        :param ext:
        :param threshold:
        :param method:
        :param margins:
        :return:
        """
        self.load_data()
        data = self.data[ext]
        left, right, bottom, top = u.check_margins(data=data, margins=margins)

        bkg, bkg_data = self.model_background_photometry(method=method, ext=ext, **background_kwargs)

        if method == "photutils":
            data_trim = u.trim_image(
                data=data,
                margins=margins
            )
            u.debug_print(2, f"{self}.generate_segmap(): data_trim.shape ==", data_trim.shape)
            threshold = photutils.segmentation.detect_threshold(
                data_trim,
                threshold,
                background=u.trim_image(bkg.background, margins=margins),
                error=u.trim_image(bkg.background_rms, margins=margins)
            )
            u.debug_print(2, f"{self}.generate_segmap(): threshold ==", threshold)
            segmap = photutils.detect_sources(data_trim, threshold, npixels=min_area)

        elif method == "sep":
            # The copying is done here to avoid 'C-contiguous' errors in SEP.
            data_trim = u.sanitise_endianness(
                u.trim_image(self.data_sub_bkg[ext], margins=margins)
            ).copy()
            err = u.trim_image(bkg.rms(), margins=margins).copy()
            u.debug_print(2, f"{self}.generate_segmap(): type(err) ==", type(err), "err.shape ==", err.shape)
            if 0 in data_trim.shape:
                # If we've trimmed the array down to nothing, we should just return something empty and avoid sep errors
                segmap = np.zeros(data_trim.shape, dtype=int)
            else:
                objs, segmap = sep.extract(
                    data_trim,
                    err=err,
                    thresh=threshold,
                    # deblend_cont=True,
                    clean=False,
                    segmentation_map=True,
                    minarea=min_area
                )

        else:
            raise ValueError(f"Unrecognised method {method}.")
        segmap_full = np.zeros(data.shape)
        u.debug_print(2, f"{self}.generate_segmap(): segmap_full ==", segmap_full)
        u.debug_print(2, f"{self}.generate_segmap(): segmap ==", segmap)
        segmap_full[bottom:top + 1, left:right + 1] = segmap.data
        return segmap_full

    def generate_mask(
            self,
            do_not_mask: SkyCoord = (),
            ext: int = 0,
            threshold: float = 4,
            method: str = "sep",
            obj_value=1,
            back_value=0,
            margins: tuple = (None, None, None, None),
    ):
        """
        Uses a segmentation map to produce a mask covering field objects.

        :param do_not_mask: SkyCoord list of objects to keep unmasked; if any
        :param ext:
        :param threshold:
        :param method:
        :param obj_value: The value to set object pixels to. For GALFIT masks, should be 1.
        :param back_value: The value to set non-object pixels to. For GALFIT masks, should be 0.
        :param margins: If only part of the image is to be masked, provide (left, right, bottom, top) in pixel
            coordinates as tuple.
        :return:
        """
        data = self.load_data()[ext]
        segmap = self.generate_segmap(
            ext=ext,
            threshold=threshold,
            method=method,
            margins=margins
        )
        self.load_wcs()

        do_not_mask = u.check_iterable(do_not_mask)

        if segmap is None:
            mask = np.zeros(data.shape)
        else:
            # Loop over the given coordinates and eliminate those segments from the mask.
            mask = np.ones(data.shape, dtype=bool)
            # This sets all the background pixels to False
            mask[segmap == 0] = False
            for coord in do_not_mask:
                if self.wcs[ext].footprint_contains(coord):
                    x_unmasked, y_unmasked = self.world_to_pixel(coord=coord, ext=ext)
                    x_unmasked = int(np.round(x_unmasked))
                    y_unmasked = int(np.round(y_unmasked))
                    # obj_id is the integer representing that object in the segmap
                    obj_id = segmap[y_unmasked, x_unmasked]
                    # If obj_id is zero, then our work here is already done (ie, the segmap routine read it as background anyway)
                    if obj_id != 0:
                        mask[segmap == obj_id] = False

        # Convert to integer (from bool)
        mask = mask.astype(int)
        mask[mask > 0] = obj_value
        mask[mask == 0] = back_value

        return mask

    def masked_data(
            self,
            mask: np.ndarray = None,
            ext: int = 0,
            **generate_mask_kwargs
    ):
        self.load_data()
        if mask is None:
            mask = self.generate_mask(**generate_mask_kwargs)

        # if mask_type == 'zeroed-out'

        return np.ma.MaskedArray(self.data[ext].copy(), mask=mask)

    def write_mask(
            self,
            output_path: str,
            ext: int = 0,
            **mask_kwargs
    ) -> 'ImagingImage':
        """
        Generates and writes a source mask to a FITS file.
        Any argument accepted by generate_mask() can be passed as a keyword.

        :param output_path: path to write the mask file to.
        :param ext: FITS extension to modify.
        :return:
        """

        mask_file = self.copy(output_path)
        mask_file.load_data()
        mask_file.data[ext] = self.generate_mask(ext=ext, **mask_kwargs) * units.dimensionless_unscaled
        mask_file.write_fits_file()

        mask_file.add_log(
            action="Converted image to source mask.",
            method=self.source_extraction,
            output_path=output_path,
        )
        mask_file.update_output_file()
        return mask_file

    def mask_nearby(self):
        return True

    def detection_threshold(self):
        return 5.

    def do_subtract_background(self):
        return True

    def sep_aperture_photometry(
            self,
            x: float, y: float,
            aperture_radius: units.Quantity = 2.0 * units.arcsec,
            ext: int = 0,
            sub_background: bool = True
    ):
        self.extract_pixel_scale()
        pixel_radius = aperture_radius.to(units.pix, self.pixel_scale_y)
        self.model_background_photometry(ext=ext, do_mask=True)
        if sub_background:
            data = self.data_sub_bkg[ext]
        else:
            data = u.sanitise_endianness(self.data[ext])
        flux, fluxerr, flag = sep.sum_circle(
            data,
            x, y,
            pixel_radius.value,
            err=self.sep_background[ext].rms(),
            gain=self.extract_gain().value
        )
        return flux, fluxerr, flag

    def sep_elliptical_photometry(
            self,
            centre: SkyCoord,
            a_world: units.Quantity,
            b_world: units.Quantity,
            theta_world: units.Quantity,
            kron_radius: float = 1.,
            ext: int = 0,
            output: str = None,
            mask_nearby=True,
            subtract_background: bool = True,
    ):

        if isinstance(output, str):
            back_output = output + "_back.fits"
            segmap_output = output + "_segmap.fits"
        else:
            back_output = None
            segmap_output = None

        self.model_background_photometry(ext=ext, write=back_output, do_mask=True)
        self.load_wcs()
        self.extract_pixel_scale()
        if not self.wcs[ext].footprint_contains(centre):
            return None, None, None, None
        x, y = self.wcs[ext].all_world2pix(centre.ra.value, centre.dec.value, 0)
        x = u.check_iterable(x)
        y = u.check_iterable(y)
        a = u.check_iterable((a_world.to(units.pix, self.pixel_scale_y)).value)
        b = u.check_iterable((b_world.to(units.pix, self.pixel_scale_y)).value)
        kron_radius = u.check_iterable(kron_radius)
        rotation_angle = self.extract_rotation_angle(ext=ext)
        theta_deg = -theta_world + rotation_angle  # + 90 * units.deg
        theta = u.theta_range(theta_deg.to(units.rad)).value

        u.debug_print(2, f"sep_elliptical_photometry: mask_nearby == {mask_nearby}")

        if isinstance(mask_nearby, ImagingImage):
            mask = mask_nearby.data[0].value
        elif mask_nearby:
            mask = self.write_mask(
                do_not_mask=centre,
                ext=ext,
                method="sep",
                output_path=segmap_output
            ).data[0].value
        else:
            mask = np.zeros_like(self.data[ext].data)

        if subtract_background:
            data = self.data_sub_bkg[ext]
            back, _, _ = sep.sum_ellipse(
                data=self.sep_background[ext].back(),
                x=x, y=y,
                a=a, b=b,
                r=kron_radius,
                theta=theta,
            )
        else:
            data = u.sanitise_endianness(self.data[ext])
            back = [0.]

        flux, flux_err, flag = sep.sum_ellipse(
            data=data,
            x=x, y=y,
            a=a, b=b,
            r=kron_radius,
            theta=theta,
            err=self.sep_background[ext].rms(),
            gain=self.extract_gain().value,
            mask=mask.astype(bool),
        )

        if isinstance(output, str):
            # objects = sep.extract(self.data_sub_bkg[ext], 1.5, err=self.sep_background[ext].rms())
            this_frame = self.nice_frame({
                'A_WORLD': a_world,
                'B_WORLD': b_world,
                'KRON_RADIUS': kron_radius
            })

            plt.close()
            with quantity_support():

                theta_plot = (theta[0] * units.rad).to(units.deg).value

                e_kron = Ellipse(
                    xy=(x[0], y[0]),
                    width=2 * kron_radius[0] * a[0],
                    height=2 * kron_radius[0] * b[0],
                    angle=theta_plot
                )
                e_kron.set_facecolor('none')
                e_kron.set_edgecolor('white')

                e = Ellipse(
                    xy=(x[0], y[0]),
                    width=2 * a[0],
                    height=2 * b[0],
                    angle=theta_plot
                )
                e.set_facecolor('none')
                e.set_edgecolor('white')

                # for i in range(len(objects)):
                #     e = Ellipse(
                #         xy=(objects["x"][i], objects["y"][i]),
                #         width=4*objects["a"][i],
                #         height=4*objects["b"][i],
                #         angle=objects["theta"][i] * 180. / np.pi)
                #     e.set_facecolor('none')
                #     e.set_edgecolor('red')
                #     ax.add_artist(e)
                #     ax.text(objects["x"][i], objects["y"][i], objects["theta"][i] * 180. / np.pi)

                ax, fig, _ = self.plot_subimage(
                    centre=centre,
                    frame=this_frame,
                    ext=ext,
                    mask=mask
                )

                e_next = copy.deepcopy(e)
                e_kron_next = copy.deepcopy(e_kron)

                ax.add_artist(e)
                ax.add_artist(e_kron)

                ax.set_title(f"{a[0], b[0], kron_radius[0], theta_plot}")

                fig.savefig(output + ".png")

                if subtract_background:
                    ax, fig, _ = self.plot_subimage(
                        centre=centre,
                        frame=this_frame,
                        ext=ext,
                        mask=mask,
                        data=data
                    )

                    ax.add_artist(e_next)
                    ax.add_artist(e_kron_next)

                    ax.set_title(f"{a[0], b[0], kron_radius[0], theta_plot}")

                    fig.savefig(output + "_back_sub.png")

        return flux, flux_err, flag, back

    def sep_elliptical_magnitude(
            self,
            centre: SkyCoord,
            a_world: units.Quantity,
            b_world: units.Quantity,
            theta_world: units.Quantity,
            kron_radius: float = 1.,
            ext: int = 0,
            output: str = None,
            mask_nearby=True,
            detection_threshold: float = None,
            **kwargs
    ):
        """

        :param centre:
        :param a_world:
        :param b_world:
        :param theta_world:
        :param kron_radius:
        :param ext:
        :param output:
        :param mask_nearby:
        :param detection_threshold:
        :param kwargs: keyword arguments to pass to the magnitude() method; header exp_time etc can be overridden here.
        :return:
        """

        if a_world < 0. or b_world < 0.:
            return None

        if detection_threshold is None:
            detection_threshold = self.detection_threshold()

        u.debug_print(2, f"sep_elliptical_magnitude(): mask_nearby == {mask_nearby}")

        flux, flux_err, flags, back = self.sep_elliptical_photometry(
            centre=centre,
            a_world=a_world,
            b_world=b_world,
            theta_world=theta_world,
            kron_radius=kron_radius,
            ext=ext,
            output=output,
            mask_nearby=mask_nearby,
            subtract_background=self.do_subtract_background()
        )

        if flux is None:
            return None

        snr = flux / flux_err
        mag, mag_err, _, _ = self.magnitude(
            flux, flux_err,
            **kwargs
        )
        for i, m in enumerate(mag):
            if snr[i] < detection_threshold or np.isnan(m):
                mag_lim, _, _, _ = self.magnitude(
                    detection_threshold * flux_err[i]
                )

                if m > mag_lim or np.isnan(m):
                    m = mag_lim
                if np.isnan(m):
                    m = -999. * units.mag
                mag_err[i] = -999. * units.mag
                mag[i] = m

        return {
            "mag": mag,
            "mag_err": mag_err,
            "snr": snr,
            "back": back,
            "flux": flux,
            "flux_err": flux_err,
            "threshold": detection_threshold
        }

    def make_galfit_version(
            self,
            output_path: str = None,
            ext: int = 0
    ):
        """
        Generate a version of this file for use with GALFIT.
        Modifies header item GAIN to conform to GALFIT's expectations (outlined in the GALFIT User Manual,
        http://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html)
        :param output_path: path to write modified file to.
        :param ext: FITS extension to modify header of.
        :return:
        """
        if output_path is None:
            output_path = self.path.replace(".fits", "_galfit.fits")
        new = self.copy(output_path)
        new.load_headers()
        new.set_header_items(
            {
                "GAIN": self.extract_header_item(key="OLD_EXPTIME", ext=ext) *
                        self.extract_header_item(key="OLD_GAIN", ext=ext)
            }
        )
        new.write_fits_file()
        return new

    def make_galfit_psf(
            self,
            output_dir: str,
            x: float,
            y: float
    ):
        # We obtain an oversampled PSF, because GALFIT works best with one.
        psfex_path = os.path.join(output_dir, f"{self.name}_galfit_psfex.psf")
        if not os.path.isfile(psfex_path):
            self.psfex(
                output_dir=output_dir,
                PSF_SAMPLING=0.5,  # Equivalent to GALFIT fine-sampling factor = 2
                # PSF_SIZE=50,
                force=True,
                set_attributes=True
            )
        else:
            self.psfex_path = psfex_path
            self.load_psfex_output()
        # Load oversampled PSF image
        psf_img = self.psf_image(x=x, y=y, match_pixel_scale=False)[0]
        psf_img /= np.max(psf_img)
        # Write our PSF image to disk for GALFIT to find
        psf_hdu = fits.hdu.PrimaryHDU(psf_img)
        psf_hdu_list = fits.hdu.HDUList(psf_hdu)
        psf_path = os.path.join(output_dir, f"{self.name}_psf.fits")
        psf_hdu_list.writeto(
            psf_path,
            overwrite=True
        )
        return psf_path

    def make_galfit_feedme(
            self,
            feedme_path: str,
            img_block_path: str,
            psf_file: str = None,
            psf_fine_sampling: int = 2,
            mask_file: str = None,
            fitting_region_margins: tuple = None,
            convolution_size: tuple = None,
            models: List[dict] = None
    ):
        if fitting_region_margins is None:
            self.load_data()
            max_x, max_y = self.data[0].shape
            fitting_region_margins = 0, max_x - 1, 0, max_y - 1
        if convolution_size is None:
            left, right, bottom, top = fitting_region_margins
            convolution_size = int(right - left), int(top - bottom)

        self.extract_pixel_scale()
        dx = (1 * units.pixel).to(units.arcsec, self.pixel_scale_x).value
        dy = (1 * units.pixel).to(units.arcsec, self.pixel_scale_y).value

        galfit.galfit_feedme(
            feedme_path=feedme_path,
            input_file=self.filename,
            output_file=img_block_path,
            zeropoint=self.zeropoint_best["zeropoint_img"].value,
            psf_file=psf_file,
            psf_fine_sampling=psf_fine_sampling,
            mask_file=mask_file,
            fitting_region_margins=fitting_region_margins,
            convolution_size=convolution_size,
            plate_scale=(dx, dy),
            models=models
        )

    def galfit(
            self,
            output_dir: str = None,
            output_prefix=None,
            frame_lower: int = 30,
            frame_upper: int = 100,
            ext: int = 0,
            model_guesses: Union[dict, List[dict]] = None,
            psf_path: str = None,
            use_frb_galfit: bool = False
    ):
        """

        :param coords:
        :param output_dir:
        :param frame_lower:
        :param frame_upper:
        :param ext:
        :param model_guesses: dict, with:
            object_type: str
            position: Either "position" can be provided as a SkyCoord object, or x & y as pixel coordinates.

        :param use_frb_galfit: Use the FRB repo frb.galaxies.galfit module. Single-sersic only; if multiple models are provided only one will be used.
        :return:
        """
        if output_prefix is None:
            output_prefix = self.name
        if model_guesses is None:
            model_guesses = [{
                "object_type": "sersic",
                "int_mag": 20.0,
                "position": self.epoch.field.objects[0].position
            }]

        if isinstance(model_guesses, dict):
            model_guesses = [model_guesses]
        gf_tbls = {}
        for i, model in enumerate(model_guesses):
            if "position" in model:
                x, y = self.world_to_pixel(
                    coord=model["position"],
                    origin=1
                )
                model_guesses[i]["x"] = x
                model_guesses[i]["y"] = y
            elif "x" in model and "y" in model:
                model_guesses["position"] = self.pixel_to_world(
                    x=model["x"],
                    y=model["y"],
                    origin=1
                )
            else:
                raise ValueError("All model dicts must have either 'position' or 'x' & 'y' keys.")
            gf_tbls[f"COMP_{i + 1}"] = []
        gf_tbls[f"COMP_{i + 2}"] = []

        if output_dir is None:
            output_dir = self.data_path
        self.load_output_file()
        new = self.make_galfit_version(
            output_path=os.path.join(output_dir, f"{output_prefix}_galfit.fits")
        )
        new.zeropoint_best = self.zeropoint_best
        new.open()

        x = model_guesses[0]["x"]
        y = model_guesses[0]["y"]
        if psf_path is None:
            psf_path = new.make_galfit_psf(
                x=x,
                y=y,
                output_dir=output_dir
            )
        # Turn the first model into something the frb repo can use, and hope it's a sersic
        if use_frb_galfit:
            model_dict = model_guesses[0].copy()
            x = int(model_dict.pop("x"))
            y = int(model_dict.pop("y"))
            model_dict["position"] = (x, y)
            model_dict.pop("object_type")

        psf_file = os.path.split(psf_path)[-1]
        psf_path_moved = os.path.join(output_dir, psf_file)
        if not os.path.isfile(psf_path_moved):
            shutil.copy(psf_path, psf_path_moved)
        psf_path = psf_path_moved

        new.load_data()
        data = new.data[ext].copy()
        new.close()

        mask_file = f"{output_prefix}_mask.fits"
        mask_path = os.path.join(output_dir, mask_file)
        margins_max = u.frame_from_centre(frame_upper + 1, x, y, data)
        mask = new.write_mask(
            output_path=mask_path,
            do_not_mask=list(map(lambda m: m["position"], model_guesses)),
            ext=ext,
            method="sep",
            obj_value=1,
            back_value=0,
            margins=margins_max
        )

        self.extract_pixel_scale(ext)

        for frame in range(frame_lower, frame_upper + 1):
            margins = u.frame_from_centre(frame, x, y, data)
            print("Generating mask...")
            data_trim = u.trim_image(data, margins=margins)
            mask_data = u.trim_image(mask.data[ext], margins=margins).value
            feedme_file = f"{output_prefix}_{frame}.feedme"
            feedme_path = os.path.join(output_dir, feedme_file)
            img_block_file = f"{output_prefix}_galfit-out_{frame}.fits"
            img_block_path = os.path.join(output_dir, img_block_file)
            if not use_frb_galfit:
                new.make_galfit_feedme(
                    feedme_path=feedme_path,
                    img_block_path=img_block_file,
                    psf_file=psf_file,
                    psf_fine_sampling=2,
                    mask_file=mask_file,
                    fitting_region_margins=margins,
                    convolution_size=(frame * 2, frame * 2),
                    models=model_guesses
                )
                galfit.galfit(
                    config=feedme_file,
                    output_dir=output_dir
                )
            else:
                import frb.galaxies.galfit as galfit_frb
                galfit_frb.run(
                    imgfile=new.path,
                    psffile=psf_path,
                    outdir=output_dir,
                    configfile=feedme_file,
                    outfile=img_block_path,
                    finesample=2,
                    badpix=mask_path,
                    region=margins,
                    convobox=(frame * 2, frame * 2),
                    zeropoint=self.zeropoint_best["zeropoint_img"].value,
                    skip_sky=False,
                    **model_dict
                )
            shutil.copy(os.path.join(output_dir, "fit.log"),
                        os.path.join(output_dir, f"{output_prefix}_{frame}_fit.log"))

            try:
                img_block = fits.open(img_block_path)
            except FileNotFoundError:
                return None

            results_header = img_block[2].header
            components = galfit.extract_fit_params(results_header)
            for compname in components:
                component = components[compname]
                pos = self.pixel_to_world(component["x"], component["y"])
                component["ra"] = pos.ra
                component["dec"] = pos.dec
                if "r_eff" in component:
                    component["r_eff_ang"] = component["r_eff"].to(units.arcsec, self.pixel_scale_x)
                    component["r_eff_ang_err"] = component["r_eff_err"].to(units.arcsec, self.pixel_scale_x)
                # TODO: The below assumes RA and Dec are along x & y (neglecting image rotation), which isn't great
                component["ra_err"] = component["x_err"].to(units.deg, self.pixel_scale_x)
                component["dec_err"] = component["y_err"].to(units.deg, self.pixel_scale_y)
                component["frame"] = frame
                results_table = table.QTable([component])
                gf_tbls[compname].append(results_table)

            mask_ones = np.invert(mask_data.astype(bool)).astype(int)

            # Masked data
            img_block.insert(4, img_block[1].copy())
            img_block[4].data *= mask_ones  # + #
            img_block[4].data += mask_data * np.median(img_block[1].data)

            # Masked, subtracted data
            img_block.insert(5, img_block[3].copy())
            img_block[5].data *= mask_ones  # + #
            img_block[5].data += mask_data * np.median(img_block[3].data)

            for idx in [2, 3]:
                img_block[idx].header.insert('OBJECT', ('PCOUNT', 0))
                img_block[idx].header.insert('OBJECT', ('GCOUNT', 1))

            img_block.writeto(img_block_path, overwrite=True)

        component_tables = {}
        for compname in gf_tbls:
            gf_tbl = table.vstack(gf_tbls[compname])
            component_tables[compname] = gf_tbl

        shutil.copy(p.path_to_config_galfit(), output_dir)

        return component_tables

    def galfit_object(
            self,
            obj: objects.Galaxy,
            pivot_component: int = 2,
            **kwargs
    ):

        photometry, _ = obj.select_photometry(
            fil=self.filter_name,
            instrument=self.instrument_name,
        )

        if "model_guesses" in kwargs:
            model_guesses = kwargs["model_guesses"]
        else:
            model_guesses = [{
                "object_type": "sersic"
            }]

        for model in model_guesses:
            model["position"] = obj.position
            model["int_mag"] = photometry["mag"].value

        model_tbls = self.galfit(
            model_guesses=model_guesses,
            **kwargs
        )

        best_params = galfit.sersic_best_row(model_tbls[f"COMP_{pivot_component}"])
        best_params["r_eff_proj"] = obj.projected_size(best_params["r_eff_ang"]).to("kpc")
        best_params["r_eff_proj_err"] = obj.projected_size(best_params["r_eff_ang_err"]).to("kpc")
        return best_params

    @classmethod
    def select_child_class(cls, instrument_name: str, **kwargs):
        if not isinstance(instrument_name, str):
            instrument_name = str(instrument_name)
        instrument_name = instrument_name.lower()
        if instrument_name in cls.class_dict:
            subclass = cls.class_dict[instrument_name]
        else:
            raise ValueError(f"Unrecognised instrument {instrument_name}")
        return subclass

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "filter": "FILTER",
            "ra": "CRVAL1",
            "dec": "CRVAL2",
            "ref_pix_x": "CRPIX1",
            "ref_pix_y": "CRPIX2",
            "ra_old": "_RVAL1",
            "dec_old": "_RVAL2",
            "airmass": "AIRMASS",
            "airmass_err": "AIRMASS_ERR",
            "astrometry_err": "ASTM_RMS",
            "ra_err": "RA_RMS",
            "dec_err": "DEC_RMS",
            "psf_fwhm": "PSF_FWHM"
        })
        return header_keys

    @classmethod
    def count_exposures(cls, image_paths: list):
        return len(image_paths)

    def rank_photometric_cat(self, cats: list):
        """
        Gives the ranking of photometric catalogues available for calibration, ranked by similarity to filter set.
        :return:
        """

        self.instrument.gather_filters()
        self._filter_from_name()

        differences = {}

        for cat in cats:
            if cat in cat_instruments:
                other_instrument_name = cat_instruments[cat]
                other_instrument = inst.Instrument.from_params(other_instrument_name)
                other_instrument.gather_filters()
                if self.filter.band_name in other_instrument.bands:
                    other_filter = other_instrument.bands[self.filter.band_name]
                    differences[cat] = self.filter.compare_wavelength_range(
                        other=other_filter
                    )
            elif cat == "instrument_archive":
                differences[cat] = 0 * units.angstrom
            elif cat == "calib_pipeline":
                differences[cat] = 0.1 * units.angstrom

        print(differences)
        differences = dict(sorted(differences.items(), key=lambda x: x[1]))
        return list(differences.keys()), list(differences.values())


def deepest(
        img_1: ImagingImage,
        img_2: ImagingImage,
        sigma: int = 3,
        depth_type: str = "secure",
        snr_type: str = "SNR_PSF"
):
    if img_1.depth[depth_type][snr_type][f"{sigma}-sigma"] > \
            img_2.depth[depth_type][snr_type][f"{sigma}-sigma"]:
        return img_1
    else:
        return img_2


def _set_class_dict():
    from .__init__ import (
        DESCutout,
        GSAOIImage,
        HubbleImage,
        FORS2Image,
        PanSTARRS1Cutout,
        HAWKIImage,
        WISECutout
    )

    ImagingImage.class_dict = {
        "none": ImagingImage,
        "decam": DESCutout,
        "gs-aoi": GSAOIImage,
        "hst-wfc3_uvis2": HubbleImage,
        "hst-wfc3_ir": HubbleImage,
        "panstarrs1": PanSTARRS1Cutout,
        "vlt-fors2": FORS2Image,
        "vlt-hawki": HAWKIImage,
        "wise": WISECutout
    }
