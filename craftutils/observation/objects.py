from typing import Union, List
import os
import copy

import matplotlib.pyplot as plt
import numpy as np

from astropy.coordinates import SkyCoord, Longitude
import astropy.units as units
import astropy.table as table
import astropy.cosmology as cosmo
from astropy.modeling import models, fitting
from astropy.visualization import quantity_support
import astropy.time as time
import astropy.io.fits as fits

import craftutils.params as p
import craftutils.astrometry as astm
import craftutils.utils as u
import craftutils.retrieve as r
import craftutils.observation as obs
import craftutils.observation.instrument as inst
import craftutils.observation.filters as filters
from craftutils.photometry import distance_modulus

cosmology = cosmo.Planck18

quantity_support()

position_dictionary = {
    "alpha": {
        "decimal": 0.0,
        "hms": None},
    "delta": {
        "decimal": 0.0,
        "dms": None
    },
}

uncertainty_dict = {
    "sys": 0.0,
    "stat": 0.0
}

__all__ = []


@u.export
def skycoord_to_position_dict(skycoord: SkyCoord):
    ra_float = skycoord.ra.value
    dec_float = skycoord.dec.value

    s = skycoord.to_string("hmsdms")
    ra = s[:s.find(" ")]
    dec = s[s.find(" ") + 1:]

    position = {
        "alpha": {"decimal": ra_float, "hms": ra},
        "delta": {"decimal": dec_float, "dms": dec},
    }

    return position


@u.export
class PositionUncertainty:
    def __init__(
            self,
            uncertainty: Union[float, units.Quantity, dict, tuple] = None,
            position: SkyCoord = None,
            ra_err_sys: Union[float, units.Quantity] = None,
            ra_err_stat: Union[float, units.Quantity] = None,
            dec_err_sys: Union[float, units.Quantity] = None,
            dec_err_stat: Union[float, units.Quantity] = None,
            a_stat: Union[float, units.Quantity] = None,
            a_sys: Union[float, units.Quantity] = None,
            b_stat: Union[float, units.Quantity] = None,
            b_sys: Union[float, units.Quantity] = None,
            theta: Union[float, units.Quantity] = None,
            sigma: float = None,
            **kwargs
    ):
        """
        If a single value is provided for uncertainty, the uncertainty ellipse will be assumed to be circular.
        Values in dictionary, if provided, override values given as arguments.
        Position values provided without units are assumed to be in degrees.
        On the other hand, uncertainty values provided without units are assumed to be in arcseconds;
        except for uncertainties in RA, which are assumed in RA seconds.
        :param uncertainty:
        :param position:
        :param ra_err_sys:
        :param ra_err_stat:
        :param dec_err_sys:
        :param dec_err_stat:
        :param a_stat:
        :param a_sys:
        :param b_stat:
        :param b_sys:
        :param theta:
        :param sigma: The confidence interval (expressed in multiples of sigma) of the uncertainty ellipse.
        """
        self.sigma = sigma
        # Assign values from dictionary, if provided.
        if type(uncertainty) is dict:
            ra_key = None
            if "ra" in uncertainty and uncertainty["ra"] is not None:
                ra_key = "ra"
            elif "alpha" in uncertainty and uncertainty["alpha"] is not None:
                ra_key = "alpha"

            if ra_key is not None:
                if "sys" in uncertainty[ra_key] and uncertainty[ra_key]["sys"] is not None:
                    ra_err_sys = uncertainty[ra_key]["sys"]
                    if isinstance(ra_err_sys, str):
                        ra_err_sys = (Longitude(ra_err_sys) * np.cos(position.dec)).to("arcsec")
                if "stat" in uncertainty[ra_key] and uncertainty[ra_key]["stat"] is not None:
                    ra_err_stat = uncertainty[ra_key]["stat"]
                    if isinstance(ra_err_stat, str):
                        ra_err_stat = (Longitude(ra_err_stat) * np.cos(position.dec)).to("arcsec")

            dec_key = None
            if "dec" in uncertainty and uncertainty["dec"] is not None:
                dec_key = "dec"
            if "delta" in uncertainty and uncertainty["delta"] is not None:
                dec_key = "delta"

            if dec_key is not None:
                if "sys" in uncertainty[dec_key] and uncertainty[dec_key]["sys"] is not None:
                    dec_err_sys = uncertainty[dec_key]["sys"]
                if "stat" in uncertainty[dec_key] and uncertainty[dec_key]["stat"] is not None:
                    dec_err_stat = uncertainty[dec_key]["stat"]

            if "a" in uncertainty and uncertainty["a"] is not None:
                if "sys" in uncertainty["a"] and uncertainty["a"]["sys"] is not None:
                    a_sys = uncertainty["a"]["sys"]
                if "stat" in uncertainty["a"] and uncertainty["a"]["stat"] is not None:
                    a_stat = uncertainty["a"]["stat"]
            if "b" in uncertainty and uncertainty["b"] is not None:
                if "sys" in uncertainty["b"] and uncertainty["b"]["sys"] is not None:
                    b_sys = uncertainty["b"]["sys"]
                if "stat" in uncertainty["b"] and uncertainty["a"]["stat"] is not None:
                    b_stat = uncertainty["b"]["stat"]
            if "theta" in uncertainty and uncertainty["theta"] is not None:
                theta = uncertainty["theta"]
        # If uncertainty is a single value, assume a circular uncertainty region without distinction between systematic
        # and statistical.
        elif uncertainty is not None:
            a_stat = uncertainty
            a_sys = 0.0 * units.arcsec
            b_stat = uncertainty
            b_sys = 0.0 * units.arcsec
            theta = 0.0 * units.deg

        if ra_err_stat is None and "alpha_err_stat" in kwargs:
            ra_err_stat = kwargs["alpha_err_stat"]
        if ra_err_sys is None and "alpha_err_sys" in kwargs:
            ra_err_sys = kwargs["alpha_err_sys"]
        if dec_err_stat is None and "delta_err_stat" in kwargs:
            dec_err_stat = kwargs["delta_err_stat"]
        if dec_err_sys is None and "delta_err_sys" in kwargs:
            dec_err_sys = kwargs["delta_err_sys"]

        # Check whether we're specifying uncertainty using equatorial coordinates or ellipse parameters.
        u.debug_print(2, "PositionUncertainty.__init__(): a_stat, a_sys, b_stat, b_sys, theta, position ==", a_stat,
                      a_sys, b_stat, b_sys, theta, position)
        u.debug_print(2, "PositionUncertainty.__init__(): ra_err_sys, ra_err_stat, dec_err_sys, dec_err_stat ==",
                      ra_err_sys, ra_err_stat, dec_err_sys, dec_err_stat)
        if a_stat is not None and a_sys is not None and b_stat is not None and b_sys is not None and theta is not None:
            ellipse = True
        elif ra_err_sys is not None and ra_err_stat is not None and dec_err_sys is not None and dec_err_stat is not None and position is not None:
            ellipse = False
        else:
            raise ValueError(
                "Either all ellipse values (a, b, theta) or all equatorial values (ra, dec, position) must be provided.")

        ra_err_sys = u.check_quantity(number=ra_err_sys, unit=units.arcsec)
        ra_err_stat = u.check_quantity(number=ra_err_stat, unit=units.arcsec)
        dec_err_sys = u.check_quantity(number=dec_err_sys, unit=units.arcsec)
        dec_err_stat = u.check_quantity(number=dec_err_stat, unit=units.arcsec)
        # Convert equatorial uncertainty to ellipse with theta=0
        if not ellipse:
            ra = position.ra
            dec = position.dec
            a_sys = ra_err_sys
            # SkyCoord(0.0 * units.degree, dec).separation(SkyCoord(ra_err_sys, dec))
            a_stat = ra_err_stat
            # SkyCoord(0.0 * units.degree, dec).separation(SkyCoord(ra_err_stat, dec))
            b_sys = dec_err_sys
            # SkyCoord(ra, dec).separation(SkyCoord(ra, dec + dec_err_sys))
            b_stat = dec_err_stat
            # SkyCoord(ra, dec).separation(SkyCoord(ra, dec + dec_err_stat))
            # a_sys, b_sys = max(a_sys, b_sys), min(a_sys, b_sys)
            # a_stat, b_stat = max(a_stat, b_stat), min(a_stat, b_stat)
            theta = 0.0 * units.degree
        # Or use ellipse parameters as given.
        else:
            a_sys = u.check_quantity(number=a_sys, unit=units.arcsec)
            a_stat = u.check_quantity(number=a_stat, unit=units.arcsec)
            b_sys = u.check_quantity(number=b_sys, unit=units.arcsec)
            b_stat = u.check_quantity(number=b_stat, unit=units.arcsec)
            theta = u.check_quantity(number=theta, unit=units.arcsec)

        self.a_sys = a_sys
        self.a_stat = a_stat
        self.b_sys = b_sys
        self.b_stat = b_stat
        self.theta = theta

        self.ra_sys = ra_err_sys
        self.dec_sys = dec_err_sys
        self.ra_stat = ra_err_stat
        self.dec_stat = dec_err_stat

    def __str__(self):
        return f"PositionUncertainty: a_stat={self.a_stat}, b_stat={self.b_stat}; a_sys={self.a_sys}, b_sys={self.b_sys}"

    def uncertainty_quadrature(self):
        return np.sqrt(self.a_sys ** 2 + self.a_stat ** 2), np.sqrt(self.b_sys ** 2 + self.b_stat ** 2)

    def uncertainty_quadrature_equ(self):
        return np.sqrt(self.ra_sys ** 2 + self.ra_stat ** 2), np.sqrt(self.dec_sys ** 2 + self.dec_stat ** 2)

    # TODO: Finish this

    def to_dict(self):
        return {
            "a_sys": self.a_sys,
            "a_stat": self.a_stat,
            "b_sys": self.b_sys,
            "b_stat": self.b_stat,
            "theta": self.theta,
            "alpha_err_sys": self.ra_sys,
            "delta_err_sys": self.dec_sys,
            "alpha_err_stat": self.ra_stat,
            "delta_err_stat": self.dec_stat
        }

    @classmethod
    def default_params(cls):
        return {
            "alpha": copy.deepcopy(uncertainty_dict),
            "delta": copy.deepcopy(uncertainty_dict),
            "a": copy.deepcopy(uncertainty_dict),
            "b": copy.deepcopy(uncertainty_dict),
            "theta": 0.0,
            "sigma": None,
            "healpix_path": None
        }


@u.export
class Object:
    def __init__(
            self,
            name: str = None,
            position: Union[SkyCoord, str] = None,
            position_err: Union[float, units.Quantity, dict, PositionUncertainty, tuple] = 0.0 * units.arcsec,
            field=None,
            row: table.Row = None,
            plotting: dict = None,
            **kwargs
    ):
        self.name = name

        self.cat_row = row
        self.position = None
        self.position_err = None

        if self.cat_row is not None:
            self.position_from_cat_row()
        elif position is not None:
            self.position = astm.attempt_skycoord(position)
            if type(position_err) is not PositionUncertainty:
                self.position_err = PositionUncertainty(uncertainty=position_err, position=self.position)
            self.position_galactic = None
            if isinstance(self.position, SkyCoord):
                self.position_galactic = self.position.transform_to("galactic")

        self.position_photometry = copy.deepcopy(self.position)
        self.position_photometry_err = copy.deepcopy(self.position_err)

        if self.name is None:
            self.jname()
        self.name_filesys = None
        self.set_name_filesys()

        self.photometry = {}
        self.photometry_tbl_best = None
        self.photometry_tbl = None
        self.data_path = None
        self.output_file = None
        self.field = field
        self.irsa_extinction_path = None
        self.irsa_extinction = None
        self.ebv_sandf = None
        self.extinction_power_law = None
        self.paths = {}
        if self.data_path is not None:
            self.load_output_file()
        if isinstance(plotting, dict):
            self.plotting_params = plotting
            if "frame" in self.plotting_params and self.plotting_params["frame"] is not None:
                self.plotting_params["frame"] = u.check_quantity(self.plotting_params["frame"], units.arcsec)
        else:
            self.plotting_params = {}

        self.a = None
        self.b = None
        self.theta = None
        self.kron = None

        self.photometry_args = None
        if "photometry_args_manual" in kwargs and kwargs["photometry_args_manual"]["a"] != 0 and \
                kwargs["photometry_args_manual"]["b"]:
            self.photometry_args = kwargs["photometry_args_manual"]
            self.a = self.photometry_args["a"]
            self.b = self.photometry_args["b"]
            self.theta = self.photometry_args["theta"]
            self.kron = self.photometry_args["kron_radius"]

    def set_name_filesys(self):
        if self.name is not None:
            self.name_filesys = self.name.replace(" ", "-")

    def position_from_cat_row(self, cat_row: table.Row = None):
        if cat_row is not None:
            self.cat_row = cat_row
        self.position_photometry = SkyCoord(self.cat_row["RA"], self.cat_row["DEC"])
        self.position_photometry_err = PositionUncertainty(
            ra_err_stat=self.cat_row["RA_ERR"],
            ra_err_sys=0.0 * units.arcsec,
            dec_err_stat=self.cat_row["DEC_ERR"],
            dec_err_sys=0.0 * units.arcsec,
            position=self.position_photometry
        )
        return self.position_photometry

    def get_photometry(self):
        for cat in self.field.cats:
            pass

    def get_good_photometry(self):

        import craftutils.observation.image as image
        self.estimate_galactic_extinction()
        deepest_dict = self.select_deepest()
        deepest_path = deepest_dict["good_image_path"]

        cls = image.CoaddedImage.select_child_class(instrument=deepest_dict["instrument"])
        deepest_img = cls(path=deepest_path)
        deep_mask = deepest_img.write_mask(
            output_path=os.path.join(
                self.data_path,
                f"{self.name_filesys}_master-mask_{deepest_dict['instrument']}_{deepest_dict['filter']}_{deepest_dict['epoch_name']}.fits",
            ),
            method="sep",
            unmasked=self.position_photometry
        )

        mag_results = deepest_img.sep_elliptical_magnitude(
            centre=self.position_photometry,
            a_world=self.a,
            b_world=self.b,
            theta_world=self.theta,
            kron_radius=self.kron,
            output=os.path.join(
                self.data_path,
                f"{self.name_filesys}_{deepest_dict['instrument']}_{deepest_dict['filter']}_{deepest_dict['epoch_name']}"
            ),
            mask_nearby=deep_mask
        )
        if mag_results is not None:
            deepest_dict["mag_sep"] = mag_results["mag"][0]
            deepest_dict["mag_sep_err"] = mag_results["mag_err"][0]
            deepest_dict["snr_sep"] = mag_results["snr"][0]
            deepest_dict["back_sep"] = mag_results["back"][0]
            deepest_dict["flux_sep"] = mag_results["flux"][0]
            deepest_dict["flux_sep_err"] = mag_results["flux_err"][0]
            deepest_dict["limit_threshold"] = mag_results["threshold"]
        else:
            deepest_dict["mag_sep"] = -999. * units.mag
            deepest_dict["mag_sep_err"] = -999. * units.mag
            deepest_dict["snr_sep"] = -999.
            deepest_dict["back_sep"] = 0.
            deepest_dict["flux_sep"] = -999.
            deepest_dict["flux_sep_err"] = -999.
            deepest_dict["threshold_sep"] = -999.
            deepest_dict["limit_threshold"] = -999.
        deepest_dict["zeropoint_sep"] = deepest_img.zeropoint_best["zeropoint_img"]

        for instrument in self.photometry:
            for band in self.photometry[instrument]:
                for epoch in self.photometry[instrument][band]:
                    print(f"Extracting photometry for {self.name} in {instrument} {band}, epoch {epoch}.")
                    phot_dict = self.photometry[instrument][band][epoch]
                    if phot_dict["good_image_path"] == deepest_path:
                        continue
                    cls = image.CoaddedImage.select_child_class(instrument=instrument)
                    img = cls(path=phot_dict["good_image_path"])
                    mask_rp = deep_mask.reproject(
                        other_image=img,
                        output_path=os.path.join(
                            self.data_path,
                            f"{self.name_filesys}_mask_{phot_dict['instrument']}_{phot_dict['filter']}_{phot_dict['epoch_name']}.fits",
                        ),
                        write_footprint=False,
                        method="interp",
                        mask_mode=True
                    )
                    mag_results = img.sep_elliptical_magnitude(
                        centre=self.position_photometry,
                        a_world=self.a,  # + delta_fwhm,
                        b_world=self.b,  # + delta_fwhm,
                        theta_world=self.theta,
                        kron_radius=self.kron,
                        output=os.path.join(self.data_path, f"{self.name_filesys}_{instrument}_{band}_{epoch}"),
                        mask_nearby=mask_rp
                    )

                    if mag_results is not None:
                        phot_dict["mag_sep"] = mag_results["mag"][0]
                        phot_dict["mag_sep_err"] = mag_results["mag_err"][0]
                        phot_dict["snr_sep"] = mag_results["snr"][0]
                        phot_dict["back_sep"] = mag_results["back"][0]
                        phot_dict["flux_sep"] = mag_results["flux"][0]
                        phot_dict["flux_sep_err"] = mag_results["flux_err"][0]
                        phot_dict["limit_threshold"] = mag_results["threshold"]
                    else:
                        phot_dict["mag_sep"] = -999. * units.mag
                        phot_dict["mag_sep_err"] = -999. * units.mag
                        phot_dict["snr_sep"] = -999.
                        phot_dict["back_sep"] = 0.
                        phot_dict["flux_sep"] = -999.
                        phot_dict["flux_sep_err"] = -999.
                        phot_dict["threshold_sep"] = -999.
                        phot_dict["limit_threshold"] = -999.
                    phot_dict["zeropoint_sep"] = img.zeropoint_best["zeropoint_img"]
                    mag_results = img.sep_elliptical_magnitude(
                        centre=self.position_photometry,
                        a_world=self.a,  # + delta_fwhm,
                        b_world=self.b,  # + delta_fwhm,
                        theta_world=self.theta,
                        kron_radius=self.kron,
                        # output=os.path.join(self.data_path, f"{self.name_filesys}_{instrument}_{band}_{epoch}"),
                        mask_nearby=False
                    )
                    if mag_results is not None:
                        phot_dict["mag_sep_unmasked"] = mag_results["mag"][0]
                        phot_dict["mag_sep_unmasked_err"] = mag_results["mag_err"][0]
                        phot_dict["snr_sep_unmasked"] = mag_results["snr"][0]
                        phot_dict["flux_sep_unmasked"] = mag_results["flux"][0]
                        phot_dict["flux_sep_unmasked_err"] = mag_results["flux_err"][0]
                        phot_dict["limit_threshold"] = mag_results["threshold"]
                    else:
                        phot_dict["mag_sep_unmasked"] = -999. * units.mag
                        phot_dict["mag_sep_unmasked_err"] = -999. * units.mag
                        phot_dict["snr_sep"] = -999.
                        phot_dict["flux_sep_unmasked"] = -999.
                        phot_dict["flux_sep_unmasked_err"] = -999.
                        phot_dict["limit_threshold"] = -999.

        self.update_output_file()

    def add_photometry(
            self,
            instrument: Union[str, inst.Instrument],
            fil: Union[str, filters.Filter],
            epoch_name: str,
            mag: units.Quantity, mag_err: units.Quantity,
            snr: float,
            ellipse_a: units.Quantity, ellipse_a_err: units.Quantity,
            ellipse_b: units.Quantity, ellipse_b_err: units.Quantity,
            ellipse_theta: units.Quantity, ellipse_theta_err: units.Quantity,
            ra: units.Quantity, ra_err: units.Quantity,
            dec: units.Quantity, dec_err: units.Quantity,
            kron_radius: float,
            image_path: str,
            good_image_path: str = None,
            separation_from_given: units.Quantity = None,
            epoch_date: str = None,
            class_star: float = None,
            spread_model: float = None, spread_model_err: float = None,
            class_flag: int = None,
            mag_psf: units.Quantity = None, mag_psf_err: units.Quantity = None,
            snr_psf: float = None,
            image_depth: units.Quantity = None,
            do_mask: bool = True,
            **kwargs
    ):
        if good_image_path is None:
            good_image_path = image_path
        if isinstance(epoch_date, time.Time):
            epoch_date = epoch_date.strftime('%Y-%m-%d')
        photometry = {
            "instrument": str(instrument),
            "filter": str(fil),
            "epoch_name": epoch_name,
            "mag": u.check_quantity(mag, unit=units.mag),
            "mag_err": u.check_quantity(mag_err, unit=units.mag),
            "snr": float(snr),
            "a": u.check_quantity(ellipse_a, unit=units.arcsec, convert=True),
            "a_err": u.check_quantity(ellipse_a_err, unit=units.arcsec, convert=True),
            "b": u.check_quantity(ellipse_b, unit=units.arcsec, convert=True),
            "b_err": u.check_quantity(ellipse_b_err, unit=units.arcsec, convert=True),
            "theta": u.check_quantity(ellipse_theta, unit=units.deg, convert=True),
            "theta_err": u.check_quantity(ellipse_theta_err, unit=units.deg, convert=True),
            "ra": u.check_quantity(ra, units.deg, convert=True),
            "ra_err": u.check_quantity(ra_err, units.deg, convert=True),
            "dec": u.check_quantity(dec, units.deg, convert=True),
            "dec_err": u.check_quantity(dec_err, units.deg, convert=True),
            "kron_radius": float(kron_radius),
            "separation_from_given": u.check_quantity(separation_from_given, units.arcsec, convert=True),
            "epoch_date": str(epoch_date),
            "class_star": float(class_star),
            "spread_model": float(spread_model),
            "spread_model_err": float(spread_model_err),
            "class_flag": int(class_flag),
            "mag_psf": u.check_quantity(mag_psf, unit=units.mag),
            "mag_psf_err": u.check_quantity(mag_psf_err, unit=units.mag),
            "snr_psf": snr_psf,
            "image_depth": u.check_quantity(image_depth, unit=units.mag),
            "image_path": image_path,
            "good_image_path": good_image_path,
            "do_mask": do_mask,
        }

        if self.photometry_args is not None:
            photometry["a"] = self.a
            photometry["b"] = self.b
            photometry["a_err"] = 0 * units.arcsec
            photometry["b_err"] = 0 * units.arcsec
            photometry["theta"] = self.theta
            photometry["kron_radius"] = self.kron
            if "fix_pos" in self.photometry_args and self.photometry_args["fix_pos"]:
                photometry["ra"] = self.position_photometry.ra
                photometry["dec"] = self.position_photometry.dec

        kwargs.update(photometry)
        if instrument not in self.photometry:
            self.photometry[instrument] = {}
        if fil not in self.photometry[instrument]:
            self.photometry[instrument][fil] = {}
        self.photometry[instrument][fil][epoch_name] = kwargs
        self.update_output_file()
        return kwargs

    def find_in_cat(self, cat_name: str):
        # cat = self.field.load_catalogue(cat_name=cat_name)
        cat_path = self.field.get_path(f"cat_csv_{cat_name}")
        cat = table.QTable.read(cat_path)

        cols = r.cat_columns(cat_name)
        ra_col = cols["ra"]
        dec_col = cols["dec"]
        cat = astm.sanitise_coord(cat, dec_col)
        ra = cat[ra_col]
        dec = cat[dec_col]
        coord = SkyCoord(ra, dec, unit=units.deg)
        best_index, sep = astm.find_nearest(self.position, coord)
        return cat[best_index], sep

    def _output_dict(self):
        pos_phot_err = None
        if self.position_photometry_err is not None:
            pos_phot_err = self.position_photometry_err.to_dict()
        return {
            "position_photometry": self.position_photometry,
            "position_photometry_err": pos_phot_err,
            "photometry": self.photometry,
            "irsa_extinction_path": self.irsa_extinction_path,
            "extinction_law": self.extinction_power_law,

        }

    def load_output_file(self):
        self.check_data_path()
        if self.data_path is not None:
            outputs = p.load_output_file(self)
            if outputs is not None:
                if "position_photometry" in outputs and outputs["position_photometry"] is not None:
                    self.position_photometry = outputs["position_photometry"]
                if "position_photometry_err" in outputs and outputs["position_photometry_err"] is not None:
                    self.position_photometry_err = PositionUncertainty(
                        **outputs["position_photometry_err"],
                        position=self.position_photometry
                    )
                if "photometry" in outputs and outputs["photometry"] is not None:
                    self.photometry = outputs["photometry"]
                if "irsa_extinction_path" in outputs and outputs["irsa_extinction_path"] is not None:
                    self.irsa_extinction_path = outputs["irsa_extinction_path"]
            return outputs

    def check_data_path(self):
        if self.field is not None:
            u.debug_print(2, "", self.name)
            self.data_path = os.path.join(self.field.data_path, "objects", self.name_filesys)
            u.mkdir_check(self.data_path)
            self.output_file = os.path.join(self.data_path, f"{self.name_filesys}_outputs.yaml")
            return True
        else:
            return False

    def update_output_file(self):
        if self.check_data_path():
            p.update_output_file(self)

    def write_plot_photometry(self, output: str = None, **kwargs):
        """
        Plots available photometry (mag v lambda_eff) and writes to disk.
        :param output: Path to write plot.
        :return: matplotlib ax object containing plot info
        """
        if output is None:
            output = os.path.join(self.data_path, f"{self.name_filesys}_photometry.pdf")

        plt.close()
        axes = []
        for best in (False, True):
            ax = self.plot_photometry(**kwargs, best=best)
            ax.legend()
            if best:
                output = output.replace(".pdf", "_best.pdf")
            plt.savefig(output)
            axes.append(ax)
            plt.close()
        return axes

    def plot_photometry(
            self,
            ax=None,
            best: bool = False,
            **kwargs
    ):
        """
        Plots available photometry (mag v lambda_eff).
        :param ax: matplotlib ax object to plot with. A new object is generated if none is provided.
        :param kwargs:
        :return: matplotlib ax object containing plot info
        """
        if ax is None:
            fig, ax = plt.subplots()
        if "ls" not in kwargs:
            kwargs["ls"] = ""
        if "marker" not in kwargs:
            kwargs["marker"] = "x"
        if "ecolor" not in kwargs:
            kwargs["ecolor"] = "black"

        self.estimate_galactic_extinction()
        photometry_tbl = self.photometry_to_table(
            output=self.build_photometry_table_path(best=best),
            fmts=["ascii.ecsv", "ascii.csv"],
            best=best,
        )

        with quantity_support():

            valid = photometry_tbl[photometry_tbl["mag_sep"] > -990 * units.mag]
            plot_limit = (-999 * units.mag == valid["mag_sep_err"])
            limits = valid[plot_limit]
            mags = valid[np.invert(plot_limit)]

            ax.errorbar(
                mags["lambda_eff"],
                mags["mag_sep"],
                yerr=mags["mag_sep_err"],
                label="Magnitude",
                **kwargs,
            )
            ax.scatter(
                limits["lambda_eff"],
                limits["mag_sep"],
                label="Magnitude upper limit",
                marker="v",
            )
            ax.scatter(
                mags["lambda_eff"],
                mags["mag_sep_ext_corrected"],
                color="orange",
                label="Corrected for Galactic extinction"
            )
            ax.scatter(
                limits["lambda_eff"],
                limits["mag_sep_ext_corrected"],
                label="Magnitude upper limit",
                marker="v",
            )
            ax.set_ylabel("Apparent magnitude")
            ax.set_xlabel("$\lambda_\mathrm{eff}$ (\AA)")
            ax.invert_yaxis()
        return ax

    def build_photometry_table_path(self, best: bool = False):
        self.check_data_path()
        if best:
            return os.path.join(self.data_path, f"{self.name_filesys}_photometry_best.ecsv")
        else:
            return os.path.join(self.data_path, f"{self.name_filesys}_photometry.ecsv")

    # def update_position_from_photometry(self):
    #     self.photometry_to_table()
    #     best = self.select_deepest_sep(local_output=False)
    #     self.position = SkyCoord(best["ra"], best["dec"], unit=units.deg)
    #     self.position_err = PositionUncertainty(
    #         ra_err_stat=best["ra_err"],
    #         dec_err_stat=best["dec_err"]
    #     )

    def photometry_to_table(
            self,
            output: str = None,
            best: bool = False,
            fmts: List[str] = ("ascii.ecsv", "ascii.csv")
    ):
        """
        Converts the photometry information, which is stored internally as a dictionary, into an astropy QTable.
        :param output: Where to write table.
        :return:
        """

        if output is None:
            output = self.build_photometry_table_path()

        tbls = []
        for instrument_name in self.photometry:
            instrument = inst.Instrument.from_params(instrument_name)
            for filter_name in self.photometry[instrument_name]:
                fil = instrument.filters[filter_name]

                if best:
                    phot_dict, _ = self.select_photometry_sep(fil=filter_name, instrument=instrument_name)
                    phot_dict["band"] = filter_name
                    phot_dict["instrument"] = instrument_name
                    phot_dict["lambda_eff"] = u.check_quantity(
                        number=fil.lambda_eff,
                        unit=units.Angstrom
                    )
                    # tbl = table.QTable([phot_dict])
                    tbls.append(phot_dict)

                else:
                    for epoch in self.photometry[instrument_name][filter_name]:
                        phot_dict = self.photometry[instrument_name][filter_name][epoch].copy()
                        phot_dict["band"] = filter_name
                        phot_dict["instrument"] = instrument_name
                        phot_dict["lambda_eff"] = u.check_quantity(
                            number=fil.lambda_eff,
                            unit=units.Angstrom
                        )
                        # tbl = table.QTable([phot_dict])
                        tbls.append(phot_dict)

        if best:
            photometry_tbl = table.vstack(tbls)
            self.photometry_tbl_best = photometry_tbl.copy()
        else:
            photometry_tbl = table.QTable(tbls)
            self.photometry_tbl = photometry_tbl.copy()

        if output is not False:
            for fmt in fmts:
                u.detect_problem_table(photometry_tbl, fmt="csv")
                photometry_tbl.write(output.replace(".ecsv", fmt[fmt.find("."):]), format=fmt, overwrite=True)
        return photometry_tbl

    # def sandf_galactic_extinction(
    #     self,
    #     band: List[filters.Filter]
    # ):
    #     import extinction
    #     extinction.fitzpatrick99(tbl["lambda_eff"], a_v, r_v) * units.mag
    #     pass

    def estimate_galactic_extinction(self, ax=None, r_v: float = 3.1, **kwargs):
        import extinction
        if ax is None:
            fig, ax = plt.subplots()
        if "marker" not in kwargs:
            kwargs["marker"] = "x"

        self.retrieve_extinction_table()
        lambda_eff_tbl = self.irsa_extinction["LamEff"].to(
            units.Angstrom)
        power_law = models.PowerLaw1D()
        fitter = fitting.LevMarLSQFitter()
        fitted = fitter(power_law, lambda_eff_tbl, self.irsa_extinction["A_SandF"].value)

        tbl = self.photometry_to_table(fmts=["ascii.ecsv", "ascii.csv"])

        x = np.linspace(0, 80000, 1000) * units.Angstrom

        a_v = (r_v * self.ebv_sandf).value

        tbl["ext_gal_sandf"] = extinction.fitzpatrick99(tbl["lambda_eff"], a_v, r_v) * units.mag
        tbl["ext_gal_pl"] = fitted(tbl["lambda_eff"]) * units.mag
        tbl["ext_gal_interp"] = np.interp(
            tbl["lambda_eff"],
            lambda_eff_tbl,
            self.irsa_extinction["A_SandF"].value
        ) * units.mag

        ax.plot(
            x, extinction.fitzpatrick99(x, a_v, r_v),
            label="S\&F + F99 extinction law",
            c="red"
        )
        ax.plot(
            x, fitted(x),
            label=f"power law fit to IRSA",
            # , \\alpha={fitted.alpha.value}; $x_0$={fitted.x_0.value}; A={fitted.amplitude.value}",
            c="blue"
        )
        ax.scatter(
            lambda_eff_tbl, self.irsa_extinction["A_SandF"].value,
            label="from IRSA",
            c="green",
            **kwargs)
        ax.scatter(
            tbl["lambda_eff"], tbl["ext_gal_pl"].value,
            label="power law interpolation of IRSA",
            c="blue",
            **kwargs
        )
        ax.scatter(
            tbl["lambda_eff"], tbl["ext_gal_interp"].value,
            label="numpy interpolation from IRSA",
            c="violet",
            **kwargs
        )
        ax.scatter(
            tbl["lambda_eff"], tbl["ext_gal_sandf"].value,
            label="S\&F + F99 extinction law",
            c="red",
            **kwargs
        )
        ax.set_ylim(0, 0.6)
        ax.legend()
        plt.savefig(os.path.join(self.data_path, f"{self.name_filesys}_irsa_extinction.pdf"))
        plt.close()
        self.extinction_power_law = {
            "amplitude": fitted.amplitude.value * fitted.amplitude.unit,
            "x_0": fitted.x_0.value,
            "alpha": fitted.alpha.value
        }

        for row in tbl:
            instrument = row["instrument"]
            band = row["band"]
            epoch_name = row["epoch_name"]

            # if row["lambda_eff"] > max(lambda_eff_tbl) or row["lambda_eff"] < min(lambda_eff_tbl):
            #     key = "ext_gal_pl"
            #     self.photometry[instrument][band]["ext_gal_type"] = "power_law_fit"
            # else:
            #     key = "ext_gal_interp"
            #     self.photometry[instrument][band]["ext_gal_type"] = "interpolated"
            key = "ext_gal_sandf"
            self.photometry[instrument][band][epoch_name]["ext_gal_type"] = "s_and_f"
            self.photometry[instrument][band][epoch_name]["ext_gal"] = row[key]
            self.photometry[instrument][band][epoch_name]["mag_ext_corrected"] = row["mag"] - row[key]
            if "mag_sep" in row.colnames:
                if row["mag_sep"] > -99 * units.mag:
                    self.photometry[instrument][band][epoch_name]["mag_sep_ext_corrected"] = row["mag_sep"] - row[key]
                else:
                    self.photometry[instrument][band][epoch_name]["mag_sep_ext_corrected"] = -999 * units.mag
        # tbl_2 = self.photometry_to_table()
        # tbl_2.update(tbl)
        # tbl_2.write(self.build_photometry_table_path().replace("photometry", "photemetry_extended"))
        self.update_output_file()
        return ax

    def retrieve_extinction_table(self, force: bool = False):
        self.load_extinction_table()
        self.check_data_path()
        if force or self.irsa_extinction is None:
            raw_path = os.path.join(self.data_path, f"{self.name_filesys}_irsa_extinction.ecsv")
            r.save_irsa_extinction(
                ra=self.position.ra.value,
                dec=self.position.dec.value,
                output=raw_path
            )
            ext_tbl = table.QTable.read(raw_path, format="ascii")
            for colname in ext_tbl.colnames:
                if str(ext_tbl[colname].unit) == "mags":
                    ext_tbl[colname]._set_unit(units.mag)
            tbl_path = os.path.join(self.data_path, f"{self.name_filesys}_galactic_extinction.ecsv")
            ext_tbl.write(tbl_path, overwrite=True, format="ascii.ecsv")
            self.irsa_extinction = ext_tbl
            self.irsa_extinction_path = tbl_path

        if force or self.ebv_sandf is None:
            # Get E(B-V) at this coordinate.
            tbl = r.retrieve_irsa_details(coord=self.position)
            self.ebv_sandf = tbl["ext SandF ref"][0] * units.mag

    def load_extinction_table(self, force: bool = False):
        if force or self.irsa_extinction is None:
            if self.irsa_extinction_path is not None:
                u.debug_print(1, "Loading irsa_extinction from", self.irsa_extinction_path)
                self.irsa_extinction = table.QTable.read(self.irsa_extinction_path, format="ascii.ecsv")

    def jname(self):
        if self.position is not None:
            name = astm.jname(
                coord=self.position,
                ra_precision=2,
                dec_precision=1
            )
            if self.name is None:
                self.name = name
            return name

    def get_photometry_table(self, output: bool = False, best: bool = False, force: bool = False):
        if not self.photometry:
            self.load_output_file()
        if output is True:
            output = None
        tbl = None
        if best and (force or self.photometry_tbl_best is None or len(self.photometry_tbl_best) == 0):
            tbl = self.photometry_to_table(output=output, best=True)
        elif not best and (force or self.photometry_tbl is None or len(self.photometry_tbl) == 0):
            tbl = self.photometry_to_table(output=output, best=False)
        return tbl

    def select_photometry(self, fil: str, instrument: str, local_output: bool = True):
        self.get_photometry_table(output=local_output, best=True)
        fil_photom = self.photometry_tbl_best[self.photometry_tbl_best["band"] == fil]
        fil_photom = fil_photom[fil_photom["instrument"] == instrument]
        row = fil_photom[np.argmax(fil_photom["snr"])]
        photom_dict = self.photometry[instrument][fil][row["epoch_name"]]

        if len(fil_photom) > 1:
            mag_psf_err = np.std(fil_photom["mag_psf"]) / len(fil_photom)
            mag_err = np.std(fil_photom["mag_sep"]) / len(fil_photom)
        else:
            mag_psf_err = fil_photom["mag_psf_err"][0]
            mag_err = fil_photom["mag_sep_err"][0]

        mean = {
            "mag": np.mean(fil_photom["mag"]),
            "mag_err": mag_err,
            "mag_psf": np.mean(fil_photom["mag_psf"]),
            "mag_psf_err": mag_psf_err,
            "n": len(fil_photom)
        }
        # TODO: Just meaning the whole table is probably not the best way to estimate uncertainties.
        return photom_dict, mean

    def select_photometry_sep(
            self,
            fil: str,
            instrument: str,
            local_output: bool = True
    ):
        self.get_photometry_table(output=local_output, best=False)
        fil_photom = self.photometry_tbl[self.photometry_tbl["band"] == fil]
        fil_photom = fil_photom[fil_photom["instrument"] == instrument]
        row = fil_photom[np.argmax(fil_photom["snr_sep"])]
        photom_dict = self.photometry[instrument][fil][row["epoch_name"]]

        if len(fil_photom) > 1:
            mag_psf_err = np.std(fil_photom["mag_psf"]) / len(fil_photom)
            mag_err = np.std(fil_photom["mag_sep"]) / len(fil_photom)
        else:
            mag_psf_err = fil_photom["mag_psf_err"][0]
            mag_err = fil_photom["mag_sep_err"][0]

        mean = {
            "mag": np.mean(fil_photom["mag_sep"]),
            "mag_err": mag_err,
            "mag_psf": np.mean(fil_photom["mag_psf"]),
            "mag_psf_err": mag_psf_err,
            "n": len(fil_photom)
        }
        u.debug_print(2, f"Object.select_photometry_sep(): {self.name=}, {fil=}, {instrument=}")
        return photom_dict, mean

    def select_psf_photometry(self, local_output: bool = True):
        self.get_photometry_table(output=local_output, best=True)
        idx = np.argmax(self.photometry_tbl_best["snr_psf"])
        row = self.photometry_tbl_best[idx]
        return self.photometry[row["instrument"]][row["band"]][row["epoch_name"]]

    def select_best_position(self, local_output: bool = True):
        self.get_photometry_table(output=local_output)
        idx = np.argmin(self.photometry_tbl_best["ra_err"] * self.photometry_tbl_best["dec_err"])
        row = self.photometry_tbl_best[idx]
        return self.photometry[row["instrument"]][row["band"]][row["epoch_name"]]

    def select_deepest(self, local_output: bool = True):
        self.get_photometry_table(output=local_output, best=False)
        if "snr" not in self.photometry_tbl.colnames:
            return None
        idx = np.argmax(self.photometry_tbl["snr"])
        row = self.photometry_tbl[idx]
        deepest = self.photometry[row["instrument"]][row["band"]][row["epoch_name"]]
        # if self.photometry_args is None:
        self.a = deepest["a"]
        self.b = deepest["b"]
        self.theta = deepest["theta"]
        self.kron = deepest["kron_radius"]
        ra = deepest["ra"]
        dec = deepest["dec"]
        try:
            self.position_photometry = SkyCoord(ra, dec)
        except ValueError:
            print("Deepest observation is a limit only.")
        # else:
        #     deepest["a"] = self.a
        #     deepest["b"] = self.b
        #     deepest["theta"] = self.theta
        #     deepest["kron_radius"] = self.kron
        #     if "fix_pos" in self.photometry_args and self.photometry_args["fix_pos"]:
        #         deepest["ra"] = self.position.ra
        #         deepest["dec"] = self.position.dec
        #     else:
        #         ra = deepest["ra"]
        #         dec = deepest["dec"]
        #         self.position = SkyCoord(ra, dec)

        return deepest

    def select_deepest_sep(self, local_output: bool = True):
        self.get_photometry_table(output=local_output, best=True)
        if "snr_sep" not in self.photometry_tbl_best.colnames:
            return None
        idx = np.argmax(self.photometry_tbl_best["snr_sep"])
        row = self.photometry_tbl_best[idx]
        return self.photometry[row["instrument"]][row["band"]][row["epoch_name"]]

    def push_to_table(self, select: bool = False, local_output: bool = True):

        jname = self.jname()

        self.estimate_galactic_extinction()
        if select:
            self.get_good_photometry()
            self.photometry_to_table()
            deepest = self.select_deepest_sep(local_output=local_output)
        else:
            deepest = self.select_deepest(local_output=local_output)

        # best_position = self.select_best_position(local_output=local_output)
        best_psf = self.select_psf_photometry(local_output=local_output)

        row = {
            "jname": jname,
            "field_name": self.field.name,
            "object_name": self.name,
            "ra": deepest["ra"],
            "ra_err": deepest["ra_err"],
            "dec": deepest["dec"],
            "dec_err": deepest["dec_err"],
            "epoch_position": deepest["epoch_name"],
            "epoch_position_date": deepest["epoch_date"],
            "a": self.a,
            "a_err": deepest["a_err"],
            "b": self.b,
            "b_err": deepest["b_err"],
            "theta": self.theta,
            "kron_radius": self.kron,
            "epoch_ellipse": deepest["epoch_name"],
            "epoch_ellipse_date": deepest["epoch_date"],
            "theta_err": deepest["theta_err"],
            f"e_b-v": self.ebv_sandf,
            f"class_star": best_psf["class_star"],
            "spread_model": best_psf["spread_model"],
            "spread_model_err": best_psf["spread_model_err"],
            "class_flag": best_psf["class_flag"],
        }

        if isinstance(self, Galaxy) and self.z is not None:
            row["z"] = self.z
            row["d_A"] = self.D_A
            row["d_L"] = self.D_L
            row["mu"] = self.mu

        if isinstance(self, TransientHostCandidate):
            row["transient_tns_name"] = self.transient.tns_name

        for instrument in self.photometry:
            for fil in self.photometry[instrument]:

                band_str = f"{instrument}_{fil.replace('_', '-')}"

                if select:
                    best_photom, mean_photom = self.select_photometry_sep(fil, instrument, local_output=local_output)
                    row[f"mag_best_{band_str}"] = best_photom["mag_sep"]
                    row[f"mag_best_{band_str}_err"] = best_photom["mag_sep_err"]
                    row[f"snr_best_{band_str}"] = best_photom["snr_sep"]

                else:
                    best_photom, mean_photom = self.select_photometry(fil, instrument, local_output=local_output)
                    row[f"mag_best_{band_str}"] = best_photom["mag"]
                    row[f"mag_best_{band_str}_err"] = best_photom["mag_err"]
                    row[f"snr_best_{band_str}"] = best_photom["snr"]

                row[f"mag_mean_{band_str}"] = mean_photom["mag"]
                row[f"mag_mean_{band_str}_err"] = mean_photom["mag_err"]
                row[f"n_mean_{band_str}"] = mean_photom["n"]
                row[f"ext_gal_{band_str}"] = best_photom["ext_gal"]
                # else:
                #     row[f"ext_gal_{band_str}"] = best_photom["ext_gal_sandf"]
                row[f"epoch_best_{band_str}"] = best_photom[f"epoch_name"]
                row[f"epoch_best_date_{band_str}"] = str(best_photom[f"epoch_date"])
                row[f"mag_psf_best_{band_str}"] = best_photom[f"mag_psf"]
                row[f"mag_psf_best_{band_str}_err"] = best_photom[f"mag_psf_err"]
                row[f"snr_psf_best_{band_str}"] = best_photom["snr_psf"]
                row[f"mag_psf_mean_{band_str}"] = mean_photom[f"mag_psf"]
                row[f"mag_psf_mean_{band_str}_err"] = mean_photom[f"mag_psf_err"]

        # colnames = obs.master_objects_columns
        # for colname in colnames:
        #     if colname not in row:
        #         if "epoch" in colname:
        #             row[colname] = "N/A"
        #         else:
        #             row[colname] = tbl[0][colname]

        u.debug_print(2, "Object.push_to_table(): select ==", select)
        if select:
            tbl = obs.load_master_objects_table()
        else:
            tbl = obs.load_master_all_objects_table()

        obs.add_photometry(
            tbl=tbl,
            object_name=self.name,
            entry=row,
        )
        obs.write_master_objects_table()

    # def plot_ellipse(
    #         self,
    #         plot,
    #         img,
    #         ext: int = 0,
    #         colour: str = "white",
    # ):

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "position": copy.deepcopy(position_dictionary),
            "position_err": copy.deepcopy(PositionUncertainty.default_params()),
            "type": None,
            "photometry_args_manual":
                {
                    "a": 0.0 * units.arcsec,
                    "b": 0.0 * units.arcsec,
                    "theta": 0.0 * units.arcsec,
                    "kron_radius": 3.5
                },
            "plotting":
                {
                    "frame": None
                },
            "publication_doi": None
        }
        return default_params

    @classmethod
    def from_dict(cls, dictionary: dict, field=None) -> 'Object':
        """
        Construct an Object or appropriate child class (FRB, Galaxy...) from a passed dict.
        :param dictionary: dict with keys:
            'position': position dictionary as given by position_dictionary
            'position_err':
        :return: Object reflecting dictionary.
        """
        dict_pristine = dictionary.copy()
        ra, dec = p.select_coords(dictionary.pop("position"))
        if "position_err" in dictionary:
            position_err = dictionary.pop("position_err")
        else:
            position_err = PositionUncertainty.default_params()

        if "type" in dictionary and dictionary["type"] is not None:
            selected = cls.select_child_class(obj_type=dictionary["type"])
        else:
            selected = cls

        if "plotting" in dictionary:
            plotting = dictionary.pop("plotting")
        else:
            plotting = None

        if "name" in dictionary:
            name = dictionary.pop("name")
        else:
            name = None

        if selected in (Object, FRB):
            return selected(
                name=name,
                position=f"{ra} {dec}",
                position_err=position_err,
                field=field,
                plotting=plotting,
                **dictionary
            )
        else:
            return selected.from_dict(dictionary=dict_pristine, field=field)

    @classmethod
    def select_child_class(cls, obj_type: str):
        obj_type = obj_type.lower()
        if obj_type == "galaxy":
            return Galaxy
        elif obj_type == "frb":
            return FRB
        elif obj_type == "star":
            return Object
        else:
            raise ValueError(f"Didn't recognise obj_type '{obj_type}'")

    @classmethod
    def from_source_extractor_row(cls, row: table.Row, use_psf_params: bool = False):
        if use_psf_params:
            ra_key = "ALPHAPSF_SKY"
            dec_key = "DELTAPSF_SKY"
            ra_err_key = "ERRX2_WORLD"
            dec_err_key = "ERRY2_WORLD"
        else:
            ra_key = "ALPHA_SKY"
            dec_key = "DELTA_SKY"
            ra_err_key = "ERRX2PSF_WORLD"
            dec_err_key = "ERRY2PSF_WORLD"
        ra_err = np.sqrt(row[ra_err_key])
        dec_err = np.sqrt(row[dec_err_key])
        obj = cls(name=str(row["NUMBER"]),
                  position=SkyCoord(row[ra_key], row[dec_key]),
                  position_err=PositionUncertainty(
                      ra_err_stat=ra_err,
                      dec_err_stat=dec_err),
                  )
        obj.cat_row = row
        return obj


@u.export
class Star(Object):
    pass


@u.export
class Extragalactic(Object):
    def __init__(
            self,
            z: float = None,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.z = None
        self.z_err = None
        self.D_A = None
        self.D_L = None
        self.mu = None
        self.set_z(z, **kwargs)

    def set_z(self, z, **kwargs):
        self.z = z
        if z is not None:
            if "z_err" in kwargs:
                self.z_err = kwargs["z_err"]
            self.D_A = self.angular_size_distance()
            self.D_L = self.luminosity_distance()
            self.mu = self.distance_modulus()

    def angular_size_distance(self):
        if self.z is not None:
            return cosmology.angular_diameter_distance(z=self.z)

    def luminosity_distance(self):
        if self.z is not None:
            return cosmology.luminosity_distance(z=self.z)

    def comoving_distance(self):
        if self.z is not None:
            return cosmology.comoving_distance(z=self.z)

    def distance_modulus(self):
        d = self.luminosity_distance()
        if d is not None:
            mu = distance_modulus(d)
            return mu

    def absolute_magnitude(
            self,
            apparent_magnitude: units.Quantity,
            internal_extinction: units.Quantity = 0 * units.mag,
            galactic_extinction: units.Quantity = 0 * units.mag
    ):
        mu = self.distance_modulus()
        return apparent_magnitude - mu - internal_extinction - galactic_extinction

    def absolute_photometry(self, internal_extinction: units.Quantity = 0.0 * units.mag):
        for instrument in self.photometry:
            for fil in self.photometry[instrument]:
                for epoch in self.photometry[instrument][fil]:
                    abs_mag = self.absolute_magnitude(
                        apparent_magnitude=self.photometry[instrument][fil][epoch]["mag"],
                        internal_extinction=internal_extinction
                    )
                    self.photometry[instrument][fil][epoch]["abs_mag"] = abs_mag
        self.update_output_file()

    def projected_size(self, angle: Union[units.Quantity, float]) -> units.Quantity:
        """
        When given an angular size, calculates the projected physical size at the redshift of the galaxy.
        :param angle: Angular size. If not provided as a quantity, must be in arcseconds.
        :return: Projected physical size, with units kpc
        """
        angle = u.check_quantity(angle, unit=units.arcsec).to(units.rad).value
        dist = angle * self.D_A
        return dist.to(units.kpc)

    def angular_size(self, distance: Union[units.Quantity, float]):
        """
        Given a physical projected size at the redshift of the galaxy, calculates the angular size as seen from Earth.
        :param distance: Physical projected size. If not provided as a quantity, must be in kiloparsecs.
        :return: Angular size, in arcseconds.
        """
        distance = u.check_quantity(distance, unit=units.kpc)
        theta = (distance * units.rad / self.D_A).to(units.arcsec)
        return theta


@u.export
class Galaxy(Extragalactic):
    def __init__(
            self,
            z: float = None,
            **kwargs
    ):
        super().__init__(
            z=z,
            **kwargs
        )

        self.mass = None
        if "mass" in kwargs:
            self.mass = kwargs["mass"]

        self.mass_stellar = None
        if "mass_stellar" in kwargs:
            self.mass_stellar = u.check_quantity(kwargs["mass_stellar"], units.solMass)

        self.mass_stellar_err_plus = None
        self.mass_stellar_err_minus = None
        if "mass_stellar_err_plus" in kwargs:
            self.mass_stellar_err_plus = u.check_quantity(kwargs["mass_stellar_err_plus"], units.solMass)
        elif "mass_stellar_err" in kwargs:
            self.mass_stellar_err_plus = u.check_quantity(kwargs["mass_stellar_err"],
                                                          units.solMass)
        if "mass_stellar_err_minus" in kwargs:
            self.mass_stellar_err_minus = u.check_quantity(kwargs["mass_stellar_err_minus"], units.solMass)
        elif "mass_stellar_err" in kwargs:
            self.mass_stellar_err_minus = u.check_quantity(kwargs["mass_stellar_err"],
                                                           units.solMass)

        self.sfr = None
        if "sfr" in kwargs:
            self.sfr = kwargs["sfr"] * units.solMass

        self.sfr_err = None
        if "sfr_err" in kwargs:
            self.sfr_err = kwargs["sfr_err"] * units.solMass

        self.mass_halo = None
        self.log_mass_halo = None
        self.log_mass_halo_upper = None
        self.log_mass_halo_lower = None
        if "mass_halo" in kwargs:
            self.mass_halo = u.check_quantity(kwargs["mass_halo"], units.solMass)
            self.log_mass_halo = np.log10(self.mass_halo / units.solMass)

        self.halo_mnfw = None
        self.halo_yf17 = None
        self.halo_mb15 = None
        self.halo_mb04 = None

        self.cigale_model_path = None
        self.cigale_model = None

        self.cigale_sfh_path = None
        self.cigale_sfh = None

        self.cigale_results_path = None
        self.cigale_results = None

    def load_cigale_model(self, force: bool = False):
        if self.cigale_model_path is None:
            print(f"Cannot load CIGALE model; {self}.cigale_model_path has not been set.")
        elif force or self.cigale_model is None:
            self.cigale_model = fits.open(self.cigale_model_path)

        if self.cigale_sfh_path is None:
            print(f"Cannot load CIGALE SFH; {self}.cigale_sfh_path has not been set.")
        elif force or self.cigale_sfh is None:
            self.cigale_sfh = fits.open(self.cigale_sfh_path)

        return self.cigale_model, self.cigale_sfh  # , self.cigale_results

    def _output_dict(self):
        output = super()._output_dict()
        output.update({
            "mass_stellar": self.mass_stellar,
            "mass_stellar_err_plus": self.mass_stellar_err_plus,
            "mass_stellar_err_minus": self.mass_stellar_err_minus,
            "sfr": self.sfr,
            "sfr_err": self.sfr_err,
            "cigale_model_path": self.cigale_model_path,
            "cigale_sfh_path": self.cigale_sfh_path,
            "cigale_results": self.cigale_results
        })
        return output

    def load_output_file(self):
        outputs = super().load_output_file()
        if outputs is not None:
            if "mass_stellar" in outputs and outputs["mass_stellar"] is not None:
                self.mass_stellar = outputs["mass_stellar"]
            if "mass_stellar_err" in outputs and outputs["mass_stellar_err"] is not None:
                self.mass_stellar_err = outputs["mass_stellar_err"]
            if "sfr" in outputs and outputs["sfr"] is not None:
                self.sfr = outputs["sfr"]
            if "sfr_err" in outputs and outputs["sfr_err"] is not None:
                self.sfr_err = outputs["sfr_err"]
            if "cigale_model_path" in outputs and outputs["cigale_model_path"] is not None:
                self.cigale_model_path = outputs["cigale_model_path"]
            if "cigale_sfh_path" in outputs and outputs["cigale_sfh_path"] is not None:
                self.cigale_sfh_path = outputs["cigale_sfh_path"]
            if "cigale_results" in outputs and outputs["cigale_results"] is not None:
                self.cigale_results = outputs["cigale_results"]
        return outputs

    def h(self):
        return cosmology.H(z=self.z) / (100 * units.km * units.second ** -1 * units.Mpc ** -1)

    def halo_mass(self):
        from frb.halos.utils import halomass_from_stellarmass
        if self.mass_stellar is None:
            raise ValueError(f"{self}.mass_stellar has not been defined.")
        self.log_mass_halo = halomass_from_stellarmass(
            log_mstar=np.log10(self.mass_stellar / units.solMass),
            z=self.z
        )
        self.mass_halo = (10 ** self.log_mass_halo) * units.solMass
        if self.mass_stellar_err_plus is None:
            self.mass_stellar_err_plus = 0. * units.solMass
        self.log_mass_halo_upper = halomass_from_stellarmass(
            log_mstar=np.log10((self.mass_stellar + self.mass_stellar_err_plus) / units.solMass),
            z=self.z
        )
        if self.mass_stellar_err_minus is None:
            self.mass_stellar_err_minus = 0. * units.solMass
        self.log_mass_halo_lower = halomass_from_stellarmass(
            log_mstar=np.log10((self.mass_stellar - self.mass_stellar_err_minus) / units.solMass),
            z=self.z
        )

        return self.mass_halo, self.log_mass_halo

    def halo_concentration_parameter(self):
        if self.log_mass_halo is None:
            self.halo_mass()
        c = 4.67 * (self.mass_halo / (10 ** 14 * self.h() ** -1 * units.solMass)) ** (-0.11)
        return float(c)

    def halo_model_mnfw(self, y0=2., alpha=2., **kwargs):
        from frb.halos.models import ModifiedNFW
        if self.log_mass_halo is None:
            self.halo_mass()
        self.halo_mnfw = ModifiedNFW(
            log_Mhalo=self.log_mass_halo,
            z=self.z,
            cosmo=cosmology,
            c=self.halo_concentration_parameter(),
            y0=y0,
            alpha=alpha,
            **kwargs
        )
        self.halo_mnfw.coord = self.position
        return self.halo_mnfw

    def halo_model_yf17(self, **kwargs):
        from frb.halos.models import YF17
        if self.log_mass_halo is None:
            self.halo_mass()
        self.halo_yf17 = YF17(
            log_Mhalo=self.log_mass_halo,
            z=self.z,
            cosmo=cosmology,
            **kwargs
        )
        return self.halo_yf17

    def halo_model_mb04(self, r_c=147 * units.kpc, **kwargs):
        from frb.halos.models import MB04
        if self.log_mass_halo is None:
            self.halo_mass()
        self.halo_mb04 = MB04(
            log_Mhalo=self.log_mass_halo,
            z=self.z,
            cosmo=cosmology,
            c=self.halo_concentration_parameter(),
            Rc=r_c,
            **kwargs
        )
        return self.halo_mb04

    def halo_model_mb15(self, **kwargs):
        from frb.halos.models import MB15
        if self.log_mass_halo is None:
            self.halo_mass()
        self.halo_mb15 = MB15(
            log_Mhalo=self.log_mass_halo,
            z=self.z,
            cosmo=cosmology,
            **kwargs
        )
        return self.halo_mb15

    def halo_dm_cum(
            self,
            rmax: float = 1.,
            rperp: units.Quantity = 0. * units.kpc,
            step_size: units.Quantity = 0.1 * units.kpc
    ):
        d, dm = self.halo_mnfw.Ne_Rperp(
            rperp,
            step_size=step_size,
            rmax=rmax,
            cumul=True
        )
        tbl = table.QTable({
            "d": d * units.kpc,
            "d_abs": d * units.kpc + self.comoving_distance(),
            "DM": dm * dm_units / (1 + self.z),
        })
        return tbl

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        default_params.update({
            "z": None,
            "z_err": None,
            "type": "galaxy"
        })
        return default_params

    # TODO: There do not need to be separate methods per class for this. Just pass dictionary as a **kwargs and be done with it
    @classmethod
    def from_dict(cls, dictionary: dict, field=None):
        ra, dec = p.select_coords(dictionary.pop("position"))
        if "position_err" in dictionary:
            position_err = dictionary.pop("position_err")
        else:
            position_err = PositionUncertainty.default_params()
        return cls(name=dictionary.pop("name"),
                   position=f"{ra} {dec}",
                   position_err=position_err,
                   z=dictionary.pop("z"),
                   field=field,
                   **dictionary)


@u.export
class TransientHostCandidate(Galaxy):
    def __init__(
            self,
            transient: 'Transient',
            z: float = 0.0,
            **kwargs
    ):
        super().__init__(
            z=z,
            **kwargs
        )
        self.transient = transient

        self.P_O = None
        if "P_O" in kwargs:
            self.P_O = kwargs["P_O"]
        self.P_xO = None
        if "P_xO" in kwargs:
            self.P_xO = kwargs["P_xO"]
        self.P_Ox = None
        if "P_Ox" in kwargs:
            self.P_Ox = kwargs["P_Ox"]


dm_units = units.parsec * units.cm ** -3

dm_host_median = {
    "james_22A": 129 * dm_units,
    "james_22B": 186 * dm_units
}


@u.export
class Transient(Object):
    def __init__(
            self,
            host_galaxy: TransientHostCandidate = None,
            date: time.Time = None,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.host_galaxy = host_galaxy
        self.host_candidate_tables = {}
        self.host_candidates = []
        if not isinstance(date, time.Time) and date is not None:
            date = time.Time(date)
        self.date = date
        self.tns_name = None
        if "tns_name" in kwargs:
            self.tns_name = kwargs["tns_name"]


@u.export
class FRB(Transient):
    def __init__(
            self,
            dm: Union[float, units.Quantity] = None,
            **kwargs
    ):
        """
        Initialise.

        :param dm: The dispersion measure of the FRB. If provided without units, pc cm^-3 will be assumed.
        :param kwargs:
        """
        super().__init__(
            **kwargs
        )
        self.dm = dm
        if self.dm is not None:
            self.dm = u.check_quantity(self.dm, unit=dm_units)

        self.x_frb = None

    def generate_x_frb(self):
        from frb.frb import FRB
        self.x_frb = FRB(
            frb_name=self.name,
            coord=self.position,
            DM=self.dm
        )
        return self.x_frb

    def probabilistic_association(
            self,
            p_u: float,
            img: 'craftutils.observation.image.ImagingImage',
            radius: float = 10
    ):
        """
        Performs a customised PATH run on an image.

        :param p_u: The prior for the probability of the host being unseen in the image.
        :param img: The image on which to run PATH.
        :param frb_object: the FRB in question.
        :param radius: Maximum distance in arcseconds for an object to be considered as a candidate.
        :return:
        """
        import frb.associate.frbassociate as associate
        import astropath.path as path
        astm_rms = img.extract_astrometry_err()
        a, b = self.position_err.uncertainty_quadrature()
        a = np.sqrt(a ** 2 + astm_rms ** 2)
        b = np.sqrt(b ** 2 + astm_rms ** 2)
        x_frb = self.generate_x_frb()
        x_frb.set_ee(
            a=a.value,
            b=b.value,
            theta=0.,
            cl=0.68,
        )
        #     img.load_output_file()
        img.extract_pixel_scale()
        instname = img.instrument.name.replace("-", "_").upper()
        filname = f'{instname}_{img.filter.band_name}'
        config = dict(
            max_radius=radius,
            skip_bayesian=False,
            npixels=9,
            image_file=img.path,
            cut_size=30.,
            filter=filname,
            ZP=img.zeropoint_best["zeropoint_img"].value,
            deblend=True,
            cand_bright=17.,
            cand_separation=radius * units.arcsec,
            plate_scale=(1 * units.pix).to(units.arcsec, img.pixel_scale_y),
        )
        print("P(U) ==", p_u)
        priors = path.priors.load_std_priors()["adopted"]
        priors["U"] = p_u
        try:
            ass = associate.run_individual(
                config=config,
                #         show=True,
                #         verbose=True,
                FRB=x_frb,

                prior=priors
                #     skip_bayesian=True
            )
            p_ux = ass.P_Ux
            print("P(U|x) ==", p_ux)
            cand_tbl = table.QTable.from_pandas(ass.candidates)
            p_ox = cand_tbl[0]["P_Ox"]
            print("Max P(O|x_i) ==", p_ox)
            print("\n\n")
            cand_tbl["ra"] *= units.deg
            cand_tbl["dec"] *= units.deg
            cand_tbl["separation"] *= units.arcsec
            cand_tbl[filname] *= units.mag
            self.host_candidate_tables[img.name] = cand_tbl
            self.update_output_file()
        except IndexError:
            cand_tbl = None
            p_ox = None
            p_ux = None

        return cand_tbl, p_ox, p_ux

    def consolidate_candidate_tables(self, sort_by="separation"):
        # Build a shared catalogue of host candidates.
        path_cat = None
        for tbl_name in self.host_candidate_tables:
            if tbl_name == "consolidated":
                continue
            print(tbl_name)
            cand_tbl = self.host_candidate_tables[tbl_name]
            if path_cat is None:
                path_cat = cand_tbl["label", "ra", "dec", "separation"]
            path_cat, matched, dist = astm.match_catalogs(
                cat_1=path_cat, cat_2=cand_tbl,
                ra_col_1="ra", dec_col_1="dec",
                keep_non_matches=True,
                tolerance=0.7 * units.arcsec
            )

            for prefix in ["label", "P_Ox", "mag"]:

                if f"{prefix}_{tbl_name}" not in matched.colnames:
                    print(f"{prefix}_{tbl_name}")
                    matched[f"{prefix}_{tbl_name}"] = matched[prefix]

                for col in list(filter(lambda c: c.startswith(prefix + "_"), path_cat.colnames)):
                    matched[col] = np.ones(len(matched)) * -999.

                path_cat[f"{prefix}_{tbl_name}"] = np.ones(len(path_cat)) * -999.
                path_cat[f"{prefix}_{tbl_name}"][path_cat["matched"]] = matched[f"{prefix}_{tbl_name}"][
                    matched["matched"]]

            for row in matched[np.invert(matched["matched"])]:
                print(f'Adding label {row["label"]} from {tbl_name} table. ra={row["ra"]}, dec={row["dec"]}')
                path_cat.add_row(row[path_cat.colnames])

        # path_cat["coord"] = SkyCoord(path_cat["ra"], path_cat["dec"])
        path_cat.sort(sort_by, reverse=True)
        path_cat["id"] = np.zeros(len(path_cat), dtype=str)
        for i, row in enumerate(path_cat):
            row["id"] = chr(65 + i)
        self.host_candidate_tables["consolidated"] = path_cat
        for row in path_cat:
            idn = self.name.replace("FRB", "")
            host_candidate = Galaxy(
                z=None,
                position=SkyCoord(row["ra"], row["dec"]),
                field=self.field,
                name=f"HC{row['id']}_{idn}"
            )
            self.host_candidates.append(host_candidate)
        self.update_output_file()
        return path_cat

    def write_candidate_tables(self):
        table_paths = {}
        for img_name in self.host_candidate_tables:
            cand_tbl = self.host_candidate_tables[img_name]
            write_path = os.path.join(self.data_path, f"PATH_table_{img_name}.ecsv")
            if "coords" in cand_tbl.colnames:
                cand_tbl.remove_column("coords")
            cand_tbl.write(write_path, overwrite=True)
            table_paths[img_name] = p.split_data_dir(write_path)
        return table_paths

    def _output_dict(self):
        output = super()._output_dict()
        cand_list = []
        for obj in self.host_candidates:
            new_dict = Galaxy.default_params()
            new_dict.update({
                "name": obj.name,
                "position": obj.position,
                "z": obj.z,
            })
            cand_list.append(new_dict)

        output.update({
            "host_candidate_tables": self.write_candidate_tables(),
            "host_candidates": cand_list
        })
        return output

    def load_output_file(self):
        outputs = super().load_output_file()
        if outputs is not None:
            if "host_candidate_tables" in outputs:
                tables = outputs["host_candidate_tables"]
                for table_name in tables:
                    tbl_path = tables[table_name]
                    tbl_path = p.join_data_dir(tbl_path)
                    if os.path.isfile(tbl_path):
                        try:
                            tbl = table.QTable.read(tbl_path)
                            self.host_candidate_tables[table_name] = tbl
                        except StopIteration:
                            continue

            if "host_candidates" in outputs:
                for obj in outputs["host_candidates"]:
                    self.host_candidates.append(
                        Galaxy(
                            z=obj["z"],
                            position=obj["position"],
                            name=obj["name"],
                            field=self.field
                        )
                    )

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        default_params.update({
            "host_galaxy": Galaxy.default_params(),
            "date": "0000-01-01",
            "tns_name": None
        })
        return default_params

    @classmethod
    def _date_from_name(cls, name):
        if name.startswith("FRB"):
            name = name
            name.replace(" ", "")
            date_str = name[3:]
            while date_str[-1].isalpha():
                # Get rid of TNS-style trailing letters
                date_str = date_str[:-1]
            if len(name) == 9:
                # Then presumably we have format FRBYYDDMM
                date_str = "20" + date_str
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            return date_str

        else:
            print("Date could not be resolved from object name.")
            return None

    def date_from_name(self):
        date_str = self._date_from_name(self.name)
        try:
            date = time.Time(date_str)
            self.date = date
            return date
        except ValueError:
            return date_str

    def dm_mw_ism_ne2001(self, distance: Union[units.Quantity, float] = 100. * units.kpc):
        """
        Borrowed from frb.mw
        :param distance:
        :return:
        """
        # from frb.mw import ismDM

        from ne2001 import density
        distance = u.dequantify(distance, unit=units.kpc)
        ne = density.ElectronDensity()
        dm_ism = ne.DM(
            self.position.galactic.l.value,
            self.position.galactic.b.value,
            distance
        )
        return dm_ism

    def dm_mw_ism_ymw16(self, distance: Union[units.Quantity, float] = 50. * units.kpc):
        import pygedm
        dm, tau = pygedm.dist_to_dm(
            self.position.galactic.l,
            self.position.galactic.b,
            distance,
            method="ymw16"
        )
        return dm

    def dm_mw_ism(
            self,
            ism_model: str = "ne2001",
            distance: Union[units.Quantity, float] = 100. * units.kpc,
    ):
        ism_model = ism_model.lower()
        if ism_model == "ne2001":
            func = self.dm_mw_ism_ne2001
        elif ism_model == "ymw16":
            func = self.dm_mw_ism_ymw16
        else:
            raise ValueError(f"Model {ism_model} not recognised.")
        dm_ism = func(distance=distance)
        return dm_ism

    def dm_mw_ism_cum(
            self,
            max_distance: units.Quantity = 20. * units.kpc,
            step_size: units.Quantity = 0.1 * units.kpc,
            max_dm: units.Quantity = 30. * dm_units,
            model: str = "ne2001"
    ):
        max_dm = u.check_quantity(max_dm, dm_units)
        i = 0 * units.kpc
        dm_this = 0 * dm_units
        dm = []
        d = []

        if model == "ne2001":
            func = self.dm_mw_ism_ne2001
        elif model == "ymw16":
            func = self.dm_mw_ism_ymw16
        else:
            raise ValueError(f"Model {model} not recognised.")

        while i < max_distance and dm_this < max_dm:
            d.append(i * 1.)
            dm_this = func(distance=i)
            dm.append(dm_this)
            i += step_size

        return table.QTable({
            "DM": dm,
            "d": d
        })

    def dm_mw_halo(
            self,
            model: Union[str, "frb.halos.models.ModifiedNFW"] = "all",
            **kwargs,
    ):
        import frb.halos.models as halos
        # from frb.mw import haloDM

        if isinstance(model, str):
            model = model.lower()
            halo_models = {
                "yf17": halos.YF17,
                "pz19": halos.MilkyWay,
                "mb15": halos.MB15
            }
            if model == "all":
                outputs = {}
                for halo_model in halo_models:
                    outputs[f"dm_halo_mw_{halo_model}"] = self._dm_mw_halo(
                        halo_model=halo_models[halo_model](),
                        **kwargs
                    )
                return outputs
            elif model not in halo_models:
                raise ValueError(f"Supported halo models are {list(halo_models.keys())}, not {model}")
            else:
                return self._dm_mw_halo(
                    halo_model=halo_models[model](),
                    **kwargs
                )
        else:
            return self._dm_mw_halo(
                halo_model=model,
                **kwargs
            )

    def _dm_mw_halo(
            self,
            halo_model=None,
            distance: Union[units.Quantity, float] = 1.,
            zero_distance: units.Quantity = 10 * units.kpc,
    ):
        """

        :param distance: Distance from MW centre to which to evaluate DM. If a non-Quantity number is passed, it will be
            interpreted as a multiple of the model's virial radius (R200).
        :param zero_distance: The distance to which to zero the inner volume of the halo
        :param halo_model: Halo model to evaluate.
        :return:
        """

        from ne2001 import density
        if halo_model is None:
            from frb.halos.models import MilkyWay
            halo_model = MilkyWay()
        if not isinstance(distance, units.Quantity):
            distance = halo_model.r200 * distance
        u.dequantify(distance, units.kpc)
        # Zero out inner volume
        zero_distance = zero_distance.to(units.kpc)
        halo_model.zero_inner_ne = zero_distance.value  # kpc
        params = dict(F=1., e_density=1.)
        model_ne = density.NEobject(halo_model.ne, **params)
        dm_halo = model_ne.DM(
            self.position.galactic.l.value,
            self.position.galactic.b.value,
            distance
        )
        return dm_halo

    def dm_mw_halo_cum(
            self,
            rmax: float = 1.,
            step_size: units.Quantity = 1 * units.kpc,
            halo_model: "frb.halos.models.ModifiedNFW" = None,
            **kwargs
    ):
        if halo_model is None:
            from frb.halos.models import MilkyWay
            halo_model = MilkyWay()
        max_distance = rmax * halo_model.r200
        i = 0 * units.kpc
        dm = []
        d = []
        while i < max_distance:
            d.append(i * 1.)
            dm_this = self._dm_mw_halo(halo_model=halo_model, distance=i, **kwargs)
            dm.append(dm_this)
            i += step_size

        return table.QTable({
            "DM": dm,
            "d": d
        })

    def dm_exgal(
            self,
            ism_model: str = "ne2001",
            halo_model: str = "pz19"
    ):
        if halo_model == "all":
            raise ValueError("Please specify a single halo model.")
        dm_mw = self.dm_mw_halo(model=halo_model) + self.dm_mw_ism(ism_model=ism_model)
        return self.dm - dm_mw

    def dm_cosmic(
            self,
            z_max: float = None,
            **kwargs
    ):
        from frb.dm.igm import average_DM
        if z_max is None:
            z_max = self.host_galaxy.z
        return average_DM(z_max, cosmo=cosmology, **kwargs)

    def dm_halos_avg(self, z_max: float = None, **kwargs):
        import frb.halos.hmf as hmf
        from frb.dm.igm import average_DMhalos
        hmf.init_hmf()
        if z_max is None:
            z_max = self.host_galaxy.z
        return average_DMhalos(z_max, cosmo=cosmology, **kwargs)

    # def estimate_dm_excess(self):
    #     dm_ism = self.estimate_dm_mw_ism()
    #     dm_cosmic = self.estimate_dm_cosmic()
    #     dm_halo = 60 * dm_units
    #     return self.dm - dm_ism - dm_cosmic - dm_halo

    def z_from_dm(
            self,
            dm_host: units.Quantity = 0,
            **kwargs
    ):
        from frb.dm.igm import z_from_DM
        dm = self.dm_exgal(**kwargs) - dm_host
        return z_from_DM(
            DM=dm,
            coord=None,
            cosmo=cosmology,
            corr_nuisance=False
        )

    def foreground_accounting(
            self,
            rmax=1.,
            cat_search: str = None,
            step_size_halo: units.Quantity = 0.1 * units.kpc,
            neval_cosmic: int = 10000,
            foreground_objects: list = None,
            load_objects: bool = True,
            skip_other_models: bool = False
    ):

        from frb.halos.hmf import halo_incidence

        outputs = self.dm_mw_halo(distance=rmax)

        if load_objects:
            self.field.load_all_objects()

        host = self.host_galaxy
        if foreground_objects is None:
            foreground_objects = list(
                filter(
                    lambda o: isinstance(o, Galaxy) and o.z <= self.host_galaxy.z and o.mass_stellar is not None,
                    self.field.objects
                )
            )
        if host not in foreground_objects and host.z is not None:
            foreground_objects.append(host)

        foreground_zs = list(map(lambda o: o.z, foreground_objects))

        frb_err_ra, frb_err_dec = self.position_err.uncertainty_quadrature_equ()
        frb_err_dec = frb_err_dec.to(units.arcsec)
        cosmic_tbl = table.QTable()

        print("DM_FRB:")
        print(self.dm)

        print("DM_MW:")

        print("\tDM_MWISM:")
        # outputs["dm_ism_mw_cum"] = self.estimate_dm_mw_ism_cum(max_dm=outputs["dm_ism_mw_ne2001"] - 0.5 * dm_units)
        print("\t\tDM_MWISM_NE2001")
        outputs["dm_ism_mw_ne2001"] = self.dm_mw_ism_ne2001()
        print("\t\t", outputs["dm_ism_mw_ne2001"])

        print("\t\tDM_MWISM_YMW16")
        outputs["dm_ism_mw_ymw16"] = self.dm_mw_ism_ymw16()
        print("\t\t", outputs["dm_ism_mw_ymw16"])

        print("\tDM_MWHalo_PZ19:")
        print("\t", outputs["dm_halo_mw_pz19"])

        if not skip_other_models:
            print("\tDM_MWHalo_YF17:")
            print("\t", outputs["dm_halo_mw_yf17"])

            print("\tDM_MWHalo_MB15:")
            print("\t", outputs["dm_halo_mw_mb15"])

        print("\tDM_MW:")
        outputs["dm_mw"] = outputs["dm_halo_mw_pz19"] + outputs["dm_ism_mw_ne2001"]
        print("\t", outputs["dm_mw"])

        print("DM_exgal:")
        outputs["dm_exgal"] = self.dm - outputs["dm_mw"]
        print(outputs["dm_exgal"])

        print("Avg DM_cosmic:")
        if host.z is not None:
            z_max = host.z
        else:
            z_max = 2.0
        dm_cosmic, z = self.dm_cosmic(cumul=True, neval=neval_cosmic, z_max=z_max)
        cosmic_tbl["z"] = z
        cosmic_tbl["comoving_distance"] = cosmology.comoving_distance(cosmic_tbl["z"])
        cosmic_tbl["dm_cosmic_avg"] = dm_cosmic
        outputs["dm_cosmic_avg"] = dm_cosmic[-1]
        print(outputs["dm_cosmic_avg"])
        print("\tAvg DM_halos:")
        dm_halos, _ = self.dm_halos_avg(rmax=rmax, neval=neval_cosmic, cumul=True, z_max=z_max)
        cosmic_tbl["dm_halos_avg"] = dm_halos
        outputs["dm_halos_avg"] = dm_halos[-1]
        print("\t", outputs["dm_halos_avg"])
        print("\tAvg DM_igm:")
        outputs["dm_igm"] = outputs["dm_cosmic_avg"] - outputs["dm_halos_avg"]
        cosmic_tbl["dm_igm"] = cosmic_tbl["dm_cosmic_avg"] - cosmic_tbl["dm_halos_avg"]
        print("\t", outputs["dm_igm"])

        print("Empirical DM_halos:")
        halo_inform = []
        halo_models = {}
        halo_profiles = {}
        dm_halo_host = 0. * dm_units
        dm_halo_cum = {}
        cosmic_tbl["dm_halos_emp"] = cosmic_tbl["dm_halos_avg"] * 0
        for obj in foreground_objects:
            print(f"\tDM_halo_{obj.name}: ({obj.z=})")
            # if load_objects:
            #     obj.load_output_file()

            obj.select_deepest()

            if obj.position_photometry is not None:
                pos = obj.position_photometry
            else:
                pos = obj.position

            halo_info = {
                "id": obj.name,
                "z": obj.z,
                "ra": pos.ra,
                "dec": pos.dec
            }

            if cat_search is not None:
                cat_row, sep = obj.find_in_cat(cat_search)
                if sep < 1 * units.arcsec:
                    halo_info["id_cat"] = cat_row["objName"]
                    halo_info["ra_cat"] = cat_row["raStack"]
                    halo_info["dec_cat"] = cat_row["decStack"]
                else:
                    halo_info["id_cat"] = "--"
                halo_info["offset_cat"] = sep.to(units.arcsec)

            halo_info["offset_angle"] = offset_angle = self.position.separation(obj.position_photometry).to(
                units.arcsec)
            if obj.position_photometry_err.dec_stat is not None:
                fg_pos_err = max(
                    obj.position_photometry_err.dec_stat,
                    obj.position_photometry_err.ra_stat)
            else:
                fg_pos_err = 0 * units.arcsec
            halo_info["distance_angular_size"] = obj.angular_size_distance()
            halo_info["distance_luminosity"] = obj.luminosity_distance()
            halo_info["distance_comoving"] = obj.comoving_distance()
            halo_info["offset_angle_err"] = offset_angle_err = np.sqrt(fg_pos_err ** 2 + frb_err_dec ** 2)
            halo_info["r_perp"] = offset = obj.projected_size(offset_angle).to(units.kpc)
            halo_info["r_perp_err"] = obj.projected_size(offset_angle_err).to(units.kpc)
            halo_info["mass_stellar"] = fg_m_star = obj.mass_stellar
            halo_info["mass_stellar_err_plus"] = fg_m_star_err_plus = obj.mass_stellar_err_plus
            halo_info["mass_stellar_err_minus"] = fg_m_star_err_minus = obj.mass_stellar_err_minus
            halo_info["log_mass_stellar"] = np.log10(fg_m_star / units.solMass)
            halo_info["log_mass_stellar_err_plus"] = u.uncertainty_log10(
                arg=fg_m_star,
                uncertainty_arg=fg_m_star_err_plus
            )
            halo_info["log_mass_stellar_err_minus"] = u.uncertainty_log10(
                arg=fg_m_star,
                uncertainty_arg=fg_m_star_err_minus
            )
            obj.halo_mass()
            halo_info["mass_halo"] = obj.mass_halo
            halo_info["log_mass_halo"] = obj.log_mass_halo
            halo_info["log_mass_halo_upper"] = obj.log_mass_halo_upper
            halo_info["log_mass_halo_lower"] = obj.log_mass_halo_lower
            halo_info["h"] = obj.h()
            halo_info["c"] = obj.halo_concentration_parameter()

            mnfw = obj.halo_model_mnfw()
            yf17 = obj.halo_model_yf17()
            mb04 = obj.halo_model_mb04()
            mb15 = obj.halo_model_mb15()

            halo_nes = []
            rs = []
            dm_val = np.inf
            i = 0
            # r_perp = 0 * units.kpc
            while dm_val > rmax:  # 1.
                r_perp = i * step_size_halo
                dm_val = mnfw.Ne_Rperp(
                    r_perp,
                    rmax=rmax,
                    step_size=step_size_halo
                ).value / (1 + obj.z)
                halo_nes.append(dm_val)
                rs.append(r_perp.value)
                i += 1

            # halo_info["r_lim"] = r_perp

            halo_info["dm_halo"] = dm_halo = mnfw.Ne_Rperp(
                Rperp=offset,
                rmax=rmax,
                step_size=step_size_halo
            ) / (1 + obj.z)
            halo_info["dm_halo_yf17"] = yf17.Ne_Rperp(
                Rperp=offset,
                rmax=rmax,
                step_size=step_size_halo
            ) / (1 + obj.z)
            halo_info["dm_halo_mb04"] = mb04.Ne_Rperp(
                Rperp=offset,
                rmax=rmax,
                step_size=step_size_halo
            ) / (1 + obj.z)
            halo_info["dm_halo_mb15"] = mb15.Ne_Rperp(
                Rperp=offset,
                rmax=rmax,
                step_size=step_size_halo
            ) / (1 + obj.z)

            halo_info["r_200"] = mnfw.r200
            if obj.name.startswith("HG"):
                halo_info["dm_halo"] = dm_halo_host = dm_halo / 2
                halo_info["id_short"] = "HG"
            else:
                halo_info["id_short"] = obj.name[:obj.name.find("_")]

            print("\t", halo_info["dm_halo"])

            halo_models[obj.name] = mnfw

            halo_inform.append(halo_info)

            halo_profiles[obj.name] = halo_nes

            if host.z is not None:
                halo_info["n_intersect_greater"] = halo_incidence(
                    Mlow=obj.mass_halo.value,
                    zFRB=host.z,
                    radius=halo_info["r_perp"]
                )

            m_low = 10 ** (np.floor(obj.log_mass_halo))
            m_high = 10 ** (np.ceil(obj.log_mass_halo))
            if m_low < 2e10:
                m_high += 2e10 - m_low
                m_low = 2e10

            halo_info["mass_halo_partition_high"] = m_high * units.solMass
            halo_info["mass_halo_partition_low"] = m_low * units.solMass

            halo_info["log_mass_halo_partition_high"] = np.log10(m_high)
            halo_info["log_mass_halo_partition_low"] = np.log10(m_low)

            if host.z is not None:
                halo_info["n_intersect_partition"] = halo_incidence(
                    Mlow=m_low,
                    Mhigh=m_high,
                    zFRB=host.z,
                    radius=halo_info["r_perp"]
                )

            if obj.cigale_results is not None:
                halo_info["u-r"] = obj.cigale_results["bayes.param.restframe_u_prime-r_prime"] * units.mag
            if obj.sfr is not None:
                halo_info["sfr"] = obj.sfr
            if obj.sfr_err is not None:
                halo_info["sfr_err"] = obj.sfr_err

            if halo_info["dm_halo"] > 0. * dm_units:
                dm_halo_cum_this = obj.halo_dm_cum(
                    rmax=rmax,
                    rperp=offset,
                    step_size=step_size_halo
                )

                if obj is not host:
                    dm_halo_cum[obj.name] = dm_halo_cum_this
                    cosmic_tbl["dm_halos_emp"] += np.interp(
                        cosmic_tbl["comoving_distance"],
                        dm_halo_cum_this["d_abs"],
                        dm_halo_cum_this["DM"]
                    )
                # Add a cumulative halo host DM to the cosmic table
                else:
                    dm_halo_cum[obj.name] = dm_halo_cum_this[:len(dm_halo_cum_this) // 2]
                    cosmic_tbl["dm_halo_host"] = np.interp(
                        cosmic_tbl["comoving_distance"],
                        dm_halo_cum_this["d_abs"],
                        dm_halo_cum_this["DM"]
                    )

        #         plt.plot(rs, halo_nes)
        #         plt.plot([offset.value, offset.value], [0, max(halo_nes)])

        halo_tbl = table.QTable(halo_inform)
        halo_tbl["dm_halo"] = halo_tbl["dm_halo"].to(dm_units)
        cosmic_tbl["dm_cosmic_emp"] = cosmic_tbl["dm_halos_emp"] + cosmic_tbl["dm_igm"]

        print("\tEmpirical DM_halos:")
        outputs["dm_halos_emp"] = halo_tbl["dm_halo"].nansum() - dm_halo_host
        outputs["dm_halos_yf17"] = halo_tbl["dm_halo_yf17"].nansum() - dm_halo_host
        outputs["dm_halos_mb04"] = halo_tbl["dm_halo_mb04"].nansum() - dm_halo_host
        outputs["dm_halos_mb15"] = halo_tbl["dm_halo_mb15"].nansum() - dm_halo_host

        print("\t", outputs["dm_halos_emp"])

        print("\tEmpirical DM_cosmic:")
        outputs["dm_cosmic_emp"] = outputs["dm_igm"] + outputs["dm_halos_emp"]
        print("\t", outputs["dm_cosmic_emp"])

        outputs["dm_host_median"] = 0 * dm_units
        if host.z is not None:
            print("DM_host:")
            # Obtained using James 2021
            print("\tMedian DM_host:")
            outputs["dm_host_median_james22A"] = dm_host_median["james_22A"] / (1 + host.z)
            outputs["dm_host_median"] = dm_host_median["james_22B"] / (1 + host.z)
            print("\t", outputs["dm_host_median"])
            # print("\tMax-probability DM_host:")
            # outputs["dm_host_max_p_james22A"] = 98 * dm_units / (1 + host.z)
            # print("\t", outputs["dm_host_max_p"])

            print("\tDM_halo_host")
            outputs["dm_halo_host"] = dm_halo_host
            print("\t", outputs["dm_halo_host"])

            print("\tDM_host,ism:")
            outputs["dm_host_ism"] = outputs["dm_host_median"] - outputs["dm_halo_host"]
            print("\t", outputs["dm_host_ism"])

        print("Excess DM estimate:")
        outputs["dm_excess_avg"] = self.dm - outputs["dm_cosmic_avg"] - outputs["dm_mw"] - outputs["dm_host_median"]
        print("\t", outputs["dm_excess_avg"])

        print("Empirical Excess DM:")
        outputs["dm_excess_emp"] = self.dm - outputs["dm_cosmic_emp"] - outputs["dm_mw"] - outputs["dm_host_median"]
        print("\t", outputs["dm_excess_emp"])

        #     r_eff_proj = foreground.projected_distance(r_eff).to(units.kpc)
        #     r_eff_proj_err = foreground.projected_distance(r_eff_err).to(units.kpc)
        #     print("Projected effective radius:")
        #     print(r_eff_proj, "+/-", r_eff_proj_err)
        #     print("FG-normalized offset:")
        #     print(offset / r_eff_proj)

        outputs["halo_table"] = halo_tbl
        outputs["halo_models"] = halo_models
        outputs["halo_dm_profiles"] = halo_profiles
        outputs["halo_dm_cum"] = dm_halo_cum
        outputs["dm_cum_table"] = cosmic_tbl

        return outputs

    @classmethod
    def from_dict(cls, dictionary: dict, name: str = None, field=None):
        frb = super().from_dict(dictionary=dictionary)
        # if "dm" in dictionary:
        #     frb.dm = u.check_quantity(dictionary["dm"], dm_units)
        dictionary["host_galaxy"]["transient"] = frb
        host_galaxy = TransientHostCandidate.from_dict(dictionary=dictionary["host_galaxy"], field=field)
        frb.host_galaxy = host_galaxy
        return frb

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        default_params.update({
            "dm": 0.0 * dm_units,
            "snr": 0.0,
        })
        return default_params
