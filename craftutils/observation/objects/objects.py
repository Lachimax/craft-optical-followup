from typing import Union, List, Callable
import os
import copy

import matplotlib.pyplot as plt
import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as units
import astropy.table as table
from astropy.modeling import models, fitting
from astropy.visualization import quantity_support
import astropy.time as time

import craftutils.params as p
import craftutils.astrometry as astm
import craftutils.utils as u
import craftutils.retrieve as r
import craftutils.plotting as pl
import craftutils.observation.instrument as inst
import craftutils.observation.filters as filters

from craftutils.observation.generic import Generic
from .position import PositionUncertainty, position_dictionary

quantity_support()

object_index = {}


def object_from_index(
        name: str,
        tolerate_missing: bool = False
):
    if name in object_index:
        return object_index[name]
    elif tolerate_missing:
        return None
    else:
        raise ValueError(f"Object with name {name} is not found in object_index.")


def object_to_index(
        obj: 'Object',
        allow_overwrite: bool = False
):
    # print(object_index)
    u.debug_print(1, f"Adding {str(type(obj))} {obj.name} to object index.")
    if not isinstance(obj, Object):
        raise TypeError(f"obj {obj} is not an Object.")
    name = obj.name
    if not allow_overwrite and name in object_index:
        raise ValueError(f"Object with name {name} already exists in object_index.")
    object_index[name] = obj
    return obj


@u.export
class Object(Generic):
    optical = False
    radio = False

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
        super().__init__(
            **kwargs
        )
        self.field = field
        self.name = None
        self.name_filesys = None
        self.cat_row = None
        self.position = None
        self.position_err = None
        if position is not None:
            self.position = astm.attempt_skycoord(position)
            if type(position_err) is not PositionUncertainty:
                self.position_err = PositionUncertainty(uncertainty=position_err, position=self.position)
            self.position_galactic = None
            if isinstance(self.position, SkyCoord):
                self.position_galactic = self.position.transform_to("galactic")

        self.set_name(name=name)

        if self.name:
            object_to_index(self, allow_overwrite=True)

        if self.cat_row is not None:
            self.position_from_cat_row()

        self.position_photometry = copy.deepcopy(self.position)
        self.position_photometry_err = copy.deepcopy(self.position_err)

        self.set_name_filesys()

        self.photometry = {}
        self.photometry_tbl_best = None
        self.photometry_tbl = None
        self.data_path = None
        self.output_file = None
        self.param_path = None
        if "param_path" in kwargs:
            self.param_path = kwargs["param_path"]
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
                kwargs["photometry_args_manual"]["b"] is not None:
            self.photometry_args = kwargs["photometry_args_manual"]
            self.a = self.photometry_args["a"]
            self.b = self.photometry_args["b"]
            self.theta = self.photometry_args["theta"]
            self.kron = self.photometry_args["kron_radius"]

        self.other_names: List[str] = []
        if "other_names" in kwargs:
            self.other_names = kwargs["other_names"]

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{str(type(self))} {self.name} at {self.position.to_string('hmsdms')}"

    def set_name(self, name: str = None):
        if name is None:
            name = self.jname()
        self.name = name
        self.set_name_filesys()
        return self.name

    def _get_object(
            self,
            obj_name: str,
            tolerate_missing: bool = True,
    ):
        if self.field is not None and obj_name in self.field.objects:
            return self.field.objects[obj_name]
        else:
            return object_from_index(obj_name, tolerate_missing=tolerate_missing)

    def set_name_filesys(self):
        if self.name is None and self.position is not None:
            self.name = self.jname()
        if not isinstance(self.name, str):
            self.name = str(self.name)
        self.name_filesys = self.name.replace(" ", "-")
        self.check_data_path(make=False)

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

    def surface_brightness_at_position(self, img):
        x, y = img.world_to_pixel(coord=self.position)
        x = int(x)
        y = int(y)
        pixels, err = img.surface_brightness()
        p_xy = pixels[y, x]
        err_xy = err[y, x]
        ext = self.galactic_extinction(fil=img.filter) / units.arcsec ** 2
        return p_xy - ext, err_xy

    def get_good_photometry(self):

        import craftutils.observation.image as image
        self.apply_galactic_extinction()
        deepest_dict = self.select_deepest()
        deepest_path = deepest_dict["good_image_path"]

        print("Deepest image determined to be", deepest_path)

        cls = image.CoaddedImage.select_child_class(instrument_name=deepest_dict["instrument"])
        deepest_img = cls(path=deepest_path)
        deep_mask, objs = deepest_img.write_mask(
            output_path=os.path.join(
                self.data_path,
                f"{self.name_filesys}_master-mask_{deepest_dict['instrument']}_{deepest_dict['filter']}_{deepest_dict['epoch_name']}.fits",
            ),
            method="sep",
            do_not_mask=self.position_photometry
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

        from .transient_host import TransientHostCandidate
        if isinstance(self, TransientHostCandidate):
            sb, sb_err = self.transient.surface_brightness_at_position(img=deepest_img)
            deepest_dict["transient_position_surface_brightness"] = float(sb.value) * units.mag / units.arcsec ** 2
            deepest_dict["transient_position_surface_brightness_err"] = float(sb_err.value) * units.mag / units.arcsec ** 2

        if mag_results is not None:
            deepest_dict["mag_sep"] = mag_results["mag"][0]
            deepest_dict["mag_sep_err"] = mag_results["mag_err"][0]
            deepest_dict["snr_sep"] = mag_results["snr"][0]
            deepest_dict["back_sep"] = float(mag_results["back"][0])
            deepest_dict["flux_sep"] = float(mag_results["flux"][0])
            deepest_dict["flux_sep_err"] = float(mag_results["flux_err"][0])
            deepest_dict["limit_threshold"] = float(mag_results["threshold"])
            deepest_dict["peak"] = float(mag_results["peak"])
        else:
            deepest_dict["mag_sep"] = -999. * units.mag
            deepest_dict["mag_sep_err"] = -999. * units.mag
            deepest_dict["snr_sep"] = -999.
            deepest_dict["back_sep"] = 0.
            deepest_dict["flux_sep"] = -999.
            deepest_dict["flux_sep_err"] = -999.
            deepest_dict["threshold_sep"] = -999.
            deepest_dict["limit_threshold"] = -999.
            deepest_dict["peak"] = - 999.
        deepest_dict["zeropoint_sep"] = deepest_img.zeropoint_best["zeropoint_img"]

        for instrument in self.photometry:
            for band in self.photometry[instrument]:
                for epoch in self.photometry[instrument][band]:
                    print(f"Extracting photometry for {self.name} in {instrument} {band}, epoch {epoch}.")
                    phot_dict = self.photometry[instrument][band][epoch]
                    if phot_dict["good_image_path"] == deepest_path:
                        continue
                    cls = image.CoaddedImage.select_child_class(instrument_name=instrument)
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

                    if isinstance(self, TransientHostCandidate):
                        sb, sb_err = self.transient.surface_brightness_at_position(img=img)
                        phot_dict["transient_position_surface_brightness"] = float(sb.value) * units.mag / units.arcsec ** 2
                        phot_dict["transient_position_surface_brightness_err"] = float(sb_err.value) * units.mag / units.arcsec ** 2

                    if mag_results is not None:
                        phot_dict["mag_sep"] = mag_results["mag"][0]
                        phot_dict["mag_sep_err"] = mag_results["mag_err"][0]
                        phot_dict["snr_sep"] = float(mag_results["snr"][0])
                        phot_dict["back_sep"] = float(mag_results["back"][0])
                        phot_dict["flux_sep"] = float(mag_results["flux"][0])
                        phot_dict["flux_sep_err"] = float(mag_results["flux_err"][0])
                        phot_dict["limit_threshold"] = float(mag_results["threshold"])
                        phot_dict["peak"] = float(mag_results["peak"])
                    else:
                        phot_dict["mag_sep"] = -999. * units.mag
                        phot_dict["mag_sep_err"] = -999. * units.mag
                        phot_dict["snr_sep"] = -999.
                        phot_dict["back_sep"] = 0.
                        phot_dict["flux_sep"] = -999.
                        phot_dict["flux_sep_err"] = -999.
                        phot_dict["threshold_sep"] = -999.
                        phot_dict["limit_threshold"] = -999.
                        phot_dict["peak"] = -999.
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
                        phot_dict["snr_sep_unmasked"] = float(mag_results["snr"][0])
                        phot_dict["flux_sep_unmasked"] = float(mag_results["flux"][0])
                        phot_dict["flux_sep_unmasked_err"] = float(mag_results["flux_err"][0])
                        phot_dict["peak_unmasked"] = float(mag_results["peak"])
                    else:
                        phot_dict["mag_sep_unmasked"] = -999. * units.mag
                        phot_dict["mag_sep_unmasked_err"] = -999. * units.mag
                        phot_dict["snr_sep"] = -999.
                        phot_dict["flux_sep_unmasked"] = -999.
                        phot_dict["flux_sep_unmasked_err"] = -999.
                        phot_dict["peak_unmasked"] = -999.
                    img.close()
                    del img
                    mask_rp.close()
                    del mask_rp

        deepest_img.close()
        del deepest_img
        deep_mask.close()
        del deep_mask

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
        print(image_depth)
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

        output_dict = super()._output_dict()

        pos_phot_err = None
        if self.position_photometry_err is not None:
            pos_phot_err = self.position_photometry_err.to_dict()
        pos_err = None
        if self.position_err is not None:
            pos_err = self.position_err.to_dict()
        output_dict.update({
            "a": self.a,
            "b": self.b,
            "theta": self.theta,
            "kron_radius": self.kron,
            "position_input": self.position,
            "position_input_err": pos_err,
            "position_photometry": self.position_photometry,
            "position_photometry_err": pos_phot_err,
            "photometry": self.photometry,
            "irsa_extinction_path": self.irsa_extinction_path,
            "extinction_law": self.extinction_power_law,
            "ebv_sandf": self.ebv_sandf,
            "jname": self.jname()
        })
        return output_dict

    def load_output_file(self):
        self.check_data_path()
        if self.data_path is not None:
            outputs = super().load_output_file()
            if outputs is not None:
                if "a" in outputs and outputs["a"] is not None:
                    self.a = outputs["a"]
                if "b" in outputs and outputs["b"] is not None:
                    self.b = outputs["b"]
                if "theta" in outputs and outputs["theta"] is not None:
                    self.theta = outputs["theta"]
                if "kron_radius" in outputs and outputs["kron_radius"] is not None:
                    self.kron_radius = outputs["kron_radius"]
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

    def check_data_path(self, make: bool = True):
        if self.field is not None:
            u.debug_print(2, "", self.name)
            if self.name_filesys is None:
                self.set_name_filesys()
            self.data_path = str(os.path.join(self.field.data_path, "objects", self.name_filesys))
            if make:
                u.mkdir_check(self.data_path)
            self.output_file = os.path.join(self.data_path, f"{self.name_filesys}_outputs.yaml")
            return True
        else:
            return False

    def _updateable(self):
        p_dict = super()._updateable()
        p_dict.update({
            "other_names": self.other_names,
        })
        return p_dict

    def update_output_file(self):
        if self.check_data_path():
            super().update_output_file()

    def write_plot_photometry(self, output: str = None, **kwargs):
        """Plots available photometry (mag v lambda_eff) and writes to disk.

        :param output: Path to write plot.
        :return: matplotlib ax object containing plot info
        """

        if not self.photometry:
            self.load_output_file()
            if not self.photometry:
                print(f"No photometry found for {self.name}")
                return

        if output is None:
            output = os.path.join(self.data_path, f"{self.name_filesys}_photometry.pdf")

        plt.close()
        for best in (False, True):
            fig, ax = self.plot_photometry(**kwargs, best=best)
            ax.legend(loc=(1.1, 0.))
            if best:
                output = output.replace(".pdf", "_best.pdf")
            plt.savefig(output, bbox_inches="tight")
            plt.close(fig)
            del fig, ax

        output_n = output.replace("_photometry_best.pdf", "_photometry_time.pdf")

        for ext_corr in (False, True):
            fig, ax = self.plot_photometry_time(
                extinction_corrected=ext_corr,
                **kwargs
            )
            ax.legend(loc=(1.1, 0.))
            if ext_corr:
                output_n = output_n.replace(".pdf", "_gal_ext.pdf")
            plt.savefig(output_n, bbox_inches="tight")
            plt.close(fig)
            del fig, ax

    def plot_photometry_time(
            self,
            ax: plt.Axes = None,
            fig: plt.Figure = None,
            extinction_corrected: bool = True,
            mag_type: str = "sep",
            **kwargs
    ) -> plt.axes:
        if not self.photometry:
            self.load_output_file()
            if not self.photometry:
                print(f"No photometry found for {self.name}")
                return

        if ax is None:
            fig, ax = plt.subplots()
        if "ls" not in kwargs:
            kwargs["ls"] = ""
        if "marker" not in kwargs:
            kwargs["marker"] = "x"
        if "ecolor" not in kwargs:
            kwargs["ecolor"] = "black"

        if extinction_corrected:
            self.apply_galactic_extinction()
            key = f"mag_{mag_type}_ext_corrected"
        else:
            key = f"mag_{mag_type}"
        photometry_tbl = self.photometry_to_table(
            output=self.build_photometry_table_path(best=False),
            fmts=["ascii.ecsv", "ascii.csv"],
            best=False,
        ).copy()

        bands = list(set(photometry_tbl["band"]))
        bands.sort(key=lambda n: n.lower())

        with quantity_support():

            for i, band in enumerate(bands):
                c = pl.colours[i]

                valid = photometry_tbl[photometry_tbl["mag_sep"] > -990 * units.mag]
                valid = valid[list(map(lambda row: "combined" not in row["epoch_name"], valid))]
                valid = valid[valid["epoch_date"] != "None"]
                valid["time_obj"] = time.Time(valid["epoch_date"], format="isot", scale="utc")
                valid["time_obj"] = valid["time_obj"].mjd
                in_band = valid[valid["band"] == band]

                plot_limit = (-999 * units.mag == in_band["mag_sep_err"])
                limits = in_band[plot_limit]
                mags = in_band[np.invert(plot_limit)]

                ax.errorbar(
                    mags["time_obj"],
                    mags[key],
                    yerr=mags["mag_sep_err"],
                    label=f"{band}",
                    color=c,
                    **kwargs,
                )
                ax.scatter(
                    limits["time_obj"],
                    limits[key],
                )
                ax.scatter(
                    limits["time_obj"],
                    limits[key],
                    marker="v",
                    color=c,
                )

                ax.set_ylabel("Apparent magnitude")
                ax.set_xlabel("MJD")
                ax.invert_yaxis()
        return fig, ax

    def plot_photometry(
            self,
            ax: plt.axes = None,
            best: bool = False,
            **kwargs
    ) -> plt.axes:
        """
        Plots available photometry (mag v lambda_eff).
        :param ax: matplotlib ax object to plot with. A new object is generated if none is provided.
        :param kwargs:
        :return: matplotlib ax object containing plot info
        """

        if not self.photometry:
            self.load_output_file()
            if not self.photometry:
                print(f"No photometry found for {self.name}")
                return

        if ax is None:
            fig, ax = plt.subplots()
        if "ls" not in kwargs:
            kwargs["ls"] = ""
        if "marker" not in kwargs:
            kwargs["marker"] = "x"
        if "ecolor" not in kwargs:
            kwargs["ecolor"] = "black"

        self.apply_galactic_extinction()
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
                label="Measurement",
                **kwargs,
            )
            ax.scatter(
                limits["lambda_eff"],
                limits["mag_sep"],
                label="Upper limit",
                marker="v",
            )
            ax.scatter(
                mags["lambda_eff"],
                mags["mag_sep_ext_corrected"],
                color="orange",
                label="Extinction-corrected measurement"
            )
            ax.scatter(
                limits["lambda_eff"],
                limits["mag_sep_ext_corrected"],
                label="Extinction-corrected upper limit",
                marker="v",
            )
            ax.set_ylabel("Apparent magnitude")
            ax.set_xlabel("$\lambda_\mathrm{eff}$ (\AA)")
            ax.invert_yaxis()
        return fig, ax

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

        if not self.photometry:
            self.load_output_file()
            if not self.photometry:
                print(f"No photometry found for {self.name}")
                return

        if output is None:
            output = self.build_photometry_table_path()

        tbls = []
        for instrument_name in self.photometry:
            instrument = inst.Instrument.from_params(instrument_name)
            for filter_name in self.photometry[instrument_name]:
                fil = instrument.filters[filter_name]

                if best:
                    phot_dict, _ = self.select_photometry_sep(fil=filter_name, instrument=instrument_name)
                    if phot_dict is None:
                        return None
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
                # u.detect_problem_row(photometry_tbl, fmt="csv")
                photometry_tbl.write(output.replace(".ecsv", fmt[fmt.find("."):]), format=fmt, overwrite=True)
        return photometry_tbl

    # def sandf_galactic_extinction(
    #     self,
    #     band: List[filters.Filter]
    # ):
    #     import extinction
    #     extinction.fitzpatrick99(tbl["lambda_eff"], a_v, r_v) * units.mag
    #     pass

    def galactic_extinction(
            self,
            fil: filters.Filter,
            r_v: float = 3.1,
            spectrum: table.QTable = None,
            extinction_law: Callable = None,
    ):
        self.retrieve_extinction_table()
        return fil.galactic_extinction(
            e_bv=self.ebv_sandf,
            r_v=r_v,
            spectrum=spectrum,
            extinction_law=extinction_law
        )

    def apply_galactic_extinction(
            self,
            # ax=None,
            r_v: float = 3.1,
            **kwargs
    ):

        # if ax is None:
        #     fig, ax = plt.subplots()
        # if "marker" not in kwargs:
        #     kwargs["marker"] = "x"

        self.retrieve_extinction_table()

        tbl = self.photometry_to_table(fmts=["ascii.ecsv", "ascii.csv"])

        # x = np.linspace(0, 80000, 1000) * units.Angstrom

        # lambda_eff_tbl = self.irsa_extinction["LamEff"].to(units.Angstrom)
        # power_law = models.PowerLaw1D()
        # fitter = fitting.LevMarLSQFitter()
        # try:
        #     fitted = fitter(power_law, lambda_eff_tbl, self.irsa_extinction["A_SandF"].value)
        #     tbl["ext_gal_pl"] = fitted(tbl["lambda_eff"]) * units.mag
        #     ax.plot(
        #         x, fitted(x),
        #         label=f"power law fit to IRSA",
        #         # , \\alpha={fitted.alpha.value}; $x_0$={fitted.x_0.value}; A={fitted.amplitude.value}",
        #         c="blue"
        #     )
        #     self.extinction_power_law = {
        #         "amplitude": fitted.amplitude.value * fitted.amplitude.unit,
        #         "x_0": fitted.x_0.value,
        #         "alpha": fitted.alpha.value
        #     }
        # except fitting.NonFiniteValueError:
        #     fitted = None
        #     tbl["ext_gal_pl"] = -999. * units.mag

        if not self.photometry:
            self.load_output_file()
            if not self.photometry:
                print(f"No photometry found for {self.name}")
                return

        exts = []
        fils = []
        for row in tbl:
            fil = filters.Filter.from_params(row["filter"], row["instrument"])
            fils.append(fil)
            exts.append(self.galactic_extinction(fil=fil, r_v=r_v))

        tbl["ext_gal_sandf"] = exts

        # tbl["ext_gal_interp"] = np.interp(
        #     tbl["lambda_eff"],
        #     lambda_eff_tbl,
        #     self.irsa_extinction["A_SandF"].value
        # ) * units.mag
        #
        # ax.plot(
        #     x, self.galactic_extinction(x, r_v=r_v).value,
        #     label="S\&F + F99 extinction law",
        #     c="red"
        # )
        # ax.scatter(
        #     lambda_eff_tbl, self.irsa_extinction["A_SandF"].value,
        #     label="from IRSA",
        #     c="green",
        #     **kwargs)
        # ax.scatter(
        #     tbl["lambda_eff"], tbl["ext_gal_pl"].value,
        #     label="power law interpolation of IRSA",
        #     c="blue",
        #     **kwargs
        # )
        # ax.scatter(
        #     tbl["lambda_eff"], tbl["ext_gal_interp"].value,
        #     label="numpy interpolation from IRSA",
        #     c="violet",
        #     **kwargs
        # )
        # ax.scatter(
        #     tbl["lambda_eff"], tbl["ext_gal_sandf"].value,
        #     label="S\&F + F99 extinction law",
        #     c="red",
        #     **kwargs
        # )
        # ax.set_ylim(0, 0.6)
        # ax.legend()
        # plt.savefig(os.path.join(self.data_path, f"{self.name_filesys}_irsa_extinction.pdf"))
        # plt.close()

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
        return exts

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
            tbl.write(os.path.join(self.data_path, "IRSA_extinction.ecsv"), overwrite=True)
            self.ebv_sandf = tbl["ext SandF ref"][0] * units.mag

    def load_extinction_table(self, force: bool = False):
        if force or self.irsa_extinction is None:
            if self.irsa_extinction_path is not None:
                u.debug_print(1, "Loading irsa_extinction from", self.irsa_extinction_path)
                self.irsa_extinction = table.QTable.read(self.irsa_extinction_path, format="ascii.ecsv")

    def jname(self, ra_precision=2, dec_precision=1):
        if self.position is not None:
            name = astm.jname(
                coord=self.position,
                ra_precision=ra_precision,
                dec_precision=dec_precision
            )
            return name

    def get_photometry_table(self, output: bool = False, best: bool = False, force: bool = False):
        if not self.photometry:
            self.load_output_file()
            if not self.photometry:
                print(f"No photometry found for {self.name}")
                return
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
        if len(fil_photom) == 0:
            print(f"No photometry found for band {fil} in instrument {instrument}.")
            return None, None
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
        if "snr_sep" not in fil_photom.colnames:
            print("It looks like this object is missing SEP photometry; try running refine_photometry for this field.")
            return None, None
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
        force_dict = None
        if "force_template_image" in self.param_file:
            force_dict = self.param_file["force_template_image"]
        self.get_photometry_table(output=local_output, best=False)
        if self.photometry_tbl is None:
            return None
        if force_dict is None:
            if "snr" not in self.photometry_tbl.colnames:
                return None
            idx = np.argmax(self.photometry_tbl["snr"])
            row = self.photometry_tbl[idx]
            instrument = row["instrument"]
            band = row["band"]
            epoch = row["epoch_name"]
        else:
            instrument = force_dict["instrument"]
            band = force_dict["band"]
            epoch = force_dict["epoch_name"]

        deepest = self.photometry[instrument][band][epoch]
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

    def select_deepest_sep(self, local_output: bool = True) -> Union[dict, None]:
        force_dict = None
        if "force_template_image" in self.param_file:
            force_dict = self.param_file["force_template_image"]
        self.get_photometry_table(output=local_output, best=True)
        if not isinstance(self.photometry_tbl_best, table.Table):
            print(f"No photometry found for {self.name}")
            return None
        if force_dict is None:
            if "snr_sep" not in self.photometry_tbl_best.colnames:
                print(f"No photometry found for {self.name}")
                return None
            idx = np.argmax(self.photometry_tbl_best["snr_sep"])
            row = self.photometry_tbl_best[idx]
            instrument = row["instrument"]
            band = row["band"]
            epoch = row["epoch_name"]
        else:
            instrument = force_dict["instrument"]
            band = force_dict["band"]
            epoch = force_dict["epoch_name"]

        return self.photometry[instrument][band][epoch]

    def assemble_row(
            self,
            **kwargs
    ):
        # if not self.photometry:
        #     return None
        select = True
        if "select" in kwargs:
            select = kwargs["select"]
        local_output = True
        if "local_output" in kwargs:
            select = kwargs["local_output"]

        jname = self.jname(4, 3)

        self.retrieve_extinction_table()

        ra_err, dec_err, theta = self.position_err.uncertainty_quadrature()

        row = {
            "jname": jname,
            "field_name": self.field.name,
            "object_name": self.name,
            "position": self.position.to_string("hmsdms"),
            "ra": self.position.ra.to(units.degree),
            "ra_err": ra_err.to("arcsec"),
            "dec": self.position.dec.to(units.degree),
            "dec_err": dec_err.to("arcsec"),
            f"e_b-v": self.ebv_sandf,
        }

        if self.optical:
            self.apply_galactic_extinction()
            if select:
                self.get_photometry_table(output=local_output, best=True)
                if not isinstance(self.photometry_tbl_best, table.Table):
                    self.get_good_photometry()
                self.photometry_to_table()
                deepest = self.select_deepest_sep(local_output=local_output)
            else:
                deepest = self.select_deepest(local_output=local_output)

            # best_position = self.select_best_position(local_output=local_output)
            best_psf = self.select_psf_photometry(local_output=local_output)

            row.update({
                "ra": deepest["ra"],
                "ra_err": deepest["ra_err"].to("arcsec"),
                "dec": deepest["dec"],
                "dec_err": deepest["dec_err"].to("arcsec"),
                "epoch_position": deepest["epoch_name"],
                "epoch_position_date": deepest["epoch_date"],
                "a": deepest["a"],
                "a_err": deepest["a_err"],
                "b": deepest["b"],
                "b_err": deepest["b_err"],
                "theta": deepest["theta"],
                "theta_err": deepest["theta_err"],
                "kron_radius": deepest["kron_radius"],
                "epoch_ellipse": deepest["epoch_name"],
                "epoch_ellipse_date": deepest["epoch_date"],
                f"class_star": best_psf["class_star"],
                "spread_model": best_psf["spread_model"],
                "spread_model_err": best_psf["spread_model_err"],
                "class_flag": best_psf["class_flag"],
            })

            for instrument in self.photometry:
                for fil in self.photometry[instrument]:

                    band_str = f"{instrument}_{fil.replace('_', '-')}"

                    if select:
                        best_photom, mean_photom = self.select_photometry_sep(
                            fil, instrument,
                            local_output=local_output
                        )
                        row[f"mag_best_{band_str}"] = best_photom["mag_sep"]
                        row[f"mag_best_{band_str}_err"] = best_photom["mag_sep_err"]
                        row[f"snr_best_{band_str}"] = best_photom["snr_sep"]

                    else:
                        best_photom, mean_photom = self.select_photometry(
                            fil,
                            instrument,
                            local_output=local_output
                        )
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
        return row, "optical"

    def push_to_table(
            self,
            **kwargs
    ):

        row, tbl_name = self.assemble_row(**kwargs)

        print(self.name)
        for key in sorted(row.keys()):
            val = row[key]
            print("\t", key, val)

        if row is None:
            return None

        import craftutils.observation.output.objects as output_objs
        # if select:
        tbl = output_objs.load_objects_table(tbl_name)
        # else:
        # tbl = obs.load_master_all_objects_table()

        tbl.add_entry(
            key=self.name,
            entry=row,
        )
        tbl.write_table()
        return row

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
            "publication_doi": None,
            "field": None,
            "other_names": [],
        }
        return default_params

    def to_param_dict(self):
        dictionary = self.default_params()
        dictionary.update({
            "name": self.name,
            "position": self.position,
        })
        if self.field is not None:
            dictionary["field"] = self.field.name
        return dictionary

    def to_param_yaml(self, path: str = None, keep_old: bool = False):
        print(f"Writing param yaml for {self.name}")
        dictionary = self.to_param_dict()
        default = self.default_params()
        if path is None and self.field is not None:
            path = str(os.path.join(self.field._obj_path(), self.name + ".yaml"))
        if os.path.isfile(path) and keep_old:
            dict_old = p.load_params(path)
            for key, value in dictionary.items():
                if key not in default or default[key] != value:
                    dict_old[key] = value
            dictionary = dict_old
        p.save_params(file=path, dictionary=dictionary)
        return dictionary

    @classmethod
    def from_dict(cls, dictionary: dict, **kwargs):
        """
        Construct an Object or appropriate child class (FRB, Galaxy...) from a passed dict.

        :param dictionary: dict with keys:
            'position': position dictionary as given by position_dictionary
            'position_err':
        :return: Object reflecting dictionary.
        """
        dictionary.update(kwargs)
        dict_pristine = dictionary.copy()
        position = dictionary.pop("position")
        if isinstance(position, dict):
            ra, dec = p.select_coords(position)
            position = f"{ra} {dec}"
        elif isinstance(position, str):
            astm.attempt_skycoord(position)
        elif not isinstance(position, SkyCoord):
            raise TypeError(f"position type {type(position)} not recognised. "
                            f"Can be SkyCoord, string (hms dms), or dictionary (see objects.position_dictionary)")

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

        return selected(
            name=name,
            position=position,
            position_err=position_err,
            plotting=plotting,
            **dictionary
        )

    @classmethod
    def select_child_class(cls, obj_type: str):
        obj_type = obj_type.lower()
        if obj_type == "galaxy":
            from .galaxy import Galaxy
            return Galaxy
        elif obj_type == "frb":
            from .frb import FRB
            return FRB
        elif obj_type == "star":
            from .star import Star
            return Star
        elif obj_type == "transienthostcandidate":
            from .transient_host import TransientHostCandidate
            return TransientHostCandidate
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
