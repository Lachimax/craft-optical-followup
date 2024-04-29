# Code by Lachlan Marnoch, 2021 - 2024

import os
import numpy as np
from typing import Union, List, Tuple

import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb

from astropy.coordinates import SkyCoord
import astropy.units as units

import craftutils.observation as obs
import craftutils.observation.objects as objects
import craftutils.observation.image as image
import craftutils.astrometry as astm

import craftutils.plotting as pl
import craftutils.utils as u
import craftutils.params as p
from .field import Field


class FRBField(Field):
    def __init__(
            self,
            name: str = None,
            centre_coords: Union[SkyCoord, str] = None,
            param_path: str = None,
            data_path: str = None,
            objs: List[objects.Object] = None,
            frb: Union[objects.FRB, dict] = None,
            extent: units.Quantity = None,
            **kwargs
    ):
        if centre_coords is None:
            if frb is not None:
                centre_coords = frb.position

        # Input attributes
        super().__init__(
            name=name,
            centre_coords=centre_coords,
            param_path=param_path,
            data_path=data_path,
            objs=objs,
            extent=extent,
            **kwargs
        )

        self.frb = frb
        if self.frb is not None:
            if isinstance(self.frb, str):
                self.frb = self.objects_dict[self.frb]
            if isinstance(self.frb, dict):
                self.frb = objects.FRB.from_dict(self.frb)
            self.frb.field = self
            self.frb.get_host()
            if self.frb.host_galaxy not in self.objects and self.frb.host_galaxy is not None:
                self.add_object(self.frb.host_galaxy)
        self.epochs_imaging_old = {}

    def plot_host_colour(
            self,
            output_path: str,
            red: image.ImagingImage,
            blue: image.ImagingImage,
            green: image.ImagingImage = None,
            fig: plt.Figure = None,
            centre: SkyCoord = None,
            show_frb: bool = True,
            show_coords: bool = True,
            frame: units.Quantity = 30 * units.pix,
            n: int = 1, n_x: int = 1, n_y: int = 1,
            frb_kwargs: dict = None,
            imshow_kwargs: dict = None,
            ext: Union[tuple, int] = (0, 0, 0),
            vmaxes: tuple = (None, None, None),
            vmins: tuple = (None, None, None),
            scale_to_jansky: bool = False,
            scale_to_rgb: bool = False,
            **kwargs
    ):
        pl.latex_setup()

        if not isinstance(self.frb, objects.FRB):
            raise TypeError("self.frb has not been set properly for this FRBField.")
        if centre is None:
            centre = self.frb.host_galaxy.position
        if fig is None:
            fig = plt.figure()
        if isinstance(ext, int):
            ext = (ext, ext, ext)
        if imshow_kwargs is None:
            imshow_kwargs = {}

        path_split = os.path.split(output_path)[-1]

        frame = u.check_quantity(frame, unit=units.pix)

        red_data, red_trimmed = red.prep_for_colour(
            output_path=output_path.replace(path_split, f"{red.name}_trimmed.fits"),
            frame=frame,
            centre=centre,
            vmax=vmaxes[0],
            vmin=vmins[0],
            ext=ext[0],
            scale_to_jansky=scale_to_jansky
        )

        blue_data, _ = blue.prep_for_colour(
            output_path=output_path.replace(path_split, f"{blue.name}_trimmed.fits"),
            frame=frame,
            centre=centre,
            vmax=vmaxes[1],
            vmin=vmins[1],
            ext=ext[1],
            scale_to_jansky=scale_to_jansky
        )

        if green is None:
            green_data = (red_data + blue_data) / 2
        else:
            green_data, _ = green.prep_for_colour(
                output_path=output_path.replace(path_split, f"{green.name}_trimmed.fits"),
                frame=frame,
                centre=centre,
                vmax=vmaxes[2],
                vmin=vmins[2],
                ext=ext[2],
                scale_to_jansky=scale_to_jansky
            )

        if scale_to_rgb:
            max_all = max(np.max(red_data), np.max(green_data), np.max(blue_data))
            factor = max_all / 255
            red_data /= factor
            green_data /= factor
            blue_data /= factor

        colour = make_lupton_rgb(
            red_data,
            green_data,
            blue_data,
            Q=7,
            stretch=30
        )

        if "origin" not in imshow_kwargs:
            imshow_kwargs["origin"] = "lower"

        if show_coords:
            projection = red_trimmed.wcs[ext[0]]
        else:
            projection = None
        ax = fig.add_subplot(n_x, n_y, n, projection=projection)

        if not show_coords:
            ax.get_xaxis().set_visible(False)
            ax.set_yticks([])
            ax.invert_yaxis()

        ax.imshow(
            colour,
            **imshow_kwargs,
        )
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        # ax.set_xlabel("Right Ascension (J2000)", size=16)
        # ax.set_ylabel("Declination (J2000)", size=16, rotation=0, labelpad=-20)
        ax.tick_params(labelsize=10)
        # ax.yaxis.set_label_position("right")
        # plt.tight_layout()

        if show_frb:
            self.frb_ellipse_to_plot(ext=ext[0], frb_kwargs=frb_kwargs, img=red_trimmed, ax=ax)

        fig.savefig(output_path)
        return ax, fig, colour

    def plot_host(
            self,
            img: image.ImagingImage,
            ext: int = 0,
            fig: plt.Figure = None,
            ax: plt.Axes = None,
            centre: SkyCoord = None,
            show_frb: bool = True,
            frame: units.Quantity = 30 * units.pix,
            n: int = 1, n_x: int = 1, n_y: int = 1,
            # ticks: int = None, interval: str = 'minmax',
            # font_size: int = 12,
            # reverse_y=False,
            frb_kwargs: dict = None,
            imshow_kwargs: dict = None,
            normalize_kwargs: dict = None,
            output_path: str = None,
            show_legend: bool = False,
            latex_kwargs: dict = None,
            do_latex_setup: bool = True,
            draw_scale_bar: bool = False,
            scale_bar_kwargs: dict = None,
            include_img_err: bool = True,
            **kwargs
    ) -> Tuple[plt.Axes, plt.Figure, dict]:

        if imshow_kwargs is None:
            imshow_kwargs = {}
        if frb_kwargs is None:
            frb_kwargs = {}
        if latex_kwargs is None:
            latex_kwargs = {}
        if scale_bar_kwargs is None:
            scale_bar_kwargs = {}

        if do_latex_setup:
            pl.latex_setup(**latex_kwargs)
        if not isinstance(self.frb, objects.FRB):
            raise TypeError("self.frb has not been set properly for this FRBField.")
        if centre is None:
            centre = self.frb.host_galaxy.position
        if centre is None:
            centre = self.frb.position

        if draw_scale_bar:
            kwargs["scale_bar_object"] = self.frb.host_galaxy

        ax, fig, other_args = img.plot_subimage(
            centre=centre,
            frame=frame,
            ext=ext,
            fig=fig,
            ax=ax,
            n=n, n_x=n_x, n_y=n_y,
            imshow_kwargs=imshow_kwargs,
            normalize_kwargs=normalize_kwargs,
            scale_bar_kwargs=scale_bar_kwargs,
            obj=self.frb.host_galaxy,
            **kwargs
        )

        if show_frb:
            self.frb_ellipse_to_plot(
                ext=ext,
                frb_kwargs=frb_kwargs,
                img=img,
                ax=ax,
                include_img_err=include_img_err
            )
            if show_legend:
                ax.legend()

        if output_path is not None:
            fig.savefig(output_path)

        return ax, fig, other_args

    def frb_ellipse_to_plot(
            self,
            ax,
            img: image.ImagingImage,
            ext: int = 0,
            colour: str = None,
            frb_kwargs: dict = None,
            plot_centre: bool = False,
            include_img_err: bool = True
    ):
        if frb_kwargs is None:
            frb_kwargs = {}
        img.load_headers()
        frb = self.frb.position
        uncertainty = self.frb.position_err
        a, b = uncertainty.uncertainty_quadrature()
        if a == 0 * units.arcsec or b == 0 * units.arcsec:
            a, b = uncertainty.uncertainty_quadrature_equ()
        theta = uncertainty.theta.to(units.deg)
        rotation_angle = img.extract_rotation_angle(ext=ext)
        theta = theta - rotation_angle
        img.extract_pixel_scale()

        if "include_img_err" in frb_kwargs:
            include_img_err = frb_kwargs.pop("include_img_err")
        if "edgecolor" not in frb_kwargs:
            if colour is None:
                colour = "white"
            frb_kwargs["edgecolor"] = colour
        if "facecolor" not in frb_kwargs:
            frb_kwargs["facecolor"] = "none"
        img_err = None
        if include_img_err:
            img_err = img.extract_astrometry_err()
        if img_err is not None:
            a = np.sqrt(a ** 2 + img_err ** 2)
            b = np.sqrt(b ** 2 + img_err ** 2)
        ax = img.plot_ellipse(
            ax=ax,
            coord=frb,
            a=a, b=b,
            theta=theta,
            plot_centre=plot_centre,
            centre_kwargs=dict(
                c=frb_kwargs["edgecolor"],
                marker="x"
            ),
            **frb_kwargs,
        )

        return ax

    @classmethod
    def stages(cls):
        field_stages = super().stages()
        stages = {
            "update_photometry": field_stages["update_photometry"],
            "probabilistic_association": {
                "method": cls.proc_probabilistic_association,
                "message": "Run PATH on available imaging?"
            },
            "refine_photometry": field_stages["refine_photometry"]
        }
        return stages

    def proc_probabilistic_association(self, output_dir: str, **kwargs):
        pass

    @classmethod
    def default_params(cls):
        default_params = super().default_params()

        default_params.update({
            "type": "FRBField",
            "frb": objects.FRB.default_params(),
            "subtraction":
                {
                    "template_epochs":
                        {
                            "des": None,
                            "fors2": None,
                            "xshooter": None,
                            "sdss": None
                        }
                },
        })

        return default_params

    @classmethod
    def new_yaml(cls, name: str, path: str = None, **kwargs) -> dict:
        """
        Generates a new parameter .yaml file for an FRBField.

        :param name: Name of the field.
        :param path: Path to write .yaml to.
        :param kwargs: Other keywords to insert or replace in the output yaml.
        :return: dict reflecting content of yaml file.
        """
        param_dict = super().new_yaml(name=name, path=None)
        param_dict["frb"] = name
        for kwarg in kwargs:
            param_dict[kwarg] = kwargs[kwarg]
        if path is not None:
            path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict)
        u.debug_print(2, "FRBField.new_yaml(): param_dict:", param_dict)
        return param_dict

    @classmethod
    def yaml_from_furby_dict(
            cls,
            furby_dict: dict,
            output_path: str,
            healpix_path: str = None) -> dict:
        """
        Constructs a param .yaml file from a dict representing a FURBY json file.
        :param furby_dict: the .json file read in as a dict
        :param output_path: The path to write output yaml file to.
        :param healpix_path: Optional, path to FITS file containing healpix information.
        :return: Dictionary containing the same information as the written .yaml
        """

        u.mkdir_check(output_path)

        field_name = furby_dict["Name"]
        frb = objects.FRB.default_params()
        coords = objects.position_dictionary.copy()

        ra = furby_dict["RA"]
        dec = furby_dict["DEC"]

        pos_coord = astm.attempt_skycoord((ra * units.deg, dec * units.deg))
        ra_str, dec_str = astm.coord_string(pos_coord)

        coords["ra"]["decimal"] = ra
        coords["dec"]["decimal"] = dec
        coords["ra"]["hms"] = ra_str
        coords["dec"]["dms"] = dec_str

        obs.load_furby_table()
        row, _ = obs.get_row_furby(field_name)
        if row is not None:
            frb["position_err"]["a"]["stat"] = row["sig_ra"]
            frb["position_err"]["b"]["stat"] = row["sig_dec"]

        frb["dm"] = furby_dict["DM"] * objects.dm_units
        frb["name"] = field_name
        frb["position"] = coords.copy()
        frb["position_err"]["healpix_path"] = healpix_path
        frb["host_galaxy"]["name"] = "HG" + field_name[3:]
        param_dict = cls.new_yaml(
            name=field_name,
            path=output_path,
            centre=coords,
            frb=frb,
            snr=furby_dict["S/N"],
            survey="furby"
        )

        return param_dict

    @classmethod
    def param_from_furby_json(cls, json_path: str, healpix_path: str = None):
        """
        Constructs a param .yaml file from a FURBY json file and places it in the default location.
        :param json_path: The path to the FURBY .json file.
        :param healpix_path: Optional, path to FITS file containing healpix information.
        :return:
        """
        furby_dict = p.load_json(json_path)
        u.debug_print(2, "FRBField.param_from_furby_json(): json_path ==", json_path)
        u.debug_print(2, "FRBField.param_from_furby_json(): furby_dict ==", furby_dict)
        field_name = furby_dict["Name"]
        output_path = str(os.path.join(p.param_dir, "fields", field_name))

        param_dict = cls.yaml_from_furby_dict(
            furby_dict=furby_dict,
            healpix_path=healpix_path,
            output_path=output_path
        )
        return param_dict
