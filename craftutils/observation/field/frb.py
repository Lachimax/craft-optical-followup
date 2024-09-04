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
import craftutils.observation.filters as filters
import craftutils.observation.instrument as inst
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
                self.frb = self.objects[self.frb]
            if isinstance(self.frb, dict):
                self.frb = objects.FRB.from_dict(self.frb)
            self.frb.field = self
            self.frb.get_host()
            if self.frb.host_galaxy not in self.objects and self.frb.host_galaxy is not None:
                self.add_object(self.frb.host_galaxy)
        self.epochs_imaging_old = {}
        self.path_runs = {}
        self.best_path_img = None

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
        return fig, ax, colour

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
    ) -> Tuple[plt.Figure, plt.Axes, dict]:
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

        fig, ax, other_args = img.plot_subimage(
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

        return fig, ax, other_args

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
        a, b, theta = uncertainty.uncertainty_quadrature()
        # if a == 0 * units.arcsec or b == 0 * units.arcsec:
        #     a, b, theta = uncertainty.uncertainty_quadrature_equ()
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
            "finalise_imaging": field_stages["finalise_imaging"],
            "probabilistic_association": {
                "method": cls.proc_probabilistic_association,
                "message": "Run PATH on available imaging?",
                "keywords": {
                    "path_kwargs": {},
                    "path_img": None
                }
            },
            "update_photometry": field_stages["update_photometry"],
            "refine_photometry": field_stages["refine_photometry"],
            "galfit": field_stages["galfit"],
            "send_to_table": field_stages["send_to_table"]
        }
        return stages

    def proc_probabilistic_association(self, output_dir: str, **kwargs):
        path_kwargs = {
            "config": {"radius": 10}
        }
        if 'path_kwargs' in kwargs and kwargs['path_kwargs'] is not None:
            path_kwargs.update(kwargs["path_kwargs"])
        if 'path_img' in kwargs and kwargs['path_img'] is not None:
            path_img = kwargs.pop("path_img")
        else:
            path_img = None
        self.probabilistic_association(path_img=path_img, **path_kwargs)

    def probabilistic_association(self, path_img: str = None, **path_kwargs):
        self.load_imaging()
        # filter_list = self.get_filters()
        # failed = []
        if "priors" not in path_kwargs:
            path_kwargs["priors"] = {}
        fil_list = self.best_fil_for_path()
        pl.latex_setup()

        images = list(map(lambda f: self.deepest_in_band(fil=f)["image"], fil_list))

        if path_img is not None:
            path_img = self.imaging[path_img]["image"]
            if path_img not in images:
                images.insert(0, path_img)
        else:
            path_img = images[0]

        # if max_p_ox is not None:
        #     self.add_path_candidates()

        max_p_ox = None
        while max_p_ox in (None, 0.) and images:
            img = images.pop(0)
            p_us = [0., 0.1, 0.2]
            p_u_calculated = -999.
            vals = None
            print("SURVEY", self.survey, self.survey == "CRAFT_ICS")
            if self.survey.name == "CRAFT_ICS":
                vals, tbl, z_lost = self.frb.host_probability_unseen(
                    img=img,
                    sample="Gordon+2023",
                    n_z=500
                )
                if vals is not None:
                    p_u_calculated = float(vals["P(U)"]["step"])
                    p_us.append(p_u_calculated)
            for p_u in p_us:
                path_kwargs["priors"]["U"] = p_u
                cand_tbl, write_dict = self.frb.probabilistic_association(
                    img=img,
                    do_plot=True,
                    **path_kwargs
                )
                max_p_ox = write_dict["max_P(O|x_i)"]
                if max_p_ox is not None:
                    path_img = img
                    if img.name not in self.path_runs:
                        self.path_runs[img.name] = {}
                    self.path_runs[img.name][p_u] = write_dict
                    if p_u == p_u_calculated and p_u > 0.:
                        write_dict["p_u_calculation"] = vals
                        self.path_runs[img.name]["calculated"] = write_dict

        path_cat = self.frb.consolidate_candidate_tables(
            sort_by="P_Ox",
            reverse_sort=True,
            p_ox_assign=path_img.name,
            p_u=0.1
        )
        # If the custom P(U) run was unsuccessful, use the results for P(U) = 0.1
        # if p_u == 0.1:  # and max_p_ox is None:
        self.add_path_candidates()

        self.best_path_img = path_img.name

    def add_path_candidates(self):
        host_candidates = list(
            filter(
                lambda o: isinstance(o, objects.TransientHostCandidate) and o.P_Ox is not None,
                self.frb.host_candidates
            )
        )
        if isinstance(self.frb.host_galaxy, objects.Galaxy):
            print()
            print(f"Initial host {self.frb.host_galaxy.name}.z:", self.frb.host_galaxy.z)
            print()
        if len(host_candidates) > 0:
            max_pox = np.max(list(map(lambda o: o.P_Ox, host_candidates)))
            for obj in self.frb.host_candidates:
                print("Checking", obj.name)
                P_Ox = obj.P_Ox
                if P_Ox > 0.05:
                    print(f"\tAdding {obj.name}: P(O|x) = {P_Ox} > 0.05.")
                    if P_Ox >= max_pox:
                        self.frb.set_host(obj, keep_params=["z", "z_err", "other_names", "force_template_img"])
                    self.add_object(obj)
                    obj.to_param_yaml(keep_old=True)
        print()
        if self.frb.host_galaxy is not None:
            print(f"New host {self.frb.host_galaxy.name}.z:", self.frb.host_galaxy.z)
        print()

    def best_fil_for_path(
            self,
            exclude: list = ()
    ):
        # TODO: Allow for other instruments; there needs to be a depth check here so that VLT imaging gets selected over
        # survey imaging
        filter_list = self.load_imaging(instrument="vlt-fors2")
        best_fil = filters.best_for_path(filter_list, exclude=exclude)
        # path_dict = self.deepest_in_band(fil=best_fil)
        # path_img = path_dict["image"]
        # print()
        # print(f"The image selected for PATH is {path_img.name}, with depth {path_dict['depth']}")
        return filter_list

    def galfit(self, apply_filter=None, use_img=None, **kwargs):

        if apply_filter is None:
            def hg_fil(o):
                return o.name.startswith("HG")
            apply_filter = hg_fil
        if use_img is None and self.best_path_img is not None:
            use_img = self.best_path_img

        super().galfit(apply_filter=apply_filter, use_img=use_img)

    def _output_dict(self):
        output_dict = super()._output_dict()
        output_dict["path_runs"] = self.path_runs
        output_dict["best_path_img"] = self.best_path_img
        return output_dict

    def load_output_file(self, **kwargs):
        output_dict = super().load_output_file(**kwargs)
        if output_dict is not None:
            if "path_runs" in output_dict and isinstance(output_dict["path_runs"], dict):
                self.path_runs = output_dict["path_runs"]
            if "best_path_img" in output_dict and isinstance(output_dict["best_path_img"], str):
                self.best_path_img = output_dict["best_path_img"]
        return output_dict

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
        """Generates a new parameter .yaml file for an FRBField.

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

        from craftutils.observation.output.furby import furby_cat

        furby_cat.load_table()
        row, _ = furby_cat.get_row(colname="field_name", colval=field_name)
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
