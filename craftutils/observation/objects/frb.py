from typing import Union
import os

import matplotlib.pyplot as plt
import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as units
import astropy.table as table
from astropy.visualization import quantity_support
import astropy.time as time

import craftutils.params as p
import craftutils.astrometry as astm
import craftutils.utils as u
import craftutils.observation.sed as sed
from . import PositionUncertainty

from .objects import object_from_index
from .extragalactic_transient import ExtragalacticTransient, cosmology
from .transient_host import TransientHostCandidate

quantity_support()

dm_units = units.parsec * units.cm ** -3

dm_host_median = {
    "james_22A": 129 * dm_units,
    "james_22B": 186 * dm_units
}


@u.export
class FRB(ExtragalacticTransient):
    optical = False

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
        # Observed Dispersion Measure
        self.dm = dm
        if self.dm is not None:
            self.dm = u.check_quantity(self.dm, unit=dm_units)
        self.dm_err = 0 * dm_units
        if "dm_err" in kwargs:
            self.dm_err = u.check_quantity(kwargs["dm_err"], dm_units)

        # Scattering model parameters
        # ===========================
        # Intrinsic width; 2 * sigma of gaussian model
        self.width_int = None
        if "width_int" in kwargs:
            self.width_int = u.check_quantity(kwargs["width_int"], units.ms)
        self.width_int_err = None
        if "width_int_err" in kwargs:
            self.width_int_err = u.check_quantity(kwargs["width_int_err"], units.ms)
        self.width_total = None
        if "width" in kwargs:
            self.width_total = u.check_quantity(kwargs["width"], units.ms)
        self.width_total_err = None
        if "width_err" in kwargs:
            self.width_total_err = u.check_quantity(kwargs["width_err"], units.ms)
        # Scattering timescale, exponent of exponential model
        self.tau = None
        if "tau" in kwargs:
            self.tau = u.check_quantity(kwargs["tau"], units.ms)
        self.tau_err = None
        if "tau_err" in kwargs:
            self.tau_err = u.check_quantity(kwargs["tau_err"], units.ms)
        # Frequency at which scattering is measured
        self.nu_scattering = None
        if "nu_scattering" in kwargs:
            self.nu_scattering = u.check_quantity(kwargs["nu_scattering"], units.GHz)

        if self.width_total is None and self.width_int is not None and self.tau is not None:
            self.width_total = self.width_int + self.tau
            if self.width_int_err is not None and self.tau_err is not None:
                self.width_total_err = np.sqrt(self.tau_err ** 2 + self.width_int_err ** 2)

        # Detection parameters
        # ====================
        self.instrument: str = None
        if "instrument" in kwargs:
            self.instrument = kwargs["instrument"]
        self.survey: str = None
        if "survey" in kwargs:
            self.survey = kwargs["survey"]
        self.snr = None
        if "snr" in kwargs:
            self.snr = kwargs["snr"]

        # DM components
        self._dm_mw_ism_ne2001 = None
        self._dm_mw_ism_ymw16 = None

        # Tau components
        self._tau_mw_ism_ne2001 = None
        self._tau_mw_ism_ymw16 = None

        self.zdm_table: table.QTable = None

        # Placeholder for an associated `frb.frb.FRB` object
        self.x_frb = None

    def generate_x_frb(self):
        from frb.frb import FRB
        self.x_frb = FRB(
            frb_name=self.name,
            coord=self.position,
            DM=self.dm
        )
        return self.x_frb

    def assemble_row(
            self,
            **kwargs
    ):
        row, _ = super().assemble_row(**kwargs)
        if isinstance(self.host_galaxy, TransientHostCandidate):
            row["host_galaxy"] = self.host_galaxy.name
            hg_row = self.host_galaxy.assemble_row(**kwargs)
            if f"path_pox" in hg_row:
                row["path_pox"] = hg_row["path_pox"]
            if f"path_pu" in hg_row:
                row["path_pu"] = hg_row["path_pu"]
            if f"path_pux" in hg_row:
                row["path_pux"] = hg_row["path_pux"]
            if "path_img" in hg_row:
                row["path_img"] = hg_row["path_img"]

        if self.date is not None:
            row["date"] = str(self.date)
        if self.survey is not None:
            row["survey"] = self.survey
        if self.snr is not None and self.snr != 0.:
            row["snr"] = float(self.snr)
        if self.tns_name is not None:
            row["tns_name"] = self.tns_name
        if self.z is not None:
            row["z"] = self.z

        return row, "frb"

    def probabilistic_association(
            self,
            img,
            include_img_err: bool = True,
            prior_set: Union[str, dict] = "adopted",
            priors: dict = None,
            offset_priors: dict = None,
            config: dict = None,
            associate_kwargs=None,
            do_plot: bool = False,
            output_dir: str = None,
            show: bool = False,
            max_radius: units.Quantity = None,
    ) -> (table.QTable, dict):
        """Performs a customised PATH run on an image.

        :param img: The image on which to run PATH.
        :param include_img_err: If set to True, the image astrometry RMS (from img.extract_astrometry_err()) will be
            added to the transient localisation error in quadrature.
        :param prior_set:
        :param priors:
        :param offset_priors:
        :param config:
        :param associate_kwargs:
        :param do_plot:
        :param output_dir:
        :param show:
        :param max_radius:
        :return:
        """
        import frb.associate.frbassociate as associate
        import astropath.path as path
        from craftutils.observation.field import FRBField
        astm_rms = 0.
        if config is None:
            config = {}
        if priors is None:
            priors = {}
        if offset_priors is None:
            offset_priors = {"scale": 0.5}  # As per Shannon+2024
        if associate_kwargs is None:
            associate_kwargs = {"extinction_correct": True}
        if include_img_err:
            astm_rms = img.extract_astrometry_err()
        if astm_rms is None:
            astm_rms = 0.
        if output_dir is None:
            self.check_data_path()
            output_dir = str(os.path.join(self.data_path, "PATH", img.name))
            u.mkdir_check(output_dir)
        a, b, theta = self.position_err.uncertainty_quadrature()
        a = np.sqrt(a ** 2 + astm_rms ** 2)
        b = np.sqrt(b ** 2 + astm_rms ** 2)
        # theta = self.position_err.theta
        x_frb = self.generate_x_frb()
        print(f"Passing {self.name}, with {a=}, {b=}, {theta=}")
        x_frb.set_ee(
            a=a.value,
            b=b.value,
            theta=theta.value,
            cl=0.68,
        )
        #     img.load_output_file()
        img.extract_pixel_scale()
        if img.filter.frb_repo_name is None:
            if img.instrument.cigale_name is not None:
                instname = img.instrument.cigale_name
            else:
                instname = img.instrument.name.replace("-", "_").upper()
            filname = f'{instname}_{img.filter.band_name}'
        else:
            filname = img.filter.frb_repo_name
        # TODO: subtract Galactic extinction from zeropoint?

        if max_radius is not None:
            max_radius = u.dequantify(max_radius, unit=units.arcsec)
        elif "max_radius" in config:
            max_radius = config["max_radius"]
        else:
            max_radius = 40.
        import matplotlib
        # matplotlib.use("Qt5Agg")
        config_n = dict(
            max_radius=int(max_radius),
            skip_bayesian=False,
            npixels=9,
            image_file=img.path,
            cut_size=max_radius * 2,
            filter=filname,
            ZP=img.zeropoint_best["zeropoint_img"].value,
            deblend=True,
            cand_bright=17.,
            cand_separation=max_radius * units.arcsec,
            plate_scale=(1 * units.pix).to(units.arcsec, img.pixel_scale_y),
        )
        config_n.update(config)
        config = config_n

        # Load priors from astropath repo
        priors_std = path.priors.load_std_priors()
        if isinstance(prior_set, str):
            if prior_set in priors_std:
                prior_set = priors_std[prior_set]
            else:
                raise ValueError(f"Prior set '{prior_set}' not recognised; available are: {list(priors_std.keys())}")
        elif isinstance(prior_set, dict):
            priors_adopted = priors_std["adopted"]
            priors_adopted.update(prior_set)
            prior_set = priors_adopted
        # Update with passed priors
        prior_set["theta"].update(offset_priors)
        prior_set.update(priors)

        p_u = prior_set["U"] = float(prior_set["U"])

        # print("P(U) ==", p_u)
        print()
        print("Priors:", prior_set)
        print("Config:", config)
        print()

        # if "show" not in associate_kwargs:
        #     associate_kwargs["show"] = show

        # try:
        print("Trying...")
        ass = associate.run_individual(
            config=config,
            FRB=x_frb,
            prior=prior_set,
            show=True,
            **associate_kwargs,
            # extinction_correct=True
        )
        p_ux = ass.P_Ux
        print("P(U|x) ==", p_ux)
        cand_tbl = table.QTable.from_pandas(ass.candidates)
        if "P_Ux" not in cand_tbl.colnames:
            cand_tbl["P_Ux"] = [p_ux] * len(cand_tbl)

        if np.isnan(p_ux):
            p_ux = None

        cand_tbl["P_U"] = [p_u] * len(cand_tbl)
        max_p_ox = cand_tbl[0]["P_Ox"]
        print("Max P(O|x_i) ==", max_p_ox)
        if np.ma.is_masked(max_p_ox):
            max_p_ox = None
        print("\n\n")
        cand_tbl["ra"] *= units.deg
        cand_tbl["dec"] *= units.deg
        coord = SkyCoord(cand_tbl["ra"], cand_tbl["dec"])
        cand_tbl["x"], cand_tbl["y"] = img.world_to_pixel(coord=coord)
        cand_tbl["separation"] *= units.arcsec
        cand_tbl[filname] *= units.mag
        p_u_rnd = np.round(p_u, 4)
        if p_u_rnd not in self.host_candidate_tables:
            self.host_candidate_tables[p_u_rnd] = {}
        self.host_candidate_tables[p_u_rnd][img.name] = cand_tbl
        self.update_output_file()

        if do_plot and isinstance(self.field, FRBField) and (show or output_dir):
            fig = plt.figure(figsize=(12, 12))
            fig, ax, _ = self.field.plot_host(
                img=img,
                fig=fig,
                frame=cand_tbl["separation"].max(),
                centre=self.position
            )

            for n, obj in self.field.objects.items():
                if obj.position is not None:
                    x, y = img.world_to_pixel(obj.position)
                    ax.text(x=x, y=y, s=n, color="white")

            c = ax.scatter(cand_tbl["x"], cand_tbl["y"], marker="x", c=cand_tbl["P_Ox"], cmap="bwr_r")
            cbar = fig.colorbar(c)
            cbar.set_label("$P(O|x)_i$")
            if show:
                plt.show(fig)
            if output_dir:
                fig.savefig(os.path.join(output_dir, f"{self.name}_PATH_{img.name}_PU_{p_u}.pdf"))
            plt.close(fig)

        write_dict = {
            "priors": prior_set,
            "config": config_n,
            "max_P(O|x_i)": max_p_ox,
            "P(U|x)": p_ux,
            "output_dir": output_dir,
        }

        if output_dir:
            for fmt in ("csv", "ecsv"):
                cand_tbl.write(
                    os.path.join(output_dir, f"{self.name}_PATH_{img.name}_PU_{p_u}.{fmt}"),
                    format=f"ascii.{fmt}",
                    overwrite=True
                )

            p.save_params(
                os.path.join(output_dir, f"{self.name}_PATH_{img.name}_PU_{p_u}.yaml"),
                write_dict
            )

        # except IndexError:
        #     print("Failed.")
        #     write_dict = {
        #         "priors": prior_set,
        #         "config": config_n,
        #         "max_P(O|x_i)": None,
        #         "P(U|x)": None,
        #         "output_dir": output_dir,
        #     }
        #     cand_tbl = None

        return cand_tbl, write_dict

    def consolidate_candidate_tables(
            self,
            sort_by: str = "separation",
            reverse_sort: bool = False,
            p_ox_assign: str = None,
            p_u: float = 0.1,
            output_path: str = None
    ):
        # Build a shared catalogue of host candidates.
        path_cat = None
        p_u = float(np.round(p_u, 4))
        for tbl_name in self.host_candidate_tables[p_u]:
            print(tbl_name)
            if tbl_name == "consolidated":
                continue
            if p_ox_assign is None:
                p_ox_assign = tbl_name
            # print(tbl_name)
            cand_tbl = self.host_candidate_tables[p_u][tbl_name]
            if path_cat is None:
                path_cat = cand_tbl["label", "ra", "dec", "separation"]
            path_cat, matched, dist = astm.match_catalogs(
                cat_1=path_cat, cat_2=cand_tbl,
                ra_col_1="ra", dec_col_1="dec",
                keep_non_matches=True,
                tolerance=0.7 * units.arcsec
            )

            for prefix in ["label", "P_c", "P_Ox", "P_O", "p_xO", "P_Ux", "mag"]:
                if f"{prefix}_{tbl_name}" not in matched.colnames:
                    # print(f"{prefix}_{tbl_name}")
                    matched[f"{prefix}_{tbl_name}"] = matched[prefix]

                for col in list(filter(lambda c: c.startswith(prefix + "_"), path_cat.colnames)):
                    matched[col] = np.ones(len(matched)) * -999.

                path_cat[f"{prefix}_{tbl_name}"] = np.ones(len(path_cat)) * -999.
                path_cat[f"{prefix}_{tbl_name}"][path_cat["matched"]] = matched[f"{prefix}_{tbl_name}"][
                    matched["matched"]]

            for row in matched[np.invert(matched["matched"])]:
                print(
                    f'Adding label {row["label"]} from {tbl_name} table. ra={row["ra"]}, dec={row["dec"]}, P_Ox_{p_ox_assign}')
                path_cat.add_row(row[path_cat.colnames])

        # path_cat["coord"] = SkyCoord(path_cat["ra"], path_cat["dec"])
        if sort_by == "P_Ox":
            sort_by = f"P_Ox_{p_ox_assign}"
        path_cat.sort(sort_by, reverse=reverse_sort)
        ids = []
        id_strs = []
        for i, row in enumerate(path_cat):
            ids.append(i)
            id_strs.append(str(i).zfill(int(np.ceil(np.log10(len(path_cat))))))
        path_cat["id"] = ids
        path_cat["id_str"] = id_strs
        self.host_candidate_tables[p_u]["consolidated"] = path_cat
        best_i = np.argmax(path_cat[f"P_Ox_{p_ox_assign}"])
        for i, row in enumerate(path_cat):
            idn = self.name.replace("FRB", "")
            host_candidate = TransientHostCandidate(
                z=None,
                transient=self,
                position=SkyCoord(row["ra"], row["dec"]),
                field=self.field,
                name=f"HC{row['id_str']}_{idn}",
                P_O=row[f"P_O_{p_ox_assign}"],
                p_xO=row[f"p_xO_{p_ox_assign}"],
                P_Ox=row[f"P_Ox_{p_ox_assign}"],
                P_U=p_u,
                P_Ux=row[f"P_Ux_{p_ox_assign}"],
                probabilistic_association_img=p_ox_assign
            )
            self.host_candidates.append(host_candidate)
            # if i == best_i and row[f"P_Ox_{p_ox_assign}"] > 0.9:
            #     self.host_galaxy = host_candidate

        self.update_output_file()
        return path_cat

    def write_candidate_tables(self):
        table_paths = {}
        for p_u in self.host_candidate_tables:
            table_paths[p_u] = {}
            for img_name in self.host_candidate_tables[p_u]:
                cand_tbl = self.host_candidate_tables[p_u][img_name]
                write_path = os.path.join(self.data_path, "PATH", f"PATH_table_{img_name}_PU_{p_u}.ecsv")
                if "coords" in cand_tbl.colnames:
                    cand_tbl.remove_column("coords")
                cand_tbl.write(write_path, overwrite=True)
                table_paths[p_u][img_name] = p.split_data_dir(write_path)
        return table_paths

    def _output_dict(self):
        output = super()._output_dict()
        cand_list = list(map(lambda o: str(o), self.host_candidates))

        output.update({
            "host_candidate_tables": self.write_candidate_tables(),
            "host_candidates": cand_list
        })
        return output

    def load_output_file(self):
        outputs = super().load_output_file()
        if outputs is not None:
            if "host_candidate_tables" in outputs:
                p_us = outputs["host_candidate_tables"]
                for p_u in p_us:
                    if p_u not in self.host_candidate_tables:
                        self.host_candidate_tables[p_u] = {}
                    tables = outputs["host_candidate_tables"][p_u]
                    for table_name in tables:
                        tbl_path = tables[table_name]
                        tbl_path = p.join_data_dir(tbl_path)
                        if os.path.isfile(tbl_path):
                            try:
                                tbl = table.QTable.read(tbl_path)
                                self.host_candidate_tables[p_u][table_name] = tbl
                            except StopIteration:
                                continue

            if "host_candidates" in outputs:
                for obj_dict in outputs["host_candidates"]:
                    if isinstance(obj_dict, str):
                        obj_name = obj_dict
                    else:
                        obj_name = obj_dict["name"]
                    obj = object_from_index(obj_name, tolerate_missing=True)
                    if obj is None:
                        obj = obj_name
                    self.host_candidates.append(obj)

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        default_params.update({
            "type": "FRB",
            "dm": 0.0 * dm_units,
            "snr": 0.0,
            "host_galaxy": None,
            "date": "0000-01-01",
            "tau": None,
            "tau_err": None,
            "width_intrinsic": None,
            "width_intrinsic_err": None,
            "width_total": None,
            "width_total_err": None,
            "tns_name": None
        })
        return default_params

    def to_param_dict(self):
        dictionary = self.default_params()
        dictionary.update(super().to_param_dict())
        dictionary.update({
            "host_galaxy": self.host_galaxy.name,
        })
        return dictionary

    @classmethod
    def host_name(cls, frb_name: str):
        if "FRB" in frb_name:
            host_name = frb_name.replace("FRB", "HG")
        else:
            host_name = frb_name + " Host"
        return host_name

    def hg_name(self):
        if self.tns_name is not None:
            name = self.tns_name
        else:
            name = self.name
        return self.host_name(frb_name=name)

    def set_host(self, host_galaxy: TransientHostCandidate, keep_params: list = None):
        hg_name = self.hg_name()
        old_name = host_galaxy.name
        old_host = self.host_galaxy
        print(f"Assigning {old_name} as host of {self.name} and relabelling as {hg_name}")
        if keep_params is not None and old_host is not None:
            attributes = old_host.__dict__.copy()
            print("Keeping attributes:")
            for key in keep_params:
                if key in attributes and attributes[key] is not None:
                    print(f"{key}:", attributes[key])
                    host_galaxy.__setattr__(key, attributes[key])
                else:
                    if key in attributes:
                        print(f"{old_name}.{key} is None.")
                    else:
                        print(f"{key} not found in {old_name}.")
        host_galaxy.set_name(name=hg_name)
        self.host_galaxy = host_galaxy
        self.host_galaxy.transient = self
        self.update_param_file("host_galaxy")
        from craftutils.observation.field import Field
        if isinstance(self.field, Field):
            self.field.remove_object(old_name)
            self.field.remove_object(old_host)
        if self.host_galaxy.z is not None:
            self.set_z(self.host_galaxy.z)
        self.update_param_file("z")
        self.update_param_file("other_names")
        # self.field.add_object(self.host_galaxy)

    @classmethod
    def default_host_params(cls, frb_name: str, position=None, **kwargs):
        default_params = TransientHostCandidate.default_params()
        host_name = cls.host_name(frb_name)

        default_params["name"] = host_name
        default_params["transient"] = frb_name
        if position:
            default_params["position"] = position
        default_params.update(**kwargs)
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

    def get_host(self, name: str = None) -> TransientHostCandidate:
        if name is None:
            name = self.hg_name()
        super().get_host(name=name)
        if self.z is None and self.host_galaxy is not None:
            self.set_z(self.host_galaxy.z)
        return self.host_galaxy

    def date_from_name(self):
        date_str = self._date_from_name(self.name)
        try:
            date = time.Time(date_str)
            self.date = date
            return date
        except ValueError:
            return date_str

    def dm_mw_ism_ne2001_baror(
            self,
            distance: Union[units.Quantity, float] = 200 * units.kpc,
    ) -> units.Quantity:
        """
        Derives the ISM component of the DM using the Bar-Or, Prochaska implementation of NE2001:
        https://github.com/FRBs/ne2001

        :param distance: Distance to object; for extragalactic objects, use a value greater than 100 kpc.
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

    def dm_mw_ism_ne2001(
            self,
            distance: Union[units.Quantity, float] = np.inf * units.kpc,
            force: bool = False
    ) -> units.Quantity['pc cm-3']:
        if self._dm_mw_ism_ne2001 is None or force:
            self._dm_mw_ism_ne2001, self._tau_mw_ism_ne2001 = self._dm_mw_ism_pygedm(
                method="ne2001",
                distance=min(distance, 50 * units.kpc)
            )
        self._tau_mw_ism_ne2001 = self._tau_mw_ism_ne2001.to("ms")
        return self._dm_mw_ism_ne2001

    def dm_mw_ism_ymw16(
            self,
            distance: Union[units.Quantity, float] = 50 * units.kpc,
            force: bool = False
    ):
        if self._dm_mw_ism_ymw16 is None or force:
            self._dm_mw_ism_ymw16, self._tau_mw_ism_ymw16 = self._dm_mw_ism_pygedm(
                method="ymw16",
                distance=distance
            )
        self._tau_mw_ism_ymw16 = self._tau_mw_ism_ymw16.to("ms")
        return self._dm_mw_ism_ymw16

    def _dm_mw_ism_pygedm(
            self,
            method: str,
            distance: Union[units.Quantity, float] = np.inf * units.kpc,
    ):
        import pygedm
        if distance > 100 * units.kpc or np.isnan(distance):
            distance = 100 * units.kpc
        dm, tau = pygedm.dist_to_dm(
            self.position.galactic.l,
            self.position.galactic.b,
            distance,
            method=method
        )
        return dm, tau

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
            model_kwargs: dict = {},
            **kwargs,
    ):
        import frb.halos.models as halos
        # from frb.mw import haloDM

        print(kwargs)

        if isinstance(model, str):
            model = model.lower()
            halo_models = {
                "yf17": halos.YF17,
                "pz19": halos.MilkyWay,
                "mb04": halos.MB04,
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
                    halo_model=halo_models[model](**model_kwargs),
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
            **model_kwargs
    ):
        """

        :param distance: Distance from MW centre to which to evaluate DM. If a non-Quantity number is passed, it will be
            interpreted as a multiple of the model's virial radius (R200).
        :param zero_distance: The distance to which to zero the inner volume of the halo
        :param halo_model: Halo model to evaluate.
        :return:
        """

        print(model_kwargs)

        from ne2001 import density
        if halo_model is None:
            from frb.halos.models import MilkyWay
            halo_model = MilkyWay(**model_kwargs)
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
        if z_max <= 0:
            return 0 * dm_units
        else:
            return average_DM(z_max, cosmo=cosmology, **kwargs)

    def dm_halos_avg(self, z_max: float = None, **kwargs):
        import frb.halos.hmf as hmf
        from frb.dm.igm import average_DMhalos
        hmf.init_hmf()
        if z_max is None:
            if self.host_galaxy.z is not None:
                z_max = self.host_galaxy.z
            else:
                return -999 * dm_units
        if z_max <= 0:
            return 0 * dm_units
        else:
            return average_DMhalos(z_max, cosmo=cosmology, **kwargs)

    # def estimate_dm_excess(self):
    #     dm_ism = self.estimate_dm_mw_ism()
    #     dm_cosmic = self.estimate_dm_cosmic()
    #     dm_halo = 60 * dm_units
    #     return self.dm - dm_ism - dm_cosmic - dm_halo

    def dm_host_from_tau(
            self,
            afg: Union[units.Quantity, float],
            z_host: float = None,
            subtract_mw: bool = True
    ):
        """
        Implements an inverted form of Equation 9, equivalent to Equation 23, of
        Cordes et al 2022 (https://www.doi.org/10.3847/1538-4357/ac6873)
        Note that the quantity returned is in the HOST FRAME.

        :param afg: $A_\tau \widetilde{F} G$, as described in Cordes et al 2022; these parameters, related to the
            geometry of the source, host, and ISM, are usually constrained in combination.
        :param z_host: The redshift of the host galaxy. If not given, will attempt to use the `self.host_galaxy.z`
        :return: Estimated DM_host, in the host rest frame.
        """
        if z_host is None:
            z_host = self.host_galaxy.z
        if z_host is None:
            return 0 * dm_units
        if self.nu_scattering:
            nu = u.check_quantity(self.nu_scattering, units.MHz)
        else:
            return 0 * dm_units
        if self.tau:
            tau = u.check_quantity(self.tau * 1., units.ms)
        else:
            return 0 * dm_units
        if subtract_mw:
            tau_mw, _ = self.tau_mw()
            tau -= tau_mw
        afg_unit = units.pc ** (-2 / 3) * units.km ** (-1 / 3)
        afg = u.check_quantity(afg, afg_unit)
        nu_4 = (nu / units.GHz) ** 4
        z_3 = (1 + z_host) ** 3
        tau_norm = 0.48 * units.ms
        dm = (100 * dm_units * (tau * nu_4 * z_3 * (afg_unit / (tau_norm * afg))) ** (1. / 2.))
        return dm.to(dm_units)

    def tau_mw(
            self,
            x_tau: float = 4.
    ):
        """
        Implements Equation 8 of
        Cordes et al 2022 (https://www.doi.org/10.3847/1538-4357/ac6873)

        :param x_tau:
        :return:
        """
        dm = self.dm_mw_ism_ne2001()
        nu = self.nu_scattering

        tau_dm_mw = 1.9 * 10e-7 * units.ms * (nu / units.GHz) ** x_tau * (dm / dm_units) ** 1.5 * (
                1 + 3.55e-5 * (dm / dm_units) ** 3)
        tau_dm_mw = tau_dm_mw.to(units.ms)

        log_tau = np.log10(tau_dm_mw.value)
        log_tau_err = 0.76
        tau_dm_mw_err = u.uncertainty_power_2(x=log_tau, base=10., sigma_x=log_tau_err)

        return tau_dm_mw, tau_dm_mw_err * units.ms

    def tau_mw_halo(
            self,
            f=0.03 * units.pc ** -(2/3) * units.km ** (-1/3),
            a_t=1.,
            nu: units.Quantity[units.MHz] = None,
            dm_mw_halo=40 * dm_units
    ):
        """
        Uses equation 1 of Ocker+2021 (https://doi.org/10.3847/1538-4357/abeb6e) and their limit of F for the MW halo
        (0.03). It should be noted that this represents an upper limit on scattering.

        :param f:
        :param a_t:
        :param nu:
        :param dm_mw_halo:
        :return:
        """
        if nu is None:
            nu = self.nu_scattering
        return u.tau(
            a_t=a_t,
            f=f,
            nu=nu,
            g_scatt=1, # For observer embedded in the medium and source distance >> l,
            dm=dm_mw_halo
        )

    def tau_from_halo(
            self,
            halo: 'frb.halos.models.ModifiedNFW',
            f,
            r_perp,
            nu: units.Quantity[units.MHz] = None,
            dm_kwargs: dict = {},
            a_t=1.,
    ):
        """
        Encodes equation 2 of Ocker+2021 (https://doi.org/10.3847/1538-4357/abeb6e)

        :param halo:
        :param f:
        :param r_perp:
        :param nu:
        :param dm_kwargs:
        :param kwargs:
        :return:
        """

        if nu is None:
            nu = self.nu_scattering

        d_sl = cosmology.angular_diameter_distance_z1z2(halo.z, self.z)
        d_lo = cosmology.angular_diameter_distance(halo.z)
        d_so = cosmology.angular_diameter_distance(self.z)
        dm_halo = halo.Ne_Rperp(
            r_perp,
            **dm_kwargs
        ).value / (1 + halo.z)

        return u.tau_cosmological(
            a_t=a_t,
            f=f,
            dm=dm_halo,
            z_l=halo.z,
            nu=nu,
            d_sl=d_sl,
            d_lo=d_lo,
            d_so=d_so,
            l=2 * np.sqrt(halo.r200**2 - r_perp**2)
        )

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
            rmax: float = 1.,
            neval_cosmic: int = 1000,
            step_size_halo: units.Quantity = 0.1 * units.kpc,
            dm_host_ism: units.Quantity = 0,
            dm_host_ism_err: units.Quantity = 0,
            skip_other_models: bool = False,
            **halo_kwargs
    ):

        dm_host_ism = u.check_quantity(dm_host_ism, dm_units)

        if skip_other_models:
            outputs = {
                "dm_halo_mw_pz19": self.dm_mw_halo(distance=rmax, model="pz19")
            }
        else:
            outputs = self.dm_mw_halo(distance=rmax, model="all")

        host = self.host_galaxy

        # frb_err_ra, frb_err_dec = self.position_err.uncertainty_quadrature_equ()
        # frb_err_dec = frb_err_dec.to(units.arcsec)

        print("DM_FRB:", self.dm, "+/-", self.dm_err)
        print("tau_FRB:", self.tau, "+/-", self.tau_err, "at", self.nu_scattering)

        print("Milky Way:")

        print("\tISM:")
        # outputs["dm_ism_mw_cum"] = self.estimate_dm_mw_ism_cum(max_dm=outputs["dm_ism_mw_ne2001"] - 0.5 * dm_units)
        outputs["dm_ism_mw_ne2001"] = self.dm_mw_ism_ne2001()
        outputs["tau_ism_mw_ne2001"] = self._tau_mw_ism_ne2001
        print("\t\tDM_MWISM_NE2001:", outputs["dm_ism_mw_ne2001"])
        print("\t\ttau_MWISM_NE2001:", outputs["tau_ism_mw_ne2001"])

        outputs["dm_ism_mw_ymw16"] = self.dm_mw_ism_ymw16()
        outputs["tau_ism_mw_ymw16"] = self._tau_mw_ism_ymw16
        print("\t\tDM_MWISM_YMW16:", outputs["dm_ism_mw_ymw16"])
        print("\t\ttau_MWISM_YMW16:", outputs["tau_ism_mw_ymw16"])

        outputs["dm_ism_mw_err"] = np.abs(outputs["dm_ism_mw_ne2001"] - outputs["dm_ism_mw_ymw16"])

        outputs["tau_ism_mw_c22"], outputs["tau_ism_mw_c22_err"] = self.tau_mw()
        print("\t\ttau_MWISM_C22:", outputs["tau_ism_mw_c22"], "+/-", outputs["tau_ism_mw_c22_err"])

        print("\tDM_MWHalo_PZ19:", outputs["dm_halo_mw_pz19"])

        if not skip_other_models:
            print("\tDM_MWHalo_YF17:", outputs["dm_halo_mw_yf17"])
            print("\tDM_MWHalo_MB15:", outputs["dm_halo_mw_mb15"])

        outputs["tau_halo_mw"] = self.tau_mw_halo()
        print("\ttau_MWHalo: <", outputs["tau_halo_mw"])

        print("\tDM_MW:")
        outputs["dm_mw"] = outputs["dm_halo_mw_pz19"] + outputs["dm_ism_mw_ne2001"]
        outputs["dm_mw_err"] = outputs["dm_ism_mw_err"]
        print("\t", outputs["dm_mw"])

        print("DM_exgal:")
        outputs["dm_exgal"] = self.dm - outputs["dm_mw"]
        outputs["dm_exgal_err"] = np.sqrt(self.dm_err**2 + outputs["dm_mw_err"]**2)
        print(outputs["dm_exgal"])

        if host.z is not None:
            z_max = host.z
        else:
            z_max = 2.0

        # if not do_mc:
        print("Avg DM_cosmic:")
        cosmic_tbl = table.QTable()
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

        fg_outputs, halo_tbl = self.foreground_halos(
            skip_other_models=skip_other_models,
            step_size_halo=step_size_halo,
            rmax=rmax,
            cosmic_tbl=cosmic_tbl,
            **halo_kwargs
        )

        outputs.update(
            fg_outputs
        )
        # outputs["dm_halos_yf17"] = halo_tbl["dm_halo_yf17"].nansum() - dm_halo_host
        # outputs["dm_halos_mb04"] = halo_tbl["dm_halo_mb04"].nansum() - dm_halo_host
        # outputs["dm_halos_mb15"] = halo_tbl["dm_halo_mb15"].nansum() - dm_halo_host

        print("\t", outputs["dm_halos_emp"])
        # if not do_mc:
        print("\tEmpirical DM_cosmic:")
        outputs["dm_cosmic_emp"] = outputs["dm_igm"] + outputs["dm_halos_emp"]
        print("\t", outputs["dm_cosmic_emp"])

        outputs["dm_host_median"] = 0 * dm_units
        if host.z is not None:
            print("DM_host:")
            # Obtained using James 2021
            print("\tMedian DM_host (James+2022B) at z:")
            outputs["dm_host_median_james22A"] = dm_host_median["james_22A"] / (1 + host.z)
            outputs["dm_host_median"] = dm_host_median["james_22B"] / (1 + host.z)
            print("\t", outputs["dm_host_median"])
            # print("\tMax-probability DM_host:")
            # outputs["dm_host_max_p_james22A"] = 98 * dm_units / (1 + host.z)
            # print("\t", outputs["dm_host_max_p"])

            print("\tDM_halo_host")
            print("\t", outputs["dm_halo_host"])

            if dm_host_ism > 0:
                print("\tDM_host,ism (given):")
                outputs["dm_host_ism"] = dm_host_ism
            else:
                print("\tDM_host,ism (estimated from median):")
                outputs["dm_host_ism"] = outputs["dm_host_median"] - outputs["dm_halo_host"]
            outputs["dm_host_ism_err"] = dm_host_ism_err
            print("\t", outputs["dm_host_ism"], "+/-", outputs["dm_host_ism_err"])

            outputs["dm_host"] = outputs["dm_host_ism"] + outputs["dm_halo_host"]
            print("\tDM_host:", outputs["dm_host"])

            outputs["dm_host_tau"] = self.dm_host_from_tau(1)
            print("\tDM_host_tau:", outputs["dm_host_tau"])

        # if not do_mc:
        print("Excess DM estimate:")
        outputs["dm_excess_avg"] = self.dm - outputs["dm_cosmic_avg"] - outputs["dm_mw"] - outputs["dm_host"]
        print("\t", outputs["dm_excess_avg"])

        print("Empirical Excess DM:")
        outputs["dm_excess_emp"] = self.dm - outputs["dm_cosmic_emp"] - outputs["dm_mw"] - outputs["dm_host"]
        print("\t", outputs["dm_excess_emp"])

        #     r_eff_proj = foreground.projected_distance(r_eff).to(units.kpc)
        #     r_eff_proj_err = foreground.projected_distance(r_eff_err).to(units.kpc)
        #     print("Projected effective radius:")
        #     print(r_eff_proj, "+/-", r_eff_proj_err)
        #     print("FG-normalized offset:")
        #     print(offset / r_eff_proj)

        outputs["halo_table"] = halo_tbl

        # if not do_mc:
        outputs["dm_cum_table"] = cosmic_tbl

        return outputs

    def foreground_halos(
            self,
            foreground_objects: list = None,
            step_size_halo: units.Quantity = 0.1 * units.kpc,
            rmax=1.,
            fhot=0.75,
            load_objects: bool = True,
            skip_other_models: bool = False,
            do_mc: bool = False,
            cat_search: str = None,
            cosmic_tbl: table.QTable = None,
            smhm_relationship: str = "K18",
            do_profiles: bool = True,
            do_incidence: bool = False,
    ):

        from .galaxy import Galaxy
        from frb.halos.hmf import halo_incidence

        host = self.host_galaxy
        if load_objects:
            self.field.load_all_objects()

        if do_mc:
            position_frb = self.position_err.mc_ellipse(n_samples=1)[0]
            # if dm_host_ism > 0:
            #     dm_host_ism = np.random.normal(loc=dm_host_ism.value, scale=dm_host_ism_err.value) * dm_units
            #     dm_host_ism_err = 0 * dm_units
        else:
            position_frb = self.position

        outputs = {"do_mc": do_mc}
        halo_inform = []
        halo_models = {}
        halo_profiles = {}
        dm_halo_host = 0. * dm_units
        dm_halo_cum = {}
        if not do_mc and cosmic_tbl is not None:
            cosmic_tbl["dm_halos_emp"] = cosmic_tbl["dm_halos_avg"] * 0

        if foreground_objects is None:
            foreground_objects = list(
                filter(
                    lambda o: isinstance(o, Galaxy) and o.z <= self.host_galaxy.z and o.log_mass_stellar is not None,
                    self.field.objects.values()
                )
            )
        # if host not in foreground_objects and host.z is not None:
        #     foreground_objects.append(host)

        for obj in foreground_objects:
            print(f"\tDM_halo_{obj.name}: ({obj.z=})")
            # if load_objects:
            #     obj.load_output_file()

            # obj.select_deepest()
            # obj.position_photometry = None

            # print("\t\t Drawing position.")

            pos, fg_pos_err = obj.get_position()
            if do_mc:
                pos = fg_pos_err.mc_ellipse(n_samples=1)[0]
                # fg_pos_err = 0 * units.arcsec
            # else:
            #     gal_ra_err, gal_dec_err, _ = fg_pos_err.uncertainty_quadrature()
            #     # fg_pos_err = max(gal_ra_err, gal_dec_err)

            halo_info = {
                "id": obj.name,
                "z": obj.z,
                "ra": pos.ra,
                "dec": pos.dec,
            }

            if not do_mc and load_objects:
                obj.load_output_file()
                props, _ = obj.assemble_row()
                halo_info.update(props)

            if cat_search is not None:
                # print("\t\t Searching catalogue.")

                cat_row, sep = obj.find_in_cat(cat_search)
                if sep < 1 * units.arcsec:
                    halo_info["id_cat"] = cat_row["objName"]
                    halo_info["ra_cat"] = cat_row["raStack"]
                    halo_info["dec_cat"] = cat_row["decStack"]
                else:
                    halo_info["id_cat"] = "--"
                halo_info["offset_cat"] = sep.to(units.arcsec)

            halo_info["offset_angle"] = offset_angle = position_frb.separation(pos).to(units.arcsec)

            if not do_mc:
                halo_info["distance_angular_size"] = obj.angular_size_distance()
                halo_info["distance_luminosity"] = obj.luminosity_distance()
                halo_info["distance_comoving"] = obj.comoving_distance()
            # if not do_mc:
            #     halo_info["offset_angle_err"] = offset_angle_err = np.sqrt(fg_pos_err ** 2 + frb_err_dec ** 2)

            halo_info["r_perp"] = offset = obj.projected_size(offset_angle).to(units.kpc)
            # if not do_mc:
            #     halo_info["r_perp_err"] = obj.projected_size(offset_angle_err).to(units.kpc)

            if do_mc:
                print(f"\t\t Drew R_perp = {offset}")
                err_this = max(obj.log_mass_stellar_err_plus, obj.log_mass_stellar_err_minus)
                fg_logm_star = np.random.normal(
                    loc=obj.log_mass_stellar,
                    scale=err_this,
                )
                print(f"\t\t Drew log(M_star) = {fg_logm_star} (measured {obj.log_mass_stellar} +/- {err_this})")
                obj.log_mass_stellar = fg_logm_star

            else:
                fg_logm_star = obj.log_mass_stellar
                halo_info["log_mass_stellar_err_plus"] = fg_logm_star_err_plus = obj.log_mass_stellar_err_plus
                halo_info["log_mass_stellar_err_minus"] = fg_logm_star_err_minus = obj.log_mass_stellar_err_minus
                halo_info["mass_stellar_err_plus"] = u.uncertainty_power_2(
                    x=fg_logm_star,
                    sigma_x=fg_logm_star_err_plus,
                    base=10
                )
                halo_info["mass_stellar_err_minus"] = u.uncertainty_power_2(
                    x=fg_logm_star,
                    sigma_x=fg_logm_star_err_minus,
                    base=10
                )
            halo_info["log_mass_stellar"] = fg_logm_star
            halo_info["mass_stellar"] = units.solMass * 10 ** fg_logm_star
            # try:
            #     halo_info["log_mass_stellar"] = np.log10(fg_logm_star / units.solMass)
            # except units.UnitTypeError:
            #     continue
            # print(fg_logm_star, "+", fg_logm_star_err_plus, "-", fg_logm_star_err_minus)

            obj.halo_mass(relationship=smhm_relationship, do_mc=do_mc)

            halo_info["mass_halo"] = obj.mass_halo
            # halo_info["mass_halo_err"] = obj.mass_halo_err
            halo_info["log_mass_halo"] = obj.log_mass_halo
            halo_info["log_mass_halo_err"] = obj.log_mass_halo_err

            halo_info["h"] = obj.h()
            halo_info["c200"] = obj.halo_concentration_parameter()

            mnfw = obj.halo_model_mnfw(f_hot=fhot)
            if not skip_other_models and not do_mc:
                yf17 = obj.halo_model_yf17()
                mb04 = obj.halo_model_mb04()
                mb15 = obj.halo_model_mb15()

            if not do_mc and do_profiles:
                # This is for building a profile of DM with R_perp
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
                halo_profiles[obj.name] = halo_nes

            # halo_info["r_lim"] = r_perp

            print("\t\t Calculating DM_halo.")
            print(f"\t\t\t {offset=}")
            print(f"\t\t\t {rmax=}")
            print(f"\t\t\t {step_size_halo=}")
            halo_info["dm_halo"] = dm_halo = mnfw.Ne_Rperp(
                Rperp=offset,
                rmax=rmax,
                step_size=step_size_halo
            ) / (1 + obj.z)

            if not skip_other_models:
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

            if host.z is not None and do_incidence and not do_mc:
                halo_info["n_intersect_greater"] = halo_incidence(
                    Mlow=obj.mass_halo.value,
                    zFRB=host.z,
                    radius=halo_info["r_perp"]
                )

            if host.z is not None and do_incidence and not do_mc:
                m_low = 10 ** (np.floor(obj.log_mass_halo))
                m_high = 10 ** (np.ceil(obj.log_mass_halo))
                if m_low < 2e10:
                    m_high += 2e10 - m_low
                    m_low = 2e10

                halo_info["mass_halo_partition_high"] = m_high * units.solMass
                halo_info["mass_halo_partition_low"] = m_low * units.solMass

                halo_info["log_mass_halo_partition_high"] = np.log10(m_high)
                halo_info["log_mass_halo_partition_low"] = np.log10(m_low)
                halo_info["n_intersect_partition"] = halo_incidence(
                    Mlow=m_low,
                    Mhigh=m_high,
                    zFRB=host.z,
                    radius=halo_info["r_perp"]
                )

            # if obj.cigale_results is not None:
            #     halo_info["u-r"] = obj.cigale_results["bayes.param.restframe_u_prime-r_prime"] * units.mag
            # if obj.sfr is not None:
            #     halo_info["sfr"] = obj.sfr
            # if obj.sfr_err is not None:
            #     halo_info["sfr_err"] = obj.sfr_err

            if halo_info["dm_halo"] > 0. * dm_units and not do_mc:
                dm_halo_cum_this = obj.halo_dm_cum(
                    rmax=rmax,
                    rperp=offset,
                    step_size=step_size_halo
                )

                if obj is not host:
                    dm_halo_cum[obj.name] = dm_halo_cum_this
                    if not do_mc and cosmic_tbl is not None:
                        cosmic_tbl["dm_halos_emp"] += np.interp(
                            cosmic_tbl["comoving_distance"],
                            dm_halo_cum_this["d_abs"],
                            dm_halo_cum_this["DM"]
                        )
                # Add a cumulative halo host DM to the cosmic table
                else:
                    dm_halo_cum[obj.name] = dm_halo_cum_this[:len(dm_halo_cum_this) // 2]
                    if not do_mc and cosmic_tbl is not None:
                        cosmic_tbl["dm_halo_host"] = np.interp(
                            cosmic_tbl["comoving_distance"],
                            dm_halo_cum_this["d_abs"],
                            dm_halo_cum_this["DM"]
                        )

        #         plt.plot(rs, halo_nes)
        #         plt.plot([offset.value, offset.value], [0, max(halo_nes)])

        halo_tbl = table.QTable(halo_inform)
        halo_tbl["dm_halo"] = halo_tbl["dm_halo"].to(dm_units)
        if not do_mc and cosmic_tbl is not None:
            cosmic_tbl["dm_cosmic_emp"] = cosmic_tbl["dm_halos_emp"] + cosmic_tbl["dm_igm"]

        print("\tEmpirical DM_halos:")
        outputs["dm_halos_inclusive"] = halo_tbl["dm_halo"].nansum()
        outputs["dm_halos_emp"] = outputs["dm_halos_inclusive"] - dm_halo_host

        outputs["dm_halo_host"] = dm_halo_host
        outputs["halo_models"] = halo_models
        outputs["halo_dm_profiles"] = halo_profiles
        outputs["halo_dm_cum"] = dm_halo_cum

        if do_mc:
            self.field.gather_objects()

        return outputs, halo_tbl

    def host_probability_unseen(
            self,
            img: 'image.ImagingImage',
            sample: Union[str, 'sed.SEDSample'] = "Gordon+2023",
            **kwargs
    ):
        """Uses the Marnoch+2023 method to estimate the probability that this FRB's host galaxy is unseen in the given
        image.

        :param img:
        :param sample:
        :param kwargs:
        :return:
        """

        if self.read_p_z_dm() is None:
            return None, None, None
        if isinstance(sample, str):
            sample = sed.SEDSample.from_params("Gordon+2023")
        if "save_memory" not in kwargs:
            kwargs["save_memory"] = True
        if "n_z" not in kwargs:
            kwargs["n_z"] = 500
        if "z_max" not in kwargs:
            kwargs["z_max"] = 5  # (self.dm / 1000).value * 2
        sample.z_displace_sample(
            bands=[img.filter],
            **kwargs
        )
        psf = img.extract_psf_fwhm()
        limits = img.test_limit_location(self.position, ap_radius=2 * psf)
        lim_5sigma = limits["mag"][4]
        a_z = self.galactic_extinction(img.filter)
        lim_w_ext = lim_5sigma - a_z
        print("Using limit", lim_w_ext)
        vals, tbl, z_lost = sample.probability_unseen(
            band=img.filter,
            limit=lim_w_ext,
            obj=self,
            # show=True,
            plot=True,
            output=os.path.join(self.data_path, "PATH", "p_u")
        )
        vals["limit_ext_correct"] = lim_w_ext
        vals["limit_measured"] = lim_5sigma
        vals["extinction"] = a_z
        return vals, tbl, z_lost

    def read_p_z_dm(self, path: str = None):
        if path is None:
            path = os.path.join(p.data_dir, "zdm", f"{self.name}_pzgdm.npz")
        if not os.path.isfile(path):
            return None
        zdm_np = np.load(path)
        self.zdm_table = table.QTable(
            {
                "z": zdm_np["zvals"],
                "p(z|DM)_best": zdm_np["all_pzgdm"][0]
            }
        )
        for i, zdm in enumerate(zdm_np["all_pzgdm"][1:]):
            self.zdm_table[f"p(z|DM)_90_{i}"] = zdm
        return self.zdm_table

    def _p_z_dm_command(self):
        dm_ism = self.dm_mw_ism_ne2001().value
        command = f"python pz_given_dm.py -d {self.dm.value} -i {dm_ism} -s CRAFT/ICS -H 40 -o {self.name_filesys}_pzgdm "
        if self.snr is not None:
            command += f"-S {self.snr} "
        if self.width_total is not None:
            command += f"-w {self.width_total.value} "

        z = None
        if self.z:
            z = self.z
        elif self.host_galaxy and self.host_galaxy.z:
            z = self.host_galaxy.z
        if z is not None:
            command += f"-z {z} "
        return command

    @classmethod
    def from_dict(cls, dictionary: dict, **kwargs) -> 'FRB':
        frb = super().from_dict(dictionary=dictionary, **kwargs)
        # if "dm" in dictionary:
        #     frb.dm = u.check_quantity(dictionary["dm"], dm_units)
        dictionary["host_galaxy"]["transient"] = frb
        host_galaxy = TransientHostCandidate.from_dict(dictionary=dictionary["host_galaxy"])
        frb.host_galaxy = host_galaxy
        return frb
