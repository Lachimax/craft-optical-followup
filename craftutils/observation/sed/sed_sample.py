import os
from typing import Union, List, Dict

import numpy as np
import matplotlib.pyplot as plt

import astropy.table as table
import astropy.units as units
import astropy.cosmology as cosmology
from astropy.modeling import models, fitting


import craftutils.utils as u
import craftutils.observation.filters as fil
import craftutils.observation.objects as objects
from craftutils.plotting import tick_fontsize, axis_fontsize, lineweight
from craftutils.photometry import distance_modulus
from craftutils.observation.sed import SEDModel


class SEDSample:
    default_cosmology = cosmology.Planck18
    """
    A class representing a sample of SED models, for doing bulk population calculations.
    """

    def __init__(self, **kwargs):
        self.model_dict: Dict[str, SEDModel] = {}
        self.model_directory: str = None
        self.output_dir: str = None
        self.z_mag_tbls: Dict[str, table.QTable] = {}
        self.z_mag_tbl_paths: Dict[str, str] = {}
        for key, item in kwargs.items():
            setattr(self, key, item)

    def add_model(self, model: SEDModel, name: str = None):
        if name is None:
            if model.name is None:
                raise ValueError("If model.name is not set, a name must be provided.")
            name = model.name
        self.model_dict[name] = model

    def z_displace_sample(
            self,
            bands: List[fil.Filter],
            save_memory: bool = False,
            z_min: float = 0.01,
            z_max: float = 3.0,
            n_z: int = 20,
    ):
        """
        For all of the sample, displaces the models to a range of redshifts.

        :param bands: List or other iterable of `craftutils.observation.filters.Filter` objects through which to observe
            the redshifted galaxy.
        :param save_memory: If True, deletes the z-displaced models from memory after the useful information is
            extracted.
        :param z_min: Minimum redshift to take measurements at.
        :param z_max: Maximum redshift to take measurements at.
        :param n_z: number of redshifts to take measurements at.
        :param output:
        :return:
        """

        # Empty dict of dicts
        band_dicts = dict(map(lambda b: (b.machine_name(), {}), bands))

        for i, sed_name in enumerate(self.model_dict):
            print(f"Processing {i + 1} / {len(self.model_dict)}:", sed_name, "...")
            model = self.model_dict[sed_name]
            mag_tbl = model.z_mag_table(
                z_min=z_min,
                z_max=z_max,
                n_z=n_z,
                bands=bands,
                include_actual=False
            )

            if save_memory:
                model.expunge_shifted_models()

            for band_name, band_dict in band_dicts.items():
                band_dict[sed_name] = mag_tbl[band_name]

        zs = list(np.linspace(z_min, z_max, n_z))

        for band_name, band_dict in band_dicts.items():
            band_dict["z"] = zs
            self.z_mag_tbls[band_name] = table.QTable(band_dict)

        return self.z_mag_tbls

    def write_z_mag_tbls(self, directory: str = None):
        if directory is None:
            directory = os.path.join(self.output_dir, "z_mag_tables")
        u.mkdir_check(directory)
        for band_name, z_mag_tbl in self.z_mag_tbls.items():
            path = os.path.join(directory, f"{band_name}_z_mag_table.ecsv")
            z_mag_tbl.write(path, overwrite=True)
            self.z_mag_tbl_paths[band_name] = path

    def calculate_for_limit(
            self,
            band: Union[str, fil.Filter],
            limit: units.Quantity,
            output: str = None,
    ):
        # limit = u.check_quantity(limit, units.mag)
        if isinstance(band, fil.Filter):
            band_name = band.machine_name()
        else:
            band_name = band
        tbl = self.z_mag_tbls[band_name].copy()
        # Set up some columns
        tbl.add_column(np.zeros(len(tbl), dtype=int), name="n>lim")
        tbl.add_column(np.zeros(len(tbl), dtype=int), name="n<lim")
        z_lost = {}
        columns = []
        for colname in self.model_dict:
            if colname in tbl.colnames:
                tbl["n>lim"] += tbl[colname] > limit
                tbl["n<lim"] += tbl[colname] < limit
                print(colname)
                if len(tbl["z"][tbl[colname] > limit]) > 0:
                    z_lost[colname] = np.min(tbl["z"][tbl[colname] > limit])
                else:
                    z_lost[colname] = None
                columns.append(colname)
        # Record the mean and median magnitudes at each z.
        tbl["mean"] = np.zeros(len(tbl))
        tbl["median"] = np.zeros(len(tbl))
        for row in tbl:
            row["mean"] = np.mean(list(row[columns]))
            row["median"] = np.median(list(row[columns]))
        z_lost["mean"] = np.min(tbl["z"][tbl["mean"] > limit])
        z_lost["median"] = np.min(tbl["z"][tbl["median"] > limit])

        # Calculate P(U|z) as the fraction of hosts that are unseen at a given redshift.
        tbl["P(U|z)"] = tbl["n>lim"] / len(columns)
        # Some extra values
        tbl["d_L"] = cosmology.WMAP9.luminosity_distance(tbl["z"])
        tbl["mu"] = distance_modulus(tbl["d_L"])

        if isinstance(output, str):
            tbl.write(output, overwrite=True)

        return tbl, z_lost

    def probability_unseen(
            self,
            band: Union[str, fil.Filter],
            limit: units.Quantity,
            p_z_dm: Union[table.Table, np.ndarray],
            z: np.ndarray = None,
            output: str = None,
            plot: bool = False,
            show: bool = False
    ):

        if output:
            u.mkdir_check(output)

        if isinstance(p_z_dm, np.ndarray):
            if z:
                p_z_dm = table.QTable(
                    {
                        "z": z,
                        "p(z|DM)": p_z_dm
                    }
                )
            else:
                raise TypeError("z must be provided if p_z_dm is not a Table.")

        tbl, z_lost = self.calculate_for_limit(
            band=band,
            limit=limit,
            output=os.path.join(output, "z_table.ecsv")
        )

        tbl["p(z|DM)"] = np.interp(
            x=tbl["z"],
            xp=p_z_dm["z"],
            fp=p_z_dm["p(z|DM)"]
        )
        # Calculate P(U) and, at the same time, get p(z|U,DM)
        curve = tbl["P(U|z)"] * tbl["p(z|DM)"]
        p_u = np.trapz(
            y=curve,
            x=tbl["z"]
        )
        tbl["p(z|U,DM)"] = curve / p_u
        tbl["P(U|z) * p(z|DM)"] = curve

        # Add normal approximation
        gauss_init = models.Gaussian1D(mean=1.2, amplitude=1.75, stddev=1.)
        fitter = fitting.LevMarLSQFitter()
        gauss_fit = fitter(gauss_init, x=tbl["z"], y=tbl["P(U|z) * p(z|DM)"])
        p_u_gauss = np.trapz(
            y=gauss_fit(tbl["z"]),
            x=tbl["z"]
        )
        tbl["p(z|U,DM) gauss"] = gauss_fit(tbl["z"]) / p_u_gauss

        print()
        print("Normal approximation")
        print("Peak p(z|U,DM) at z =", gauss_fit.mean.value)
        print("P(U) =", p_u_gauss)

        if isinstance(band, fil.Filter):
            band_name = band.machine_name()
        else:
            band_name = band

        if plot:
            fig, ax = plt.subplots()

            leg_x = 1.13

            ax_pdf = ax.twinx()
            ax_pdf.set_ylabel("Host fraction", rotation=-90, labelpad=35, fontsize=axis_fontsize)
            ax_pdf.tick_params(right=False, labelright=False)
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(axis="y", labelright=True, labelsize=tick_fontsize)
            ax.tick_params(axis="x", labelsize=tick_fontsize)

            # Do some plotting
            ax.plot(
                tbl["z"],
                tbl["P(U|z)"],
                label="$P(U|z) = N_\mathrm{unseen}(z)/N_\mathrm{hosts}$",
                # = \dfrac{N_\mathrm{unseen}(z)}{N_\mathrm{hosts}}$"
                lw=2,
                c="cyan"
            )

            ax.set_xlabel("$z$")
            ax.set_xlim(0., np.max(tbl["z"]))
            fig.savefig(
                os.path.join(output, f"probability_{objects.cosmology.name}_{band_name}_steps_only.pdf"),
                bbox_inches="tight"
            )
            fig.savefig(
                os.path.join(output, f"probability_{objects.cosmology.name}_{band_name}_steps_only.png"),
                bbox_inches="tight",
                dpi=200
            )

            ax.set_ylabel("Probability density", fontsize=axis_fontsize)

            ax.plot(
                tbl["z"],
                tbl["p(z|DM)"],
                label="$p(z|\mathrm{DM})$",
                lw=2,
                c="purple"
            )
            ax.legend(
                loc=(leg_x, 0),
            )
            fig.savefig(
                os.path.join(output, f"probability_{objects.cosmology.name}_{band_name}_combined.pdf"),
                bbox_inches="tight"
            )
            fig.savefig(
                os.path.join(output, f"probability_{objects.cosmology.name}_{band_name}_combined.png"),
                bbox_inches="tight",
                dpi=200
            )

            ax.plot(
                tbl["z"],
                tbl["p(z|U,DM)"],
                label="$p(z|U,\mathrm{DM})$",  # = \dfrac{P(U|z)p(z|\mathrm{DM,etc.})}{P(U)}$"
                lw=2,
                c="green"
            )
            ax.legend(
                loc=(leg_x, 0)
            )
            fig.savefig(
                os.path.join(output, f"probability_{objects.cosmology.name}_{band_name}.pdf"),
                bbox_inches="tight"
            )
            fig.savefig(
                os.path.join(output, f"probability_{objects.cosmology.name}_{band_name}.png"),
                bbox_inches="tight",
                dpi=200
            )

            ax.plot(
                tbl["z"],
                tbl["p(z|U,DM) gauss"],
                label="$p(z|U,\mathrm{DM})$, Gaussian fit",  # = \dfrac{P(U|z)p(z|\mathrm{DM,etc.})}{P(U)}$"
                lw=2,
                c="darkorange",
                ls=":"
            )
            ax.legend(
                loc=(leg_x, 0),
                fontsize=tick_fontsize
            )
            fig.savefig(
                os.path.join(output, f"probability_{objects.cosmology.name}_{band_name}_gaussian.pdf"),
                bbox_inches="tight"
            )
            fig.savefig(
                os.path.join(output, f"probability_{objects.cosmology.name}_{band_name}_gaussian.png"),
                bbox_inches="tight", dpi=200
            )
            
            if show:
                plt.show(fig)

            plt.close(fig)

        p_u_dict = {"step": p_u, "gaussian": p_u_gauss}
        return p_u, tbl, z_lost
