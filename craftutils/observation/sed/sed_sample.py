import os
from typing import Union, List, Dict

import numpy as np

import astropy.table as table
import astropy.units as units

import craftutils.utils as u
import craftutils.observation.filters as fil
import craftutils.observation.objects as objects
from craftutils.observation.sed import SEDModel


class SEDSample:
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
            for key, ls in band_dict.items():
                print(key, len(ls))
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

    def calculate_for_limit(self, band: fil.Filter, limit: units.Quantity):
        pass

