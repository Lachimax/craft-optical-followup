import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.table as table
import astropy.units as units
from astropy.cosmology import WMAP9

import craftutils.utils as u

from .sed import SEDModel

__all__ = []


@u.export
class GordonProspectorModel(SEDModel):
    """
    The `GordonProspectorModel` uses the data model established by [GordonProspector]_.
    """
    default_cosmology = WMAP9

    def __init__(self, **kwargs):

        path = ""
        if "path" in kwargs:
            path = kwargs["path"]

        self.model_flux_path = None
        expect = path + '_model_spectrum_FM07.txt'
        if "model_flux_path" in kwargs:
            self.model_flux_path = kwargs["model_flux_path"]
        elif path != "" and os.path.isfile(expect):
            self.model_flux_path = expect
        # else:
        #     raise ValueError("model_flux_path not given, and no file found at", expect)

        self.model_wavelength_path = None
        expect = path + '_model_wavelengths_FM07.txt'
        if "model_wavelength_path" in kwargs:
            self.model_wavelength_path = kwargs["model_wavelength_path"]
        elif path != "" and os.path.isfile(expect):
            self.model_wavelength_path = expect
        # else:
        #     raise ValueError("model_wavelength_path not given, and no file found at", expect)

        self.observed_flux_path = None
        expect = path + '_observed_spectrum_FM07.txt'
        if "observed_flux_path" in kwargs:
            self.observed_flux_path = kwargs["observed_flux_path"]
        elif path != "" and os.path.isfile(path + expect):
            self.observed_flux_path = path + expect

        self.observed_wavelength_path = None
        expect = path + '_obsserved_wave_FM07.txt'
        if "observed_wavelength_path" in kwargs:
            self.observed_wavelength_path = kwargs["observed_wavelength_path"]
        elif path != "" and os.path.isfile(path + expect):
            self.observed_wavelength_path = path + expect

        self.observed_flux_err_path = None
        if "observed_flux_err_path" in kwargs:
            self.observed_flux_err_path = kwargs["observed_flux_err_path"]


        super().__init__(**kwargs)
        if self.model_wavelength_path is not None and self.model_flux_path is not None:
            self.load_data()

    def prep_data(self):
        super().prep_data()
        self.luminosity_per_wavelength()

    def load_data(self):
        flux = np.loadtxt(self.model_flux_path) * units.microjansky
        wave = np.loadtxt(self.model_wavelength_path) * units.Angstrom

        if self.observed_flux_path is not None and os.path.isfile(self.observed_flux_path):
            obs_flux = np.loadtxt(self.observed_flux_path) * units.microjansky
        else:
            obs_flux = []
            self.observed_flux_path = None

        if self.observed_wavelength_path is not None and os.path.isfile(self.observed_wavelength_path):
            obs_wave = np.loadtxt(self.observed_wavelength_path) * units.angstrom
        else:
            obs_wave = []
            self.observed_wavelength_path = None

        if self.observed_flux_err_path is not None and os.path.isfile(self.observed_flux_err_path):
            obs_flux_err = np.loadtxt(self.observed_flux_err_path) * units.microjansky
        else:
            obs_flux_err = []
            self.observed_flux_err_path = None

        self.model_table = table.QTable(
            {
                "wavelength": wave,
                "flux_nu": flux,
            }
        )
        obs_dict = {
            "wavelength": obs_wave,
            "flux_nu": obs_flux,
        }
        if len(obs_flux_err) == len(obs_flux):
            obs_dict["err_flux_nu"] = obs_flux_err
        self.obs_table = table.QTable(
            obs_dict
        )
        self.prep_data()
