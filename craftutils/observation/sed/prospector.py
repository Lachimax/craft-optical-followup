import os

import numpy as np

import astropy.table as table
import astropy.units as units

import craftutils.utils as u

from .sed import *

__all__ = []


@u.export
class GordonProspectorModel(SEDModel):
    """
    The `GordonProspectorModel` uses the data model established by [GordonProspector]_.
    """

    def load_data(self):
        flux = np.loadtxt(self.path + '_spectrum.txt') * units.microjansky
        wave = np.loadtxt(self.path + '_wavelengths.txt') * units.Angstrom
        obs_flux_path = self.path + '_obs_flux.txt'
        if os.path.isfile(obs_flux_path):
            obs_flux = np.loadtxt(obs_flux_path)
        else:
            obs_flux = []
        obs_wave_path = self.path + '_obs_wave.txt'
        if os.path.isfile(obs_wave_path):
            obs_wave = np.loadtxt(obs_wave_path)
        else:
            obs_wave = []
        self.model_table = table.QTable(
            {
                "wavelength": wave,
                "flux_nu": flux,
            }
        )
        self.obs_table = table.QTable(
            {
                "wavelength": obs_wave,
                "flux_nu": obs_flux,
            }
        )
        self.prep_columns()
        self.luminosity_per_wavelength()
