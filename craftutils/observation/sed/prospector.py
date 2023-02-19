import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.table as table
import astropy.units as units

import craftutils.utils as u

from .sed import SEDModel

__all__ = []


@u.export
class GordonProspectorModel(SEDModel):
    """
    The `GordonProspectorModel` uses the data model established by [GordonProspector]_.
    """

    def __init__(self, **kwargs):

        path = ""
        if "path" in kwargs:
            path = kwargs["path"]

        self.model_flux_path = None
        expect = path + '_spectrum.txt'
        if "model_flux_path" in kwargs:
            self.model_flux_path = kwargs["model_flux_path"]
        elif path is not "" and os.path.isfile(expect):
            self.model_flux_path = expect
        else:
            raise ValueError("model_flux_path not given, and no file found at", expect)

        self.model_wavelength_path = None
        expect = path + '_wavelengths.txt'
        if "model_wavelength_path" in kwargs:
            self.model_wavelength_path = kwargs["model_wavelength_path"]
        elif path != "" and os.path.isfile(expect):
            self.model_wavelength_path = expect
        else:
            raise ValueError("model_wavelength_path not given, and no file found at", expect)

        self.observed_flux_path = None
        if "observed_flux_path" in kwargs:
            self.observed_flux_path = kwargs["observed_flux_path"]
        elif path != "" and os.path.isfile(path + '_obs_flux.txt'):
            self.observed_flux_path = path + '_obs_flux.txt'

        self.observed_wavelength_path = None
        if "observed_wavelength_path" in kwargs:
            self.observed_wavelength_path = kwargs["observed_wavelength_path"]
        elif path != "" and os.path.isfile(path + '_obs_wave.txt'):
            self.observed_wavelength_path = path + '_obs_wave.txt'

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

        if self.observed_flux_path is not None:
            obs_flux = np.loadtxt(self.observed_flux_path) * units.microjansky
        else:
            obs_flux = []

        if self.observed_wavelength_path is not None:
            obs_wave = np.loadtxt(self.observed_wavelength_path)
        else:
            obs_wave = []

        if self.observed_flux_err_path is not None:
            obs_flux_err = np.loadtxt(self.observed_flux_err_path)
        else:
            obs_flux_err = []

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

    def alexa_plot(self):
        flux = self.model_table["flux_nu"]
        wave = self.model_table["wavelength"]

        obs_flux = self.obs_table["flux_nu"]
        obs_wave = self.obs_table["wavelength"]

        plt.figure(figsize=[12, 8])
        plt.plot(wave, flux, color='black', alpha=0.7, zorder=2, label='Model')
        plt.plot(obs_wave, obs_flux, color='red', alpha=1, zorder=1, label='Observed')
        # plt.xlim(2e3 * units.Angstrom, 15e3 * units.Angstrom)
        plt.ylim(flux.min(), flux.max())

        plt.legend(loc='best', fontsize=12)
        plt.ylabel(r'F$_{\nu}$ [$\mu$Jy]', fontsize=15)
        plt.xlabel(r'Observed Wavelength [$\AA$]', fontsize=15)

        plt.show()

        plt.figure(figsize=[12, 8])
        plt.plot(wave, flux, color='black', alpha=0.7, zorder=2, label='Model')
        plt.plot(obs_wave, obs_flux, color='#39BFD0', alpha=0.6, zorder=1, label='Observed')
        # plt.xlim(2e3 * units.Angstrom, 15e3 * units.Angstrom)

        plt.legend(loc='best', fontsize=12)
        plt.ylabel(r'F$_{\nu}$ [$ \mu$Jy]', fontsize=15)
        plt.xlabel(r'Observed Wavelength [$\AA$]', fontsize=15)
        plt.ylim(obs_flux.min(), obs_flux.max() / 26)

        plt.show()


