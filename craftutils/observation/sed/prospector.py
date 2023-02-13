import astropy.table as table

from .sed import *


class GordonProspectorModel(SEDModel):

    def load_data(self):
        flux = np.loadtxt(self.path + '_spectrum.txt') * units.microjansky
        wave = np.loadtxt(self.path + '_wavelengths.txt') * units.Angstrom
        self.table = table.QTable(
            {
                "flux_nu": flux,
                "wavelength": wave
            }
        )
        self.frequency_from_wavelength()
