import astropy.io.fits as fits
import astropy.table as table
import astropy.units as units
import astropy.constants as constants

from .sed import SEDModel


class CIGALEModel(SEDModel):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.hdu_list: fits.HDUList = None

    def load_data(self):
        self.hdu_list = fits.open(self.path)
        self.data_table = table.QTable(self.hdu_list[1].data)
        hdr = self.hdu_list[1].header.copy()

        for col_name in filter(lambda k: k.startswith("TTYPE"), hdr):
            col_n = int(col_name[5:])
            col_unit = units.Unit(hdr[f"TUNIT{col_n}"])
            self.data_table[col_name] *= col_unit

        self.sanitise_columns()
        self.frequency_from_wavelength()
        self.luminosity_per_frequency()
        self.flux_per_wavelength()

    @classmethod
    def columns(cls):
        return {
            "flux_nu": "Fnu",
            "luminosity_lambda": "L_lambda_total"
        }
