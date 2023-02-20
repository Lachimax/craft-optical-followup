import astropy.io.fits as fits
import astropy.table as table
import astropy.units as units
import astropy.constants as constants
import astropy.cosmology as cosmology

from .sed import SEDModel


class CIGALEModel(SEDModel):
    default_cosmology = cosmology.WMAP7

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.hdu_list: fits.HDUList = None

    def load_data(self):
        self.hdu_list = fits.open(self.path)
        self.model_table = table.QTable(self.hdu_list[1].data)
        hdr = self.hdu_list[1].header.copy()

        for key in filter(lambda k: k.startswith("TTYPE"), hdr):
            col_name = hdr[key]
            col_n = int(key[5:])
            col_unit = units.Unit(hdr[f"TUNIT{col_n}"])
            self.model_table[col_name] *= col_unit

        self.prep_data()

    @classmethod
    def columns(cls):
        return {
            "flux_nu": "Fnu",
            "luminosity_lambda": "L_lambda_total"
        }
