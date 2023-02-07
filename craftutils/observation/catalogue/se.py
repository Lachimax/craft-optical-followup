from typing import Union

import numpy as np

import astropy.table as table
import astropy.units as units

from craftutils.observation.catalogue import Catalogue
import craftutils.observation.image as image
import craftutils.params as p
import craftutils.utils as u


class SECatalogue(Catalogue):
    """
    Catalogue subclass for handling Source Extractor output.
    """
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.image: Union[image.ImagingImage, str] = None
        if "image" in kwargs:
            img = kwargs["image"]
            if isinstance(img, image.ImagingImage):
                self.image = img
            elif isinstance(img, str):
                self.image = image.from_path(path=img)
        self.se_path: str = None
        if "se_path" in kwargs:
            self.set_se_path(kwargs["se_path"])
        self.se_cat: table.QTable = None
        if "se_cat" in kwargs:
            self.se_cat = table.QTable(kwargs["se_cat"])
        self.cat_type: str = None

    @classmethod
    def _do_not_include_in_output(cls):
        do_not_include = super()._do_not_include_in_output()
        do_not_include += ["se_cat"]
        return do_not_include

    def _load_source_cat_sextractor(self, path: str, wcs_ext: int = 0):
        self.image.load_wcs()
        print("Loading source catalogue from", path)
        source_cat = table.QTable.read(path, format="ascii.sextractor")
        if "SPREAD_MODEL" in source_cat.colnames:
            source_cat = u.classify_spread_model(source_cat)
        source_cat["RA"], source_cat["DEC"] = self.image.wcs[wcs_ext].all_pix2world(
            source_cat["X_IMAGE"],
            source_cat["Y_IMAGE"],
            1
        ) * units.deg
        self.image.extract_astrometry_err()
        if self.image.ra_err is not None:
            source_cat["RA_ERR"] = np.sqrt(
                source_cat["ERRX2_WORLD"].to(units.arcsec ** 2) + self.image.ra_err ** 2)
        else:
            source_cat["RA_ERR"] = np.sqrt(
                source_cat["ERRX2_WORLD"].to(units.arcsec ** 2))
        if self.image.dec_err is not None:
            source_cat["DEC_ERR"] = np.sqrt(
                source_cat["ERRY2_WORLD"].to(units.arcsec ** 2) + self.image.dec_err ** 2)
        else:
            source_cat["DEC_ERR"] = np.sqrt(
                source_cat["ERRY2_WORLD"].to(units.arcsec ** 2))

        return source_cat

    def load_se_table(self, force: bool = False, load_as_main: bool = True):
        if self.se_path is not None:
            if force:
                self.se_cat = None
            if self.se_cat is None:
                self.se_cat = self._load_source_cat_sextractor(path=self.se_path)
            if load_as_main:
                self.table = self.se_cat
                self.cat_type = "source-extractor"
        else:
            print("source_cat could not be loaded from SE file because source_cat_sextractor_path has not been set.")

    def set_se_path(self, path: str, load: bool = True):
        self.se_path = p.check_abs_path(path)
        if load:
            self.load_se_table(force=True)
        return self.se_path

    def aperture_areas(self, pixel_scale: units.Equivalency):
        self.table["A_IMAGE"] = self.table["A_WORLD"].to(units.pix, pixel_scale)
        self.table["B_IMAGE"] = self.table["A_WORLD"].to(units.pix, pixel_scale)
        self.table["KRON_AREA_IMAGE"] = self.table["A_IMAGE"] * self.table["B_IMAGE"] * np.pi
        self.update_output_file()

    def 



