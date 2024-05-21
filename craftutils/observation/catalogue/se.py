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
    ra_key = "RA"
    dec_key = "DEC"

    def __init__(
            self,
            path: str = None,
            se_path: str = None,
            **kwargs
    ):
        self.path = None
        self.se_path: str = None
        self.image: Union[image.ImagingImage, str] = None
        if "image" in kwargs:
            img = kwargs["image"]
            if isinstance(img, image.ImagingImage):
                self.image = img
            elif isinstance(img, str):
                self.image = image.from_path(path=img)
        if path is None and se_path is None:
            raise ValueError("Either path or se_path must be given to initialise an SECatalogue")
        elif path is None:
            self.set_se_path(se_path)
            self.write()
            path = se_path.replace(".cat", ".ecsv")
        super().__init__(path=path, **kwargs)
        self.se_cat: table.QTable = None
        if "se_cat" in kwargs:
            self.se_cat = table.QTable(kwargs["se_cat"])

    @classmethod
    def _do_not_include_in_output(cls):
        do_not_include = super()._do_not_include_in_output()
        do_not_include += ["se_cat", "image"]
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
        else:
            print("source_cat could not be loaded from SE file because source_cat_sextractor_path has not been set.")

    def set_se_path(self, path: str, load: bool = True):
        self.se_path = p.check_abs_path(path)
        self.path = self.se_path.replace(".cat", ".ecsv")
        if load:
            self.load_se_table(force=True)
        return self.se_path

    def aperture_areas(self, pixel_scale: units.Equivalency):
        self.table["A_IMAGE"] = self.table["A_WORLD"].to(units.pix, pixel_scale)
        self.table["B_IMAGE"] = self.table["A_WORLD"].to(units.pix, pixel_scale)
        self.table["KRON_AREA_IMAGE"] = self.table["A_IMAGE"] * self.table["B_IMAGE"] * np.pi
        self.update_output_file()

    def object_axes(self, pixel_scale: units.Equivalency):
        self.table["A_IMAGE"] = self.table["A_WORLD"].to(units.pix, pixel_scale)
        self.table["B_IMAGE"] = self.table["B_WORLD"].to(units.pix, pixel_scale)
        self.update_output_file()

    def add_partial_column(
            self,
            column_name: str,
            subset_table: table.Table,
    ):
        """
        Takes a column from a table that is a subset of this catalogue's table, and adds the column values to the
        appropriate entries using the `NUMBER` column. Rows not in the subset table will be assigned `-99.` in that
        column.
        Trust me, it comes in handy.
        Assumes that NO entries have been removed from or added to the main source_cat since being produced by Source Extractor,
        as it requires that the relationship source_cat["NUMBER"] = i - 1 holds true.
        (The commented line is for use when that assumption is no longer valid, but is slower).

        :param column_name: Name of column to send.
        :param subset_table:
        :return:
        """
        if column_name not in self.table.colnames:
            self.table.add_column(-99 * subset_table[column_name].unit, name=column_name)
        self.table.sort("NUMBER")
        for star in subset_table:
            index = star["NUMBER"]
            # i = self.find_object_index(index, dual=False)
            self.table[index - 1][column_name] = star[column_name]

    def calibrate_magnitudes(
            self,
            zeropoint_dict: dict,
            mag_name: str,
            force: bool = True
    ):
        zeropoint_name = zeropoint_dict["catalogue"]

        if force or f"MAG_AUTO_{mag_name}" not in self.table.colnames:
            mags = self.image.magnitude(
                flux=self.table["FLUX_AUTO"],
                flux_err=self.table["FLUXERR_AUTO"],
                cat_name=zeropoint_name
            )
            self.table[mag_name] = zeropoint_dict["zeropoint"]
            self.table[f"ZPERR_{zeropoint_name}"] = zeropoint_dict["zeropoint_err"]
            self.table[f"AIRMASS_{zeropoint_name}"] = zeropoint_dict["airmass"]
            self.table[f"AIRMASSERR_{zeropoint_name}"] = zeropoint_dict["airmass_err"]
            self.table["EXT_ATM"] = zeropoint_dict["extinction"]
            self.table["EXT_ATMERR"] = zeropoint_dict["extinction_err"]
            self.table[f"{mag_name}_ATM_CORR"] = zeropoint_dict["zeropoint_img"]
            self.table[f"{mag_name}_ATM_CORRERR"] = zeropoint_dict["zeropoint_img_err"]

            self.table[f"MAG_AUTO_{mag_name}"] = mags[0]
            self.table[f"MAGERR_AUTO_{mag_name}"] = mags[1]
            self.table[f"MAG_AUTO_{mag_name}_no_ext"] = mags[2]
            self.table[f"MAGERR_AUTO_{mag_name}_no_ext"] = mags[3]

            if "FLUX_PSF" in self.table.colnames:
                mags = self.image.magnitude(
                    flux=self.table["FLUX_PSF"],
                    flux_err=self.table["FLUXERR_PSF"],
                    cat_name=zeropoint_name
                )

                self.table[f"MAG_PSF_{mag_name}"] = mags[0]
                self.table[f"MAGERR_PSF_{mag_name}"] = mags[1]
                self.table[f"MAG_PSF_{mag_name}_no_ext"] = mags[2]
                self.table[f"MAGERR_PSF_{mag_name}_no_ext"] = mags[3]

            self.update_output_file()

        else:
            print(f"Magnitudes already calibrated for {zeropoint_name}")

    def _output_dict(self):
        outputs = super()._output_dict()
        if self.image is not None:
            outputs["image_path"] = self.image.path
        else:
            outputs["image_path"] = None
        return outputs
