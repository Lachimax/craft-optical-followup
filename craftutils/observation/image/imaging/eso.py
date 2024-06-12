import astropy.units as units

import craftutils.utils as u

from .__init__ import CoaddedImage, ImagingImage
from ..eso import ESOImage


class ESOImagingImage(ImagingImage, ESOImage):

    def extract_airmass(self):
        key = self.header_keys()["airmass"]
        self.airmass = self.extract_header_item(key)
        u.debug_print(1, f"{self.name}.airmass", self.airmass)
        if self.airmass is None:
            airmass_start = self.extract_header_item("ESO TEL AIRM START")
            airmass_end = self.extract_header_item("ESO TEL AIRM END")
            self.airmass = (airmass_start + airmass_end) / 2
        u.debug_print(1, f"{self.name}.airmass", self.airmass)
        return self.airmass

    @classmethod
    def count_exposures(cls, image_paths: list):
        # Counts only chip 1 images
        n = 0
        for path in image_paths:
            image = cls(path=path)
            chip = image.extract_chip_number()
            if chip == 1:
                n += 1
        return n

    @classmethod
    def header_keys(cls) -> dict:
        header_keys = super().header_keys()
        header_keys.update(ESOImage.header_keys())
        header_keys.update({
            "noise_read": "HIERARCH ESO DET OUT1 RON",
            "filter": "HIERARCH ESO INS FILT1 NAME",
            "gain": "HIERARCH ESO DET OUT1 GAIN",
            "program_id": "HIERARCH ESO OBS PROG ID",
        })
        return header_keys


class HAWKIImage(ESOImagingImage):
    instrument_name = "vlt-hawki"

    @classmethod
    def header_keys(cls) -> dict:
        header_keys = super().header_keys()
        header_keys.update(ESOImage.header_keys())
        header_keys.update({
            "gain": "GAIN",
            "noise_read": "READNOIS"
        })
        return header_keys

    def extract_chip_number(self, ext: int = 0):
        return int(self.extract_header_item("HIERARCH ESO DET CHIP NO", ext=ext))


class HAWKICoaddedImage(CoaddedImage):
    num_chips = 4
    instrument_name = "vlt-hawki"

    def extract_chip_number(self, ext: int = 0):
        return 0

    def extract_exposure_time(self):
        return 1 * units.s

    def extract_filter(self):
        key = self.header_keys()["filter"]
        self.filter_name = self.extract_header_item(key)
        if self.filter_name is None:
            key = super().header_keys()["filter"]
            self.filter_name = self.extract_header_item(key)
        if self.filter_name is not None:
            self.filter_short = self.filter_name[0]

        self._filter_from_name()

        return self.filter_name

    def zeropoint(
            self,
            **kwargs
    ):
        print(self.filter_name)

        self.set_header_items(
            {
                "EXPTIME": 1 * units.s,
                "INTTIME": self.extract_header_item("TEXPTIME") * units.s,
            }
        )
        # self.load_data(force=True)

        # self.add_zeropoint(
        #     catalogue="calib_pipeline",
        #     zeropoint=self.extract_header_item("PHOTZP") * units.mag + self.filter.vega_magnitude_offset(),
        #     zeropoint_err=self.extract_header_item("PHOTZPER"),
        #     extinction=0.0 * units.mag,
        #     extinction_err=0.0 * units.mag,
        #     airmass=0.0,
        #     airmass_err=0.0
        # )

        zp = super().zeropoint(
            **kwargs
        )

        return zp

        # self.select_zeropoint(True)
        # return self.zeropoint_best

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "gain": "GAIN",
            "filter": "HIERARCH ESO INS FILT1 NAME"
        })
        return header_keys


class FORS2Image(ESOImagingImage):
    instrument_name = "vlt-fors2"
    num_chips = 2

    def __init__(self, path: str, frame_type: str = None, **kwargs):
        super().__init__(path=path, frame_type=frame_type, instrument_name=self.instrument_name)
        self.other_chip = None
        self.chip_number = None

    def extract_chip_number(self, ext: int = 0):
        chip_string = self.extract_header_item(key='HIERARCH ESO DET CHIP1 ID', ext=ext)
        chip = 0
        if chip_string == 'CCID20-14-5-3':
            chip = 1
        elif chip_string == 'CCID20-14-5-6':
            chip = 2
        self.chip_number = chip
        return chip

    def _output_dict(self):
        outputs = super()._output_dict()
        if self.other_chip is not None:
            other_chip = self.other_chip.output_file
        else:
            other_chip = None
        outputs.update({
            "other_chip": other_chip,
        })
        return outputs

    def load_output_file(self):
        outputs = super().load_output_file()
        if outputs is not None:
            if "other_chip" in outputs:
                self.other_chip = outputs["other_chip"]
        return outputs

    @classmethod
    def header_keys(cls) -> dict:
        header_keys = super().header_keys()
        header_keys.update(ESOImage.header_keys())
        header_keys.update({
            "noise_read": "HIERARCH ESO DET OUT1 RON",
            "gain": "HIERARCH ESO DET OUT1 GAIN",
            "program_id": "HIERARCH ESO OBS PROG ID",
            "filter": "HIERARCH ESO INS FILT1 NAME"
        })
        return header_keys


class FORS2CoaddedImage(CoaddedImage):
    instrument_name = "vlt-fors2"

    def __init__(
            self,
            path: str,
            frame_type: str = None,
            **kwargs
    ):
        super().__init__(
            path=path,
            frame_type=frame_type,
            instrument_name=self.instrument_name,
        )

    def zeropoint(
            self,
            **kwargs
    ):

        skip_zp = False
        zp = {}
        if self.filter.calib_retrievable() and "instrument_archive" not in self.zeropoints:
            zp = self.calibration_from_qc1()
            skip_retrievable = True
            if "skip_retrievable" in kwargs:
                skip_retrievable = kwargs.pop("skip_retrievable")
            if skip_retrievable:
                skip_zp = True
        elif "instrument_archive" in self.zeropoints:
            zp = self.zeropoints["instrument_archive"]
        if not skip_zp:
            zp = super().zeropoint(
                **kwargs
            )
        return zp

    def calibration_from_qc1(self):
        """
        Use the FORS2 QC1 archive to retrieve calibration parameters.
        :return:
        """
        self.extract_filter()
        u.debug_print(
            1, f"FORS2CoaddedImage.calibration_from_qc1(): {self}.instrument ==", self.instrument,
            self.instrument_name)
        fil = self.instrument.filters[self.filter_name]
        fil.retrieve_calibration_table()
        if fil.calibration_table is not None:
            self.extract_date_obs()
            row = fil.get_nearest_calib_row(mjd=self.mjd_obs)

            if self.epoch is not None and self.epoch.airmass_err is not None:
                airmass_err = self.epoch.airmass_err[self.filter_name]
            else:
                airmass_err = 0.0

            zp = self.add_zeropoint(
                zeropoint=row["zeropoint"],
                zeropoint_err=row["zeropoint_err"],
                airmass=self.extract_airmass(),
                airmass_err=airmass_err,
                extinction=row["extinction"],
                extinction_err=row["extinction_err"],
                mjd_measured=row["mjd_obs"],
                delta_t=row["mjd_obs"] - self.mjd_obs,
                n_matches=None,
                catalogue="instrument_archive"
            )

            self.extinction_atmospheric = row["extinction"]
            self.extinction_atmospheric_err = row["extinction_err"]

            self.add_log(
                action=f"Retrieved calibration values from ESO QC1 archive.",
                method=self.calibration_from_qc1,
            )
            self.update_output_file()

            return zp
        else:
            return None
