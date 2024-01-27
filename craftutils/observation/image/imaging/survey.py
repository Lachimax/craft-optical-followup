import astropy.units as units

from .coadded import CoaddedImage
from ..image import gain_unit, noise_read_unit


class SurveyCutout(CoaddedImage):
    def do_subtract_background(self):
        return False


class WISECutout(SurveyCutout):
    instrument_name = "wise"

    def extract_filter(self):
        key = self.header_keys()["filter"]
        band_n = self.extract_header_item(key)
        self.filter_name = f"W{band_n}"
        if self.filter_name is not None:
            self.filter_short = self.filter_name
        self._filter_from_name()
        return self.filter_name

    def extract_gain(self):
        """
        I couldn't find any effective gain in FITS headers or in WISE documentation, so we'll assume this.

        :return:
        """
        self.gain = 1. * gain_unit
        return self.gain

    def extract_exposure_time(self):
        """
        I couldn't find any effective exposure time in FITS headers or in WISE documentation, so we'll assume this.

        :return:
        """
        self.exposure_time = 1 * units.second
        return self.exposure_time

    def zeropoint(
            self,
            **kwargs
    ):
        self.zeropoint_best = self.add_zeropoint(
            catalogue="calib_pipeline",
            zeropoint=self.extract_header_item("MAGZP"),
            zeropoint_err=self.extract_header_item("MAGZPUNC"),
            extinction=0.0 * units.mag,
            extinction_err=0.0 * units.mag,
            airmass=0.0,
            airmass_err=0.0
        )
        return self.zeropoint_best

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "filter": "BAND"
        })
        return header_keys


class DESCutout(SurveyCutout):
    instrument_name = "decam"

    def zeropoint(
            self,
            **kwargs
    ):
        self.add_zeropoint(
            catalogue="calib_pipeline",
            zeropoint=self.extract_header_item("MAGZERO"),  # - 2.5 * np.log10(exptime)) * units.mag,
            zeropoint_err=0.0 * units.mag,
            extinction=0.0 * units.mag,
            extinction_err=0.0 * units.mag,
            airmass=0.0,
            airmass_err=0.0
        )
        zp = super().zeropoint(
            **kwargs
        )

        return zp

    def extract_unit(self, astropy: bool = False):
        unit = "ct / s"
        if astropy:
            unit = units.ct / units.s
        return unit

    def extract_exposure_time(self):
        self.exposure_time = 1. * units.second
        return self.exposure_time

    def extract_noise_read(self):
        self.noise_read = 0. * noise_read_unit
        return self.noise_read

    def extract_integration_time(self):
        return self.extract_header_item("EXPTIME") * units.second

    def extract_filter(self):
        key = self.header_keys()["filter"]
        fil_string = self.extract_header_item(key)
        self.filter_name = fil_string[:fil_string.find(" ")]
        self.filter_short = self.filter_name

        self._filter_from_name()

        return self.filter_name

    def extract_ncombine(self):
        return 1


class PanSTARRS1Cutout(SurveyCutout):
    instrument_name = "panstarrs1"

    def __init__(self, path: str, **kwargs):
        super().__init__(path=path)
        # self.instrument_name = "panstarrs1"
        self.extract_filter()
        self.exposure_time = None
        self.extract_exposure_time()

    def mask_nearby(self):
        return True

    def detection_threshold(self):
        return 10.

    def extract_filter(self):
        key = self.header_keys()["filter"]
        fil_string = self.extract_header_item(key)
        self.filter_name = fil_string[:fil_string.find(".")]
        self.filter_short = self.filter_name

        self._filter_from_name()

        return self.filter_name

    def extract_integration_time(self):
        return self.extract_exposure_time()

    def zeropoint(
            self,
            **kwargs
    ):
        """
        According to the reference below, the PS1 cutouts are scaled to zeropoint 25.
        https://outerspace.stsci.edu/display/PANSTARRS/PS1+Stack+images
        :return:
        """

        self.load_headers()

        self.add_zeropoint(
            catalogue="calib_pipeline",
            zeropoint=self.extract_header_item("FPA.ZP"),
            zeropoint_err=0.0 * units.mag,
            extinction=0.0 * units.mag,
            extinction_err=0.0 * units.mag,
            airmass=0.0,
            airmass_err=0.0
        )

        zp = super().zeropoint(
            **kwargs
        )
        return zp

    # self.select_zeropoint(True)

    # I only wrote this function below because I couldn't find the EXPTIME key in the PS1 cutouts. It is, however, there.
    # def extract_exposure_time(self):
    #     # self.load_headers()
    #     # exp_time_keys = filter(lambda k: k.startswith("EXP_"), self.headers[0])
    #     # exp_time = 0.
    #     # exp_times = []
    #     # for key in exp_time_keys:
    #     #     exp_time += self.headers[0][key]
    #     # #    exp_times.append(self.headers[0][key])
    #     #
    #     # self.exposure_time = exp_time * units.second  # np.mean(exp_times)
    #     self.exposure_time = 1.0 * units.second
    #     return self.exposure_time

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "noise_read": "HIERARCH CELL.READNOISE",
            "filter": "HIERARCH FPA.FILTERID",
            "gain": "HIERARCH CELL.GAIN",
            "ncombine": "NINPUTS"
        })
        return header_keys
