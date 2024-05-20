import craftutils.observation.image as image
from .epoch import SurveyImagingEpoch


class PanSTARRS1ImagingEpoch(SurveyImagingEpoch):
    instrument_name = "panstarrs1"
    catalogue = "panstarrs1"
    coadded_class = image.PanSTARRS1Cutout
    preferred_zeropoint = "panstarrs1"

    # TODO: Automatic cutout download; don't worry for now.
    def proc_download(self, output_dir: str, **kwargs):
        """
        Automatically download PanSTARRS1 cutout.
        :param output_dir:
        :param kwargs:
        :return:
        """
        pass
