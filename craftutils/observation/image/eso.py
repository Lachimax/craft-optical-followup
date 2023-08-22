import craftutils.utils as u

from ..image import Image

# __all__ = []

# @u.export
class ESOImage(Image):
    """
    Generic parent class for ESO images, both spectra and imaging
    """

    def extract_frame_type(self):
        obj = self.extract_object()
        category = self.extract_header_item("ESO DPR CATG")
        if category is None:
            category = self.extract_header_item("ESO PRO CATG")
        if obj == "BIAS":
            self.frame_type = "bias"
        elif "FLAT" in obj:
            self.frame_type = "flat"
        elif obj == "STD":
            self.frame_type = "standard"
        elif obj == "DARK":
            self.frame_type = "dark"
        elif category == "SCIENCE":
            self.frame_type = "science"
        elif category == "SCIENCE_REDUCED_IMG":
            self.frame_type = "science_reduced"
        u.debug_print(2, f"ESOImagingImage.extract_frame_type(): {obj=}, {category=}, {self.frame_type=}")
        return self.frame_type

    @classmethod
    def header_keys(cls):
        header_keys = super().header_keys()
        header_keys.update({
            "mode": "HIERARCH ESO INS MODE",
        })
        return header_keys
