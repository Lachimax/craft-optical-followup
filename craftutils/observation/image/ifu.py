import numpy as np

from .__init__ import ImagingImage


class IFUImage(ImagingImage):
    axis_spec = 0
    def white_light_image(
            self,
            ext: int = 1,
            output_path: str = None
    ):
        data = self.collapse_spectral(ext=ext)
        header = self.header_2d(ext=ext)
        img = ImagingImage()

    def collapse_spectral(
            self,
            ext: int = 1,
    ):
        self.load_data()
        return np.nansum(self.data[ext], axis=self.axis_spec)

    def header_2d(
            self,
            ext: int = 1
    ):
        header = self.headers[ext].copy()
        for key in ("CTYPE3", "CUNIT3", "CRVAL3"):
            if key in header:
                header.pop(key)
        return header


class MUSEImage(IFUImage):
    pass
    # def white_light_image(
    #         self,
    #         ext: 1,
    #         output_path: str = None
    # ):
    #     super().white_light_image()