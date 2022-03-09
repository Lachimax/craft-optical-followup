import os

import numpy as np

try:
    import psfex as pex
except ImportError:
    print("psfex not installed; some functionality will not be available.")

from astropy.modeling import Fittable2DModel, Parameter

from scipy.ndimage import shift

from craftutils.utils import system_command


class PSFExModel(Fittable2DModel):
    """
    Warning: does not work. I tried!
    """
    n_inputs = 2
    n_outputs = 1

    flux = Parameter()
    x_0 = Parameter()
    y_0 = Parameter()

    def __init__(self, psfex_file: str, data_shape: tuple, flux: float, x_0: float, y_0: float):
        self.psfex_file = psfex_file
        self.psfex_model = pex.PSFEx(psfex_file)
        self.data_shape = data_shape

        super().__init__(flux, x_0, y_0)

    def evaluate(self, x, y, flux, x_0, y_0):
        mock = np.zeros(shape=self.data_shape)
        model_img = self.psfex_model.get_rec(y_0, x_0)
        y_cen, x_cen = self.psfex_model.get_center(y_0, x_0)
        model_img /= np.sum(model_img)
        model_img *= flux
        mock[0:model_img.shape[0], 0:model_img.shape[1]] += model_img
        mock = shift(mock, (y_0 - y_cen, x_0 - x_cen))

        return mock[int(x), int(y)]


def psfex(catalog: str, output_name: str = None, output_dir: str = None, **kwargs):
    old_dir = os.getcwd()
    if output_dir is None:
        output_dir = os.getcwd()
    else:
        os.chdir(output_dir)
    if output_name is None:
        cat_name = os.path.split(catalog)[-1]
        output_name = cat_name.replace(".fits", ".psf")
    system_command(command="psfex", arguments=[catalog], **kwargs)
    os.chdir(old_dir)
    psfex_path = os.path.join(output_dir, output_name)
    return psfex_path


def load_psfex(model_path: str, x: float, y: float):
    """
    Since PSFEx generates a model that is dependent on image position, this is used to collapse that into a useable kernel
    for convolution and insertion purposes. See https://psfex.readthedocs.io/en/latest/Appendices.html
    :param model_path:
    :param x:
    :param y:
    :return:
    """

    psfex_model = pex.PSFEx(model_path)
    psf = psfex_model.get_rec(y, x)
    centre_psf_x, centre_psf_y = psfex_model.get_center(y, x)
    centre_x, centre_y = psf.shape[1] / 2, psf.shape[0] / 2
    psf = shift(psf, (centre_x - centre_psf_y, centre_y - centre_psf_x))

    return psf
