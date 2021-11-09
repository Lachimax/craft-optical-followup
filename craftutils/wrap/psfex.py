import os
from typing import Union

import numpy as np
import psfex as pex

from astropy.modeling import models, fitting, Fittable2DModel, Parameter

from scipy.ndimage import shift

import craftutils.fits_files as ff
from craftutils.utils import system_command



class PSFExModel(Fittable2DModel):
    """
    Warning: does not work
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
    system_command(command="psfex", arguments=[catalog], o=output_name, **kwargs)
    os.chdir(old_dir)
    psfex_path = os.path.join(output_dir, output_name)
    return psfex_path


def load_psfex(model: Union[str], x, y):
    """
    Since PSFEx generates a model that is dependent on image position, this is used to collapse that into a useable kernel
    for convolution and insertion purposes. See https://psfex.readthedocs.io/en/latest/Appendices.html
    :param model:
    :param x:
    :param y:
    :return:
    """
    model, path = ff.path_or_hdu(model)

    header = model[1].header

    a = model[1].data[0][0]

    x = (x - header['POLZERO1']) / header['POLSCAL1']
    y = (y - header['POLZERO2']) / header['POLSCAL2']

    if len(a) == 3:
        psf = a[0] + a[1] * x + a[2] * y

    elif len(a) == 6:
        psf = a[0] + a[1] * x + a[2] * x ** 2 + a[3] * y + a[4] * y ** 2 + a[5] * x * y

    elif len(a) == 10:
        psf = a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3 + a[4] * y + a[5] * x * y + a[6] * x ** 2 * y + \
              a[7] * y ** 2 + a[8] * x * y ** 2 + a[9] * y ** 3

    else:
        raise ValueError("I haven't accounted for polynomials of order > 3. My bad.")

    if path:
        model.close()

    return psf
