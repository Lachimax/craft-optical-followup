import os

import craftutils.fits_files as ff


def psfex(catalog: str, output_name: str = None, output_dir: str = None):
    old_dir = os.getcwd()
    if output_dir is None:
        output_dir = os.getcwd()
    else:
        os.chdir(output_dir)
    if output_name is None:
        cat_name = os.path.split(catalog)[-1]
        output_name = cat_name.replace(".fits", ".psf")
    sys_str = f"psfex {catalog} -o {output_name}"
    print(f"\n{sys_str}\n")
    os.system(sys_str)
    print(f"\n{sys_str}\n")
    os.chdir(old_dir)
    psfex_path = os.path.join(output_dir, output_name)
    return psfex_path


def load_psfex(model: str, x, y):
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
