import shutil
import os
from typing import Tuple, Union, List

import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
import astropy.units as units
import astropy.table as table
from astropy.visualization import LogStretch, ImageNormalize, SqrtStretch

import craftutils.utils as u


def galfit(config: str, output_dir: str):
    cwd = os.getcwd()
    os.chdir(output_dir)
    u.system_command_verbose(f"galfit {config}")
    os.chdir(cwd)

def feedme_sky_model(
        background_center: float = 1.3920 * units.adu,
        fit_background_center: bool = True,
        gradient_x: float = 0.0 * units.adu / units.pix,
        fit_gradient_x: bool = True,
        gradient_y: float = 0.0 * units.adu / units.pix,
        fit_gradient_y: bool = True,
        skip_in_output: bool = False,
        **kwargs
):
    """
    Generate input file lines in the GALFIT feedme format, for sky model.
    Parameters provided in valid astropy units will be converted to the units used by GALFIT.
    :param background_center: Background level at centre of fitting region.
        Corresponds to item 1) in the GALFIT feedme definition.
    :param fit_background_center: If True, GALFIT will fit for background_center; if False, it will remain fixed.
    :param gradient_x: dsky/dx in ADU / pixel
        Corresponds to item 2) in the GALFIT feedme definition.
    :param fit_gradient_x: If True, GALFIT will fit for gradient_x; if False, it will remain fixed.
    :param gradient_y: dsky/dy in ADU / pixel
        Corresponds to item 3) in the GALFIT feedme definition.
    :param fit_gradient_y: If True, GALFIT will fit for gradient_y; if False, it will remain fixed.
    :param skip_in_output: Skip in output image block?
        Corresponds to item Z) in the GALFIT feedme definition.
    :param pix_scale: pixel scale for conversion from angular sizes (eg arcsec) to pixels.
    :return: List of strings corresponding to lines in a .feedme input file, suitable for use with open().writelines().
    """

    return feedme_model(
        "sky",
        skip_in_output,
        None,
        (False, False),
        (u.dequantify(background_center), fit_background_center),
        (u.dequantify(gradient_x), fit_gradient_x),
        (u.dequantify(gradient_y), fit_gradient_y)
    )


def feedme_sersic_model(
        x: units.Quantity,
        y: units.Quantity,
        int_mag: units.Quantity,
        fit_x: bool = True,
        fit_y: bool = True,
        fit_int_mag: bool = True,
        r_e: units.Quantity = 3. * units.pix,
        fit_r_e: bool = True,
        n: float = 1.,
        fit_n: bool = True,
        axis_ratio: float = 0.5,
        fit_axis_ratio: bool = True,
        position_angle: units.Quantity = 0. * units.deg,
        fit_position_angle: bool = True,
        skip_in_output: bool = 0,
        rotation_params: dict = {},
        pix_scale: Tuple = (),
        **kwargs
):
    """
    Generate input file lines in the GALFIT feedme format, for sersic model.
    Parameters provided in valid astropy units will be converted to the units used by GALFIT.
    :param x: x-position of model centre, in pixels.
        Corresponds to first component of item 1) in the GALFIT feedme definition.
    :param y: y-position of model centre, in pixels.
        Corresponds to second component of item 1) in the GALFIT feedme definition.
    :param int_mag: integrated magnitude of model.
        Corresponds to item 3) in the GALFIT feedme definition.
    :param fit_x: If True, GALFIT will fit for x; if False, it will remain fixed.
    :param fit_y: If True, GALFIT will fit for y; if False, it will remain fixed.
    :param fit_int_mag: If True, GALFIT will fit for int_mag; if False, it will remain fixed.
    :param r_e: Effective radius of model, in pixels.
        Corresponds to item 4) in the GALFIT feedme definition.
    :param fit_r_e: If True, GALFIT will fit for r_e; if False, it will remain fixed.
    :param n: Sersic index n of model.
        Corresponds to item 5) in the GALFIT feedme definition.
    :param fit_n: If True, GALFIT will fit for n; if False, it will remain fixed.
    :param axis_ratio: axis ratio b/a of model.
        Corresponds to item 9) in the GALFIT feedme definition.
    :param fit_axis_ratio: If True, GALFIT will fit for axis_ratio; if False, it will remain fixed.
    :param position_angle: position angle (PA) of model, in degrees, with up=0 and left=90
        Corresponds to item 10) in the GALFIT feedme definition.
    :param fit_position_angle: If True, GALFIT will fit for position_angle; if False, it will remain fixed.
    :param skip_in_output: Skip in output image block?
        Corresponds to item Z) in the GALFIT feedme definition.
    :param rotation_params: Coordinate rotation parameters (eg for fitting spiral arms)
    :param pix_scale: pixel scale for conversion from angular sizes (eg arcsec) to pixels.
    :return: List of strings corresponding to lines in a .feedme input file, suitable for use with open().writelines().
    """
    print("feedme_sersic_model", pix_scale)
    lines = feedme_model(
        "sersic",
        skip_in_output,
        (u.dequantify(x, unit=units.pix, equivalencies=pix_scale), u.dequantify(y, unit=units.pix)),
        (fit_x, fit_y),
        (0., False),
        (u.dequantify(int_mag, unit=units.mag), fit_int_mag),
        (u.dequantify(r_e, unit=units.pix, equivalencies=pix_scale), fit_r_e),
        (u.dequantify(n), fit_n),
        (0., False),
        (0., False),
        (0., False),
        (u.dequantify(axis_ratio), fit_axis_ratio),
        (u.dequantify(position_angle, unit=units.deg), fit_position_angle),
    )

    if rotation_params:
        rot_lines = feedme_rot(
            pix_scale=pix_scale,
            **rotation_params
        )
        lines[-1:-1] = rot_lines

    return lines


def feedme_rot(
        rot_type: str = "log",
        r_in: units.Quantity = 0. * units.pix,
        fit_r_in: bool = True,
        r_out: units.Quantity = 5. * units.pix,
        fit_r_out: bool = True,
        theta_out: units.Quantity = 180. * units.deg,
        fit_theta_out: bool = True,
        r_ws: units.Quantity = 2. * units.pix,
        fit_r_ws: bool = True,
        theta_inc: units.Quantity = 45. * units.deg,
        fit_theta_inc: units.Quantity = True,
        theta_pa: units.Quantity = 45. * units.deg,
        fit_theta_pa: bool = True,
        pix_scale: Tuple = (),
        **kwargs
):
    """
    Generate input file lines in the GALFIT feedme format, for coordinate rotation (eg for fitting spiral arms).
    Parameters provided in valid astropy units will be converted to the units used by GALFIT.
    :param rot_type: "log" or "power", determining which rotation function to use.
        Corresponds to item R0) in the GALFIT feedme definition.
    :param r_in: Spiral inner radius, in pixels. If this goes extremely negative in fitting, fix to 0.
        Corresponds to item R1) in the GALFIT feedme definition.
    :param fit_r_in: If True, GALFIT will fit for r_in; if False, it will remain fixed.
    :param r_out: Spiral outer radius, in pixels.
        Corresponds to item R2) in the GALFIT feedme definition.
    :param fit_r_out: If True, GALFIT will fit for r_out; if False, it will remain fixed.
    :param theta_out: Cumulative rotation to outer radius, in degrees.
        Corresponds to item R3) in the GALFIT feedme definition.
    :param fit_theta_out: If True, GALFIT will fit for theta_out; if False, it will remain fixed.
    :param r_ws: winding scale radius, in pixels.
        Corresponds to item R4) in the GALFIT feedme definition.
    :param fit_r_ws: If True, GALFIT will fit for r_ws; if False, it will remain fixed.
    :param theta_inc: inclination to line-of-sight, in degrees.
        Corresponds to item R9) in the GALFIT feedme definition.
    :param fit_theta_inc: If True, GALFIT will fit for theta_inc; if False, it will remain fixed.
    :param theta_pa: Position angle of the axis about which the galaxy is rotated for inclination.
        Corresponds to item R10) in the GALFIT feedme definition.
    :param fit_theta_pa: If True, GALFIT will fit for theta_pa; if False, it will remain fixed.
    :param pix_scale: pixel scale for conversion from angular sizes (eg arcsec) to pixels.
    :return: List of strings corresponding to lines in a .feedme input file, suitable for use with open().writelines().
    """
    i = 0
    lines = [f"R{i}) {rot_type}\n"]
    i += 1
    args = (
        (u.dequantify(r_in, unit=units.pix, equivalencies=pix_scale), fit_r_in),
        (u.dequantify(r_out, unit=units.pix, equivalencies=pix_scale), fit_r_out),
        (u.dequantify(theta_out, unit=units.deg), fit_theta_out),
        (u.dequantify(r_ws, unit=units.pix, equivalencies=pix_scale), fit_r_ws),
        (0, False),
        (0, False),
        (0, False),
        (0, False),
        (u.dequantify(theta_inc, unit=units.deg), fit_theta_inc),
        (u.dequantify(theta_pa, unit=units.deg), fit_theta_pa)
    )

    lines.extend(_feedme_lines(i, "R", *args))
    return lines


def feedme_model(
        object_type: str,
        skip_in_output: bool,
        position: Tuple[float, float],
        fit_position: Union[bool, Tuple[bool, bool]],
        *args,
):
    """
    Generate the input file lines for an arbitrary model for use with GALFIT.
    :param object_type: string indicating the type of model, to be parsed by GALFIT.
    :param skip_in_output: Skip in output image block?
        Corresponds to item Z) in the GALFIT feedme definition.
    :param position: Tuple containing (x, y).
    :param fit_position: Tuple containing (fit_x, fit_y), each specifying whether to fit x and y position respectively.
    :param args: (value, fit), with value being the initial guess for a parameter and fit being a boolean indicating
        whether this parameter should be fit for (True) or held fixed (False).
        If value is provided without fit, fit defaults to True.
    :return: List of strings corresponding to lines in a .feedme input file, suitable for use with open().writelines().
    """
    i = 0
    lines = [f"{i}) {object_type}\n"]
    i += 1
    if position is not None:
        try:
            # See if we can get the position fit bools as a tuple
            fit_x, fit_y = fit_position
        except TypeError:
            # If not, assume it's a single value that both fit_x and fit_y adopt
            fit_x = fit_y = fit_position
        lines.append(f"{i}) {position[0]} {position[1]} {int(fit_x)} {int(fit_y)}\n")
        i += 1
    lines.extend(_feedme_lines(i, "", *args))

    lines.append(f"Z) {int(skip_in_output)}\n")
    return lines


def _feedme_lines(i_start: int = 0, prefix: str = "", *args):
    """
    Helper function for generating feedme parameter lines in format <prefix><i>) <initial value>    <fit value?>.
    :param i_start: number at which to begin item number.
    :param prefix: prefix character to place in front of item number, eg R is the prefix for R0).
    :param args: Either tuples with (value, fit_param) where fit_param is a bool specifying whether the parameter is to
        be fitted for in GALFIT or remain fixed.
    :return: List of strings corresponding to lines in a .feedme input file, suitable for use with open().writelines().
    """
    i = i_start
    lines = []
    for arg in args:
        try:
            # If the arg is a Tuple, the first entry is the initial guess and the second is a bool telling us whether to
            # fit for this param (True) or hold fixed (False).
            guess, fit = arg
        except TypeError:
            # Otherwise, we just have the value and we assume that we should fit.
            guess = arg
            fit = True
        lines.append(f"{prefix}{i}) {guess}     {int(fit)}\n")
        i += 1
    return lines


feedme_funcs = {
    "sersic": feedme_sersic_model
}


def galfit_feedme(
        feedme_path: str,
        input_file: str,
        output_file: str,
        zeropoint: units.Quantity,
        plate_scale: Union[Tuple[float, float], float],
        sigma_file: str = None,
        psf_file: str = None,
        psf_fine_sampling: int = None,
        mask_file: str = None,
        constraint_file: str = None,
        fitting_region_margins: Tuple[int, int, int, int] = (1, 93, 1, 93),
        convolution_size: Union[Tuple[int, int], int] = (100, 100),
        display_type: str = "regular",
        run_type: Union[str, int] = 0,
        models: List[dict] = {},
        sky_model_init: dict = None,
):
    """
    Generates and writes full GALFIT input .feedme parameter file.
    Any unset values will be left as the GALFIT defaults (see param/galfit/galfit.feedme)
    :param feedme_path: Path to write parameter file to.
    :param input_file: Path to input image FITS file.
        Corresponds to item A) in the GALFIT feedme definition.
    :param output_file: Path to write output image block FITS file.
        Corresponds to item B) in the GALFIT feedme definition.
    :param zeropoint: Path to write output image block FITS file.
        Corresponds to item J) in the GALFIT feedme definition.
    :param plate_scale: The pixel scale of the image, in arcsec / pixel. If provided as a tuple:
        (plate_scale_x, plate_scale_y)
        Corresponds to item I) in the GALFIT feedme definition.
    :param sigma_file: Path to sigma image, if used. GALFIT is quite good at generating these internally.
        Corresponds to item C) in the GALFIT feedme definition.
    :param psf_file: Path to input PSF image.
        Corresponds to item D) in the GALFIT feedme definition.
    :param psf_fine_sampling: PSF fine sampling factor relative to data; ie, if the PSF image has double the resolution,
        psf_fine_sampling is 2. GALFIT only understands integers for this value.
        Corresponds to item E) in the GALFIT feedme definition.
    :param mask_file: Path to pixel mask file, a FITS image or ASCII coord list.
        Corresponds to item F) in the GALFIT feedme definition.
    :param constraint_file: Path to file with parameter constraints (ASCII file).
        Corresponds to item G) in the GALFIT feedme definition.
    :param fitting_region_margins: Image region to fit (xmin xmax ymin ymax) in pixels.
        Corresponds to item H) in the GALFIT feedme definition.
    :param convolution_size: Size of the convolution box (x y) in pixels.
        Corresponds to item I) in the GALFIT feedme definition.
    :param display_type: regular, curses or both. I don't actually know what this does.
        Corresponds to item O) in the GALFIT feedme definition.
    :param run_type: optimize, model, imgblock or subcomps, in that order.
    :param models: List of dicts containing model initial guesses, excluding sky.
    :param sky_model_init: A model for the sky. Not required as GALFIT finds this pretty easily.
    :return: List of strings corresponding to lines in a .feedme input file, suitable for use with open().writelines().
    """

    if sigma_file is None:
        sigma_file = "none"
    if psf_file is None:
        psf_file = "none"
    if psf_fine_sampling is None:
        psf_fine_sampling = "none"
    if mask_file is None:
        mask_file = "none"
    if constraint_file is None:
        constraint_file = "none"
    if isinstance(convolution_size, int):
        convolution_size = (convolution_size, convolution_size)
    if type(plate_scale) in (int, float):
        plate_scale = (float(plate_scale), float(plate_scale))
    if isinstance(run_type, str):
        run_type = {
            "optimize": 0,
            "model": 1,
            "imgblock": 2,
            "subcomps": 3
        }[run_type]

    xmin, xmax, ymin, ymax = fitting_region_margins

    lines = [
        "\n",
        "===============================================================================\n",
        "# IMAGE and GALFIT CONTROL PARAMETERS\n",
        f"A) {input_file}\n",
        f"B) {output_file}\n",
        f"C) {sigma_file}\n",
        f"D) {psf_file}\n",
        f"E) {psf_fine_sampling}\n",
        f"F) {mask_file}\n",
        f"G) {constraint_file}\n",
        f"H) {xmin} {xmax} {ymin} {ymax}\n",
        f"I) {convolution_size[0]} {convolution_size[1]}\n",
        f"J) {u.dequantify(zeropoint, units.mag)}\n",
        f"K) {plate_scale[0]} {plate_scale[1]}\n",
        f"O) {display_type}\n",
        f"P) {run_type}\n",
        "\n",

    ]

    if sky_model_init is None:
        sky_model_init = {}
    sky_model_lines = feedme_sky_model(**sky_model_init)
    i = 1
    lines.append(f"# Object number: {i}\n")
    i += 1
    lines += sky_model_lines
    lines.append("\n")

    for model_dict in models:
        object_type = model_dict["object_type"]
        if object_type in feedme_funcs:
            func = feedme_funcs[object_type]
            # model_dict.pop("object_type")
        else:
            func = feedme_model
        model_lines = func(**model_dict)

        lines.append(f"# Object number: {i}\n")
        lines.extend(model_lines)
        lines.append("\n")
        i += 1

    lines.append("===============================================================================\n\n")

    u.rm_check(feedme_path)
    with open(feedme_path, "w") as output:
        output.writelines(lines)
    return lines


def extract_fit_params(header: fits.Header):
    """Extract the fitted parameters for all components from a GALFIT imgblock header.

    :param header: the GALFIT-generated image header, as read by astropy.io.fits.
    :return: nested dict, with keys being the component name (COMP_1, COMP_2...) and values being another level of
        dicts; for this second level, keys are the model parameter names and values are the best-fitting values,
    """
    i = 1
    components = {}
    while f"COMP_{i}" in header:
        comp_type = header[f"COMP_{i}"]
        # Use the dedicated function for this particular model type to extract fitted parameters
        component_dict = extract_funcs[comp_type](i, header)
        components[f"COMP_{i}"] = component_dict
        i += 1
    return components


def extract_sersic_params(component_n: int, header: fits.Header):
    """
    Pull the fitted parameter values for a Sersic profile from a GALFIT imgblock header.
    :param component_n: Component number (In header: COMP_N, where N is component_n)
    :param header: the GALFIT-generated image header, as read by astropy.io.fits.
    :return: dict with keys being parameter names and values fitted parameter values
    """
    x, x_err = _strip_values(header[f"{component_n}_XC"])
    y, y_err = _strip_values(header[f"{component_n}_YC"])
    mag, mag_err = _strip_values(header[f"{component_n}_MAG"])
    n, n_err = _strip_values(header[f"{component_n}_N"])
    r_eff, r_eff_err = _strip_values(header[f"{component_n}_RE"])
    axis_ratio, axis_ratio_err = _strip_values(header[f"{component_n}_AR"])
    theta, theta_err = _strip_values(header[f"{component_n}_PA"])

    component = {
        "object_type": "sersic",
        "x": x * units.pix,
        "x_err": x_err * units.pix,
        "y": y * units.pix,
        "y_err": y_err * units.pix,
        "mag": mag * units.mag,
        "mag_err": mag_err * units.mag,
        "n": n,
        "n_err": n_err,
        "r_eff": r_eff * units.pix,
        "r_eff_err": r_eff_err * units.pix,
        "axis_ratio": axis_ratio,
        "axis_ratio_err": axis_ratio_err,
        "position_angle": theta * units.deg,
        "position_angle_err": theta_err * units.deg
    }

    rotation_params = extract_rotation_params(
        component_n=component_n,
        header=header
    )
    if rotation_params:
        component["rotation_params"] = rotation_params

    return component


def extract_rotation_params(component_n: int, header: fits.Header):
    """
    Pull the fitted parameter values for a coordinate-rotated model from a GALFIT imgblock header.
    :param component_n: Component number (In header: COMP_N, where N is component_n)
    :param header: the GALFIT-generated image header, as read by astropy.io.fits.
    :return: dict with keys being parameter names and values fitted parameter values
    """
    if f"{component_n}_ROTF" not in header:
        return {}
    rot_type = header[f"{component_n}_ROTF"]
    r_in, r_in_err = _strip_values(header[f"{component_n}_RIN"])
    r_out, r_out_err = _strip_values(header[f"{component_n}_ROUT"])
    theta_out, theta_out_err = _strip_values(header[f"{component_n}_RANG"])
    theta_pa, theta_pa_err = _strip_values(header[f"{component_n}_SPA"])
    theta_inc, theta_inc_err = _strip_values(header[f"{component_n}_INCL"])

    component = {
        "rot_type": rot_type,
        "r_in": r_in * units.pix,
        "r_in_err": r_in_err * units.pix,
        "r_out": r_out * units.pix,
        "r_out_err": r_out_err * units.pix,
        "theta_out": theta_out * units.deg,
        "theta_out_err": theta_out_err * units.deg,
        "theta_pa": theta_pa * units.deg,
        "theta_pa_err": theta_pa_err * units.deg,
        "theta_inc": theta_inc * units.deg,
        "theta_inc_err": theta_inc_err * units.deg,
    }

    if rot_type == "log":
        r_ws, r_ws_err = _strip_values(header[f"{component_n}_RWS"])

        component.update({
            "r_ws": r_ws * units.pix,
            "r_ws_err": r_ws_err * units.pix,
        })

    elif rot_type == "power":
        alpha, alpha_err = _strip_values(header[f"{component_n}_ALPHA"])

        component.update({
            "alpha": alpha * units.pix,
            "alpha_err": alpha_err * units.pix,
        })

    else:
        raise ValueError(f"Unrecognised coordinate rotation type '{rot_type}'")

    return component


def extract_sky_params(component_n: int, header: fits.Header):
    """
    Pull the fitted parameter values for a sky model from a GALFIT imgblock header.
    :param component_n: Component number (In header: COMP_N, where N is component_n)
    :param header: the GALFIT-generated image header, as read by astropy.io.fits.
    :return: dict with keys being parameter names and values fitted parameter values
    """
    x, x_err = _strip_values(header[f"{component_n}_XC"])
    y, y_err = _strip_values(header[f"{component_n}_YC"])
    sky, sky_err = _strip_values(header[f"{component_n}_SKY"])
    dsky_dx, dsky_dx_err = _strip_values(header[f"{component_n}_DSDX"])
    dsky_dy, dsky_dy_err = _strip_values(header[f"{component_n}_DSDY"])

    component = {
        "x": x * units.pix,
        "x_err": x_err * units.pix,
        "y": y * units.pix,
        "y_err": y_err * units.pix,
        "sky": sky * units.ct,
        "sky_err": sky_err * units.ct,
        "dsky_dx": dsky_dx * units.ct / units.pix,
        "dsky_dx_err": dsky_dx_err * units.ct / units.pix,
        "dsky_dy": dsky_dy * units.ct / units.pix,
        "dsky_dy_err": dsky_dy_err * units.ct / units.pix,
    }
    return component


def _strip_values(string: str):
    """
    Helper function to remove [] (which specify that a parameter was fixed) and ** (which specify that a parameter is
    unreliable) from parameters in image block headers.
    :param string: string to strip.
    :return:
    """
    string = string.replace("[", "")
    string = string.replace("]", "")
    string = string.replace("*", "")
    return u.split_uncertainty_string(string)


def sersic_best_row(tbl: table.Table):
    """
    Find the best model from a table of Sersic models.
    :param tbl:
    :return:
    """
    best_index = max(np.argmin(tbl["r_eff_err"]), np.argmin(tbl["n_err"]))
    best_row = tbl[best_index]
    best_dict = dict(best_row)
    if "object_type" in best_dict:
        best_dict["object_type"] = str(best_dict["object_type"])
    return best_index, best_dict


extract_funcs = {
    "sky": extract_sky_params,
    "sersic": extract_sersic_params
}


# Here we encode the PA rotation as a function of radius, which allows GALFIT to fit spiral arms
# See the Galfit User's Manual (Peng 2010) for a detailed explanation.

def a_func(theta_out):
    """
    Calculates the value A as defined in Galfit User's Manual, Appendix A (Peng 2010)
    :param theta_out: Cumulative rotation to outer radius, in degrees.
        Corresponds to item R3) in the GALFIT feedme definition.
    :return: A (dimensionless)
    """
    cdef = 0.23 * units.rad
    a = 2 * cdef / (np.abs(theta_out) + cdef) - 1.00001
    return a


def b_func(
        r_in: units.Quantity,
        r_out: units.Quantity,
        theta_out: units.Quantity
):
    """
        Calculates the value B as defined in Galfit User's Manual, Appendix A (Peng 2010)
        :param r_in: Spiral inner radius, in pixels. If this goes extremely negative in fitting, fix to 0.
            Corresponds to item R1) in the GALFIT feedme definition.
        :param r_out: Spiral outer radius, in pixels.
            Corresponds to item R2) in the GALFIT feedme definition.
        :param theta_out: Cumulative rotation to outer radius, in degrees.
            Corresponds to item R3) in the GALFIT feedme definition.
        :return: B, in radians
        """
    a = a_func(theta_out)
    b = (2 * units.rad - np.arctanh(a)) * (r_out / (r_out - r_in))
    return b


def tanh_func(
        r_in: units.Quantity,
        r_out: units.Quantity,
        theta_out: units.Quantity,
        r: units.Quantity
):
    """
    Calculates the tanh function (not the hyperbolic tangent, but a function based around it) as defined in Galfit
    User's Manual, Appendix A (Peng 2010)
    :param r_in: Spiral inner radius, in pixels. If this goes extremely negative in fitting, fix to 0.
        Corresponds to item R1) in the GALFIT feedme definition.
    :param r_out: Spiral outer radius, in pixels.
        Corresponds to item R2) in the GALFIT feedme definition.
    :param theta_out: Cumulative rotation to outer radius, in degrees.
        Corresponds to item R3) in the GALFIT feedme definition.
    :param r: r coordinate of point(s); distance from the centre of the spiral model, in pixels.
    :return: output of tanh function
    """
    b = b_func(
        r_in=r_in,
        r_out=r_out,
        theta_out=theta_out
    )
    return 0.5 * (np.tanh((b * (r / r_out - 1) + 2 * units.rad)) + 1)


def _spiral_log(
        r_out: units.Quantity,
        r_ws: units.Quantity,
        theta_out: units.Quantity,
        r: units.Quantity,
):
    """
    Calculates coordinate rotation angle theta for a pure logarithmic spiral, based on the equation given in Galfit
    User's Manual, Appendix A (Peng 2010).
    :param r_out: Spiral outer radius, in pixels.
            Corresponds to item R2) in the GALFIT feedme definition.
    :param r_ws: winding scale radius, in pixels.
        Corresponds to item R4) in the GALFIT feedme definition.
    :param theta_out: Cumulative rotation to outer radius, in degrees.
        Corresponds to item R3) in the GALFIT feedme definition.
    :param r: r coordinate; distance from the centre of the spiral model, in pixels.
    :return: theta coordinate for pure log spiral
    """
    return theta_out * (np.log(r / r_ws + 1) / np.log(r_out / r_ws + 1))


def _spiral_log_tanh(
        r_in: units.Quantity,
        r_out: units.Quantity,
        r_ws: units.Quantity,
        theta_out: units.Quantity,
        r: units.Quantity
):
    """
    Calculates coordinate rotation angle theta for a tanh log spiral, as defined in Galfit User's Manual (Peng 2010)
    :param r_in: Spiral inner radius, in pixels. If this goes extremely negative in fitting, fix to 0.
            Corresponds to item R1) in the GALFIT feedme definition.
    :param r_out: Spiral outer radius, in pixels.
            Corresponds to item R2) in the GALFIT feedme definition.
    :param r_ws: winding scale radius, in pixels.
        Corresponds to item R4) in the GALFIT feedme definition.
    :param theta_out: Cumulative rotation to outer radius, in degrees.
        Corresponds to item R3) in the GALFIT feedme definition.
    :param r: r coordinate; distance from the centre of the spiral model, in pixels.
    :return: theta coordinate for tanh log spiral
    """
    tanh_spiral = tanh_func(
        r_in=r_in,
        r_out=r_out,
        theta_out=theta_out,
        r=r
    )
    log_spiral = _spiral_log(
        r=r,
        theta_out=theta_out,
        r_out=r_out,
        r_ws=r_ws
    )
    theta_spiral = log_spiral * tanh_spiral
    return theta_spiral


def tilt_spiral(
        theta_inc: units.Quantity,
        theta_pa: units.Quantity,
        r: units.Quantity,
        theta: units.Quantity,
):
    """
    Apply transformations to r coordinate to account for model inclination.
    :param theta_inc: inclination to line-of-sight, in degrees.
        Corresponds to item R9) in the GALFIT feedme definition.
    :param theta_pa: Position angle of the axis about which the galaxy is rotated for inclination.
        Corresponds to item R10) in the GALFIT feedme definition.
    :param r: r coordinate; distance from the centre of the spiral model, in pixels.
    :param r: theta coordinate
    :return: r_prime, appropriately transformed r coordinate
    """
    theta = theta - theta_pa
    a = r
    b = r * np.cos(theta_inc)
    # Use ellipse as projected circle onto coordinate plane
    r_prime = a * b / np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2)
    return r_prime


def _spiral_arms(
        theta: units.Quantity,
        theta_inc: units.Quantity,
        theta_pa: units.Quantity,
        position_angle: units.Quantity,
        x: units.Quantity,
        y: units.Quantity,
        r: units.Quantity,
        tilt: bool = True,
        fudge_factor: float = 1.
):
    """
    Helper function for generating spiral arm coordinates.
    :param theta: angular coordinate of point(s).
    :param theta_inc: inclination to line-of-sight, in degrees.
        Corresponds to item R9) in the GALFIT feedme definition.
    :param theta_pa: Position angle of the axis about which the galaxy is rotated for inclination.
        Corresponds to item R10) in the GALFIT feedme definition.
    :param position_angle: position angle (PA) of model, in degrees, with up=0 and left=90
        Corresponds to item 10) in the GALFIT feedme definition.
    :param x: x-position of model centre, in pixels.
        Corresponds to first component of item 1) in the GALFIT feedme definition.
    :param y: y-position of model centre, in pixels.
        Corresponds to second component of item 1) in the GALFIT feedme definition.
    :param r: r coordinate; distance from the centre of the spiral model, in pixels.
    :param tilt: If True, the spiral is transformed to account for its inclination angle.
        If False, the output spiral is equivalent to theta_inc == 0 deg.
    :param fudge_factor: Sometimes, infuriatingly, the
    :return: theta_1, x_1, y_1, theta_2, x_2, y_2, where 1 and 2 refer to each spiral arm.
    """
    # Rotate to GALFIT coordinates
    theta_1 = theta + position_angle - 90 * units.deg
    # Set up opposite spiral arm
    theta_2 = theta_1 - 180 * units.deg
    # Account for inclination of model
    if tilt:
        r_prime = tilt_spiral(
            r=r,
            theta_inc=theta_inc,
            theta=theta_1,
            theta_pa=theta_pa
        ) * fudge_factor
    else:
        r_prime = r
    # Calculate x and y coordinates of spiral points
    x_1, y_1 = u.polar_to_cartesian(
        theta=theta_1,
        r=r_prime,
        centre_x=x,
        centre_y=y
    )
    x_2, y_2 = u.polar_to_cartesian(
        theta=theta_2,
        r=r_prime,
        centre_x=x,
        centre_y=y
    )
    theta_out = np.append(np.flip(theta_2), theta_1)
    x_out = np.append(np.flip(x_2), x_1)
    y_out = np.append(np.flip(y_2), y_1)
    return theta_out, x_out, y_out


def spiral_log_tanh(
        r_in: units.Quantity,
        r_out: units.Quantity,
        r_ws: units.Quantity,
        theta_inc: units.Quantity,
        theta_pa: units.Quantity,
        theta_out: units.Quantity,
        position_angle: units.Quantity,
        x: units.Quantity,
        y: units.Quantity,
        r: units.Quantity,
        tilt: bool = True,
):
    """
    Calculates theta, x and y (image coordinates) for a GALFIT tanh log spiral, as defined in Galfit User's Manual
    (Peng 2010), including tilt.
    :param r_in: Spiral inner radius, in pixels. If this goes extremely negative in fitting, fix to 0.
            Corresponds to item R1) in the GALFIT feedme definition.
    :param r_out: Spiral outer radius, in pixels.
            Corresponds to item R2) in the GALFIT feedme definition.
    :param r_ws: winding scale radius, in pixels.
        Corresponds to item R4) in the GALFIT feedme definition.
    :param theta_inc: inclination to line-of-sight, in degrees.
        Corresponds to item R9) in the GALFIT feedme definition.
    :param theta_pa: Position angle of the axis about which the galaxy is rotated for inclination.
        Corresponds to item R10) in the GALFIT feedme definition.
    :param theta_out: Cumulative rotation to outer radius, in degrees.
        Corresponds to item R3) in the GALFIT feedme definition.
    :param position_angle: position angle (PA) of model, in degrees, with up=0 and left=90
        Corresponds to item 10) in the GALFIT feedme definition.
    :param x: x-position of model centre, in pixels.
        Corresponds to first component of item 1) in the GALFIT feedme definition.
    :param y: y-position of model centre, in pixels.
        Corresponds to second component of item 1) in the GALFIT feedme definition.
    :param r: r coordinate; distance from the centre of the spiral model, in pixels.
    :param tilt: If True, the spiral is transformed to account for its inclination angle.
        If False, the output spiral is equivalent to theta_inc == 0 deg.
    :return: theta_1, x_1, y_1, theta_2, x_2, y_2, where 1 and 2 refer to each spiral arm.
    """
    theta = _spiral_log_tanh(
        r_in=r_in,
        r_out=r_out,
        r_ws=r_ws,
        theta_out=theta_out,
        r=r
    )
    # Generate coordinates of spiral arms
    theta_out, x_out, y_out = _spiral_arms(
        theta=theta,
        position_angle=position_angle,
        theta_inc=theta_inc,
        theta_pa=theta_pa,
        x=x,
        y=y,
        r=r,
        tilt=tilt
    )
    return theta_out, x_out, y_out


def spiral_log(
        r_out: units.Quantity,
        r_ws: units.Quantity,
        theta_inc: units.Quantity,
        theta_pa: units.Quantity,
        theta_out: units.Quantity,
        position_angle: units.Quantity,
        x: units.Quantity,
        y: units.Quantity,
        r: units.Quantity,
        tilt: bool = True,
):
    """
    Calculates theta, x and y (image coordinates) for a pure log spiral, based on equations in Galfit User's Manual
    (Peng 2010), including tilt.
    :param r_out: Spiral outer radius, in pixels.
            Corresponds to item R2) in the GALFIT feedme definition.
    :param r_ws: winding scale radius, in pixels.
        Corresponds to item R4) in the GALFIT feedme definition.
    :param theta_inc: inclination to line-of-sight, in degrees.
        Corresponds to item R9) in the GALFIT feedme definition.
    :param theta_pa: Position angle of the axis about which the galaxy is rotated for inclination.
        Corresponds to item R10) in the GALFIT feedme definition.
    :param theta_out: Cumulative rotation to outer radius, in degrees.
        Corresponds to item R3) in the GALFIT feedme definition.
    :param position_angle: position angle (PA) of model, in degrees, with up=0 and left=90
        Corresponds to item 10) in the GALFIT feedme definition.
    :param x: x-position of model centre, in pixels.
        Corresponds to first component of item 1) in the GALFIT feedme definition.
    :param y: y-position of model centre, in pixels.
        Corresponds to second component of item 1) in the GALFIT feedme definition.
    :param r: r coordinate; distance from the centre of the spiral model, in pixels.
    :param tilt: If True, the spiral is transformed to account for its inclination angle.
        If False, the output spiral is equivalent to theta_inc == 0 deg.
    :return: theta_1, x_1, y_1, theta_2, x_2, y_2, where 1 and 2 refer to each spiral arm.
    """
    theta = _spiral_log(
        theta_out=theta_out,
        r_out=r_out,
        r_ws=r_ws,
        r=r,
    )
    # Generate coordinates of spiral arms
    theta_out, x_out, y_out = _spiral_arms(
        theta=theta,
        position_angle=position_angle,
        theta_inc=theta_inc,
        theta_pa=theta_pa,
        x=x,
        y=y,
        r=r,
        tilt=tilt
    )
    return theta_out, x_out, y_out


def spiral_from_model_dict(
        model_dict: dict,
        r: units.Quantity,
        **kwargs
):
    """
    A wrapper for spiral_log_tanh() that does the work of unpacking the model dict returned by extract_fit_params.
    :param model_dict: dict containing output model param names as keys; the dictionaries generated by
        extract_fit_params corresponding to "COMP_N" will work as they are, so long as they have rotation parameters.
    :param r: r coordinate of point(s); distance from the centre of the spiral model, in pixels.
    :param kwargs: Any other keywords you wish to pass to the spiral_log_tanh function. Warning: if keys overlap with
        model_dict, the model_dict entries will be overwritten by the kwargs entries.
    :return:
    """
    rot_type = model_dict["rotation_params"]["rot_type"]
    if rot_type == "log":
        return spiral_log_tanh(
            r=r,
            r_in=model_dict["rotation_params"]["r_in"],
            r_out=model_dict["rotation_params"]["r_out"],
            r_ws=model_dict["rotation_params"]["r_ws"],
            theta_inc=model_dict["rotation_params"]["theta_inc"],
            theta_pa=model_dict["rotation_params"]["theta_pa"],
            theta_out=model_dict["rotation_params"]["theta_out"],
            position_angle=model_dict["position_angle"],
            x=model_dict["x"],
            y=model_dict["y"],
            **kwargs
        )
    elif rot_type == "power":
        raise ValueError("power law spirals are not yet supported (but hopefully will be soon)")
    else:
        raise ValueError(f"Coordinate rotation type {rot_type} not recognised.")


def imgblock_plot(
        img_block: Union[fits.HDUList, str],
        output: str = None,
        # frame:
):
    if isinstance(img_block, str):
        img_block = fits.open(img_block)

    fitsect = img_block[2].header["FITSECT"][1:-1]
    x_sect, y_sect = fitsect.split(",")
    x_left, _ = x_sect.split(":")
    x_left = int(x_left)
    y_bottom, _ = y_sect.split(":")
    y_bottom = int(y_bottom)

    fig = plt.figure(figsize=(24, 12))

    if len(img_block) > 4:
        vimg = img_block[4].data

    else:
        vimg = img_block[1].data
    max_val = np.max(vimg) - np.median(vimg)

    names = (
        "?",
        "Data",
        "Model",
        "Data - Model",
        "Masked Data",
        "Data - Model, Masked"
    )

    from matplotlib.patches import Ellipse

    params = extract_fit_params(img_block[2].header)["COMP_2"]
    x = params["x"].value - x_left
    y = params["y"].value - y_bottom
    r_eff = params["r_eff"].value
    a = r_eff * 2
    b = r_eff * params["axis_ratio"] * 2
    theta = params["position_angle"].value

    for i, im in enumerate(img_block):
        if i == 0:
            continue
        if im.is_image:
            ax = fig.add_subplot(1, len(img_block), i + 1)
            ax.set_title(names[i])
            ax.imshow(
                im.data - np.median(im.data),
                origin="lower",
                norm=ImageNormalize(
                    # vmax=max_val + np.median(im.data),
                    # vmin=np.median(im.data) - 2 * np.std(im.data),
                    stretch=SqrtStretch()
                ),
                cmap="cmr.bubblegum"
            )
            ax.errorbar(
                x, y,
                marker="x",
                xerr=params["x_err"].value,
                yerr=params["y_err"].value,
                c="black"
            )
            e = Ellipse(
                xy=(x, y),
                width=a,
                height=b,
                angle=theta + 90,
                edgecolor="white",
                facecolor="none",
            )
            ax.add_artist(e)

    if isinstance(output, str):
        fig.savefig(output, dpi=100, bbox_inches="tight")

    return fig
