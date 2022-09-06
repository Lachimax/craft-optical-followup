import shutil
import os
from typing import Tuple, Union, List

import numpy as np

import astropy.io.fits as fits
import astropy.units as units
import astropy.table as table

import craftutils.params as p
import craftutils.utils as u


def galfit(config: str, output_dir: str):
    cwd = os.getcwd()
    os.chdir(output_dir)
    u.system_command_verbose(f"galfit {config}")
    os.chdir(cwd)


def feedme_sky_model(
        background_center: float = (1.3920, True),
        gradient_x: Union[float, Tuple[float, bool]] = (0.0, True),
        gradient_y: Union[float, Tuple[float, bool]] = (0.0, True),
        output_option: int = 0
):
    return feedme_model(
        "sky",
        output_option,
        None,
        (False, False),
        background_center,
        gradient_x,
        gradient_y
    )


def feedme_sersic_model(
        x: float, y: float,
        int_mag: Union[float, Tuple[float, bool]],
        r_e: Union[float, Tuple[float, bool]] = (3., True),
        n: Union[float, Tuple[float, bool]] = (1., True),
        axis_ratio: Union[float, Tuple[float, bool]] = (0.5, True),
        position_angle: Union[float, Tuple[float, bool]] = (0., True),
        output_option: int = 0,
        fit_position: Union[bool, Tuple[bool, bool]] = (True, True),
        rot_kwargs: dict = {},
        **kwargs
):
    lines = feedme_model(
        "sersic",
        output_option,
        (x, y),
        fit_position,
        (0., False),
        int_mag,
        r_e,
        n,
        (0., False),
        (0., False),
        (0., False),
        axis_ratio,
        position_angle,
    )

    if rot_kwargs:
        rot_lines = feedme_rot(
            **rot_kwargs
        )
        lines.insert(-1, rot_lines)

    return lines


def feedme_model(
        object_type: str,
        output_option: int,
        position: Tuple[float, float],
        fit_position: Union[bool, Tuple[bool, bool]],
        *args,
):
    """
    Generate the input file lines for an arbitrary model for use with GALFIT.
    :param object_type: string indicating the type of model, to be parsed by GALFIT.
    :param output_option:
    :param position:
    :param args: (value, fit), with value being the initial guess for a parameter and fit being a boolean indicating
        whether this parameter should be fit for (True) or held fixed (False).
        If value is provided without fit, fit defaults to True.
    :return: list of lines, suitable for use with open().writelines().
    """
    i = 0
    lines = [f"{i}) {object_type}\n"]
    i += 1
    if position is not None:
        try:
            # See if we can get the position as a tuple
            fit_x, fit_y = fit_position
        except TypeError:
            # If not, assume it's a single value that both x and y adopt
            fit_x = fit_y = fit_position
        lines.append(f"{i}) {position[0]} {position[1]} {int(fit_x)} {int(fit_y)}\n")
        i += 1
    lines += _feedme_lines(i, "", *args)

    lines.append(f"Z) {output_option}\n")
    return lines


def _feedme_lines(i_start: int = 0, prefix: str = "", *args):
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


def feedme_rot(
        rot_type: str = "log",
        r_in: Union[float, Tuple[float, bool]] = (0., True),
        r_out: Union[float, Tuple[float, bool]] = (5., True),
        theta_out: Union[float, Tuple[float, bool]] = (180., True),
        r_ws: Union[float, Tuple[float, bool]] = (2., True),
        theta_inc: Union[float, Tuple[float, bool]] = (45., True),
        theta_pa: Union[float, Tuple[float, bool]] = (45., True),
):
    i = 0
    lines = [f"{i}) {rot_type}\n"]
    i += 1
    args = (
        r_in,
        r_out,
        theta_out,
        r_ws,
        (0, False),
        (0, False),
        (0, False),
        (0, False),
        theta_inc,
        theta_pa
    )

    lines += _feedme_lines(i, "R", *args)
    return lines


feedme_funcs = {
    "sersic": feedme_sersic_model
}


def galfit_feedme(
        feedme_path: str,
        input_file: str,
        output_file: str,
        zeropoint: float,
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
    Any unset values will be left as the GALFIT defaults (see param/galfit/galfit.feedme)
    :param feedme_path:
    :param input_file:
    :param output_file:
    :param sigma_file:
    :return:
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
        f"J) {zeropoint}\n",
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
        lines += model_lines
        lines.append("\n")
        i += 1

    lines += "===============================================================================\n\n"

    u.rm_check(feedme_path)
    with open(feedme_path, "w") as output:
        output.writelines(lines)
    return lines


def extract_fit_params(header: fits.Header):
    """
    Extract the fitted parameters for all components from a GALFIT imgblock header.
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


def strip_values(string: str):
    string = string.replace("[", "")
    string = string.replace("]", "")
    string = string.replace("*", "")
    return u.split_uncertainty_string(string)


def extract_sersic_params(component_n: int, header: fits.Header):
    """
    Pull the fitted parameter values for a Sersic profile from a GALFIT imgblock header.
    :param component_n: Component number (In header: COMP_N, where N is component_n)
    :param header: the GALFIT-generated image header, as read by astropy.io.fits.
    :return: dict with keys being parameter names and values fitted parameter values
    """
    x, x_err = strip_values(header[f"{component_n}_XC"])
    y, y_err = strip_values(header[f"{component_n}_YC"])
    mag, mag_err = strip_values(header[f"{component_n}_MAG"])
    n, n_err = strip_values(header[f"{component_n}_N"])
    r_eff, r_eff_err = strip_values(header[f"{component_n}_RE"])
    axis_ratio, axis_ratio_err = strip_values(header[f"{component_n}_AR"])
    theta, theta_err = strip_values(header[f"{component_n}_PA"])

    component = {
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
    component.update(
        extract_rotation_params(
            component_n=component_n,
            header=header
        ))
    return component


def extract_rotation_params(component_n: int, header: fits.Header):
    if f"{component_n}_ROTF" not in header:
        return {}
    rot_type = header[f"{component_n}_ROTF"]
    r_in, r_in_err = strip_values(header[f"{component_n}_RIN"])
    r_out, r_out_err = strip_values(header[f"{component_n}_ROUT"])
    theta_out, theta_out_err = strip_values(header[f"{component_n}_RANG"])
    r_ws, r_ws_err = strip_values(header[f"{component_n}_RWS"])
    theta_inc, theta_inc_err = strip_values(header[f"{component_n}_INCL"])
    theta_pa, theta_pa_err = strip_values(header[f"{component_n}_SPA"])

    component = {
        "rot_type": rot_type,
        "r_in": r_in * units.pix,
        "r_in_err": r_in_err * units.pix,
        "r_out": r_out * units.pix,
        "r_out_err": r_out_err * units.pix,
        "theta_out": theta_out * units.deg,
        "theta_out_err": theta_out_err * units.deg,
        "r_ws": r_ws * units.pix,
        "r_ws_err": r_ws_err * units.pix,
        "theta_inc": theta_inc * units.deg,
        "theta_inc_err": theta_inc_err * units.deg,
        "theta_pa": theta_pa * units.deg,
        "theta_pa_err": theta_pa_err * units.deg,
    }
    return component


def extract_sky_params(component_n: int, header: fits.Header):
    x, x_err = strip_values(header[f"{component_n}_XC"])
    y, y_err = strip_values(header[f"{component_n}_YC"])
    sky, sky_err = strip_values(header[f"{component_n}_SKY"])
    dsky_dx, dsky_dx_err = strip_values(header[f"{component_n}_DSDX"])
    dsky_dy, dsky_dy_err = strip_values(header[f"{component_n}_DSDY"])

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


def sersic_best_row(tbl: table.Table):
    best_index = max(np.argmin(tbl["r_eff_err"]), np.argmin(tbl["n_err"]))
    best_row = tbl[best_index]
    best_dict = dict(best_row)
    return best_dict


extract_funcs = {
    "sky": extract_sky_params,
    "sersic": extract_sersic_params
}
# def galfit_best(path: str):
#
