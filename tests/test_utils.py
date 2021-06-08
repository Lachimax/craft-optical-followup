import os

import craftutils.utils as u
import craftutils.params as p

coadd_path = os.path.join(p.project_path, "tests", "files", "test.coadd1d")
coadd_dictionary = {"coadd1d": {"coaddfile": "foreground_coadded.fits",
                                "sensfuncfile": "YOUR_SENSFUNC_FILE",
                                "wave_method": "linear"}}
coadd_lines_stripped = ["coadd1d",
                        "coaddfile=foreground_coadded.fits",
                        "sensfuncfile=YOUR_SENSFUNC_FILE",
                        "wave_method=linear"]

with open(coadd_path) as file:
    coadd_lines = file.readlines()
    coadd_param_lines = coadd_lines[4:8]

pypeit_path = os.path.join(p.project_path, "tests", "files", "test.pypeit")
pypeit_dictionary = {"calibrations": {"traceframe": {"process": {"use_darkimage": "True"},
                                                     },
                                      "illumflatframe": {"process": {"use_darkimage": "True"},
                                                         },
                                      "pixelflatframe": {"process": {"use_darkimage": "True"}
                                                         },
                                      },
                     "rdx": {"spectrograph": "vlt_xshooter_nir"}
                     }
pypeit_lines_stripped = ["calibrations",
                         "traceframe",
                         "process",
                         "use_darkimage=True",
                         "illumflatframe",
                         "process",
                         "use_darkimage=True",
                         "pixelflatframe",
                         "process",
                         "use_darkimage=True",
                         "rdx",
                         "spectrograph=vlt_xshooter_nir"]
with open(os.path.join(pypeit_path)) as file:
    pypeit_lines = file.readlines()
    pypeit_param_lines = pypeit_lines[4:16]


def test_scan_nested_dict():
    assert u.scan_nested_dict(dictionary=pypeit_dictionary.copy(),
                              keys=["calibrations", "illumflatframe", "process", "use_darkimage"]) == "True"


def test_get_pypeit_param_levels():
    assert u.get_pypeit_param_levels(pypeit_param_lines.copy()) == (
        pypeit_lines_stripped, [1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 1, 2])
    assert u.get_pypeit_param_levels(coadd_param_lines.copy()) == (coadd_lines_stripped, [1, 2, 2, 2])


def test_get_scope():
    lines, levels = u.get_pypeit_param_levels(pypeit_param_lines.copy())
    assert u.get_scope(levels=levels, lines=lines) == pypeit_dictionary
    lines, levels = u.get_pypeit_param_levels(coadd_param_lines.copy())
    assert u.get_scope(levels=levels, lines=lines) == coadd_dictionary


def test_get_pypeit_user_params():
    assert u.get_pypeit_user_params(file=coadd_path) == coadd_dictionary
    assert u.get_pypeit_user_params(file=pypeit_path) == pypeit_dictionary


def test_uncertainty_log10():
    flux = 4051519.0
    flux_err = 28118.24
    assert u.uncertainty_log10(arg=flux, uncertainty_arg=flux_err, a=-2.5) == 0.00753519635032644

# def test_get_pypeit_user_params():
#     files = [coadd_path,
#              pypeit_path]
#     expected = [coadd_dictionary,
#                 pypeit_dictionary]
#     for i, file in enumerate(files):
#         with open(file) as f:
#             lines = f.readlines()
#         result = u.get_pypeit_user_params(lines)
#         assert result == expected[i]
