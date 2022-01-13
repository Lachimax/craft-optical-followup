import os
import shutil

import astropy.units as units
import pytest

import craftutils.utils as u
import craftutils.params as p

test_file_path = os.path.join(p.project_path, "tests", "files")
coadd_path = os.path.join(test_file_path, "test.coadd1d")
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


def test_check_quantity():
    number = 10.
    assert u.check_quantity(number=number, unit=units.meter) == number * units.meter
    number = 10. * units.meter
    assert u.check_quantity(number=number, unit=units.meter) == number
    number = 1000. * units.centimeter
    assert u.check_quantity(number=number, unit=units.meter, convert=True) == 10. * units.meter
    assert u.check_quantity(number=number, unit=units.meter, convert=True).unit == units.meter
    assert u.check_quantity(number=number, unit=units.meter, convert=False).unit == units.centimeter
    with pytest.raises(units.UnitsError) as e:
        u.check_quantity(number=number, unit=units.meter, allow_mismatch=False)
    assert e.type is units.UnitsError
    with pytest.raises(units.UnitsError) as e:
        u.check_quantity(number=number, unit=units.joule)
    assert e.type is units.UnitsError


def test_mkdir_check():
    path = os.path.join(test_file_path, "path_test")
    u.rm_check(path)
    u.mkdir_check(path)
    assert os.path.isdir(path)
    u.rmtree_check(path)

    paths = [
        os.path.join(test_file_path, "test_path_1"),
        os.path.join(test_file_path, "test_path_2"),
        os.path.join(test_file_path, "test_path_3")
    ]
    u.mkdir_check_args(*paths)
    for path in paths:
        assert os.path.isdir(path)
        u.rmtree_check(path)


def test_mkdir_check_args():
    dirs = [test_file_path, "path_test", "nested", "and_then"]
    u.rm_check(os.path.join(test_file_path, "path_test"))
    path_out = os.path.join(*dirs)
    path_test = u.mkdir_check_args(*dirs)
    assert path_out == path_test
    assert os.path.isdir(path_test)
    shutil.rmtree(os.path.join(test_file_path, "path_test"))


def test_mkdir_check_nested():
    u.debug_level = 2
    path_test = os.path.join(test_file_path, "path_test", "nested", "and_then")
    u.rm_check(os.path.join(test_file_path, "path_test"))
    u.mkdir_check_nested(path_test, remove_last=False)
    assert os.path.isdir(path_test)
    u.rmtree_check(os.path.join(test_file_path, "path_test"))

    path_test = os.path.join(test_file_path, "path_test", "nested", "and_then", "/")
    u.mkdir_check_nested(path_test)
    assert os.path.isdir(path_test)
    u.rmtree_check(os.path.join(test_file_path, "path_test"))

    path_test = os.path.join(test_file_path, "path_test", "nested", "and_then", "gaia.csv")
    u.mkdir_check_nested(path_test)
    assert os.path.isdir(os.path.split(path_test)[0])
    assert not os.path.isfile(path_test)
    u.rmtree_check(os.path.join(test_file_path, "path_test"))


def test_rmtree_check():
    list_test = [test_file_path, "path_test", "nested", "and_then"]
    path_test = u.mkdir_check_args(*list_test)
    u.rmtree_check(os.path.join(test_file_path, "path_test"))
    assert not os.path.isdir(path_test)


def test_dequantify():
    number = 10.
    assert u.dequantify(number=number) == 10.
    number = 10. * units.meter
    assert u.dequantify(number=number) == 10.
    number = 1000. * units.centimeter
    assert u.dequantify(number=number, unit=units.meter) == 10.

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
