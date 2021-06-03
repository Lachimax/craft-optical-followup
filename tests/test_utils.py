import os

import craftutils.utils as u
import craftutils.params as p


def test_scan_nested_dict():
    dictionary = {"calibrations": {"traceframe": {"process": {"use_darkimage": "True"},
                                                  },
                                   "illumflatframe": {"process": {"use_darkimage": "True"},
                                                      },
                                   "pixelflatframe": {"process": {"use_darkimage": "True"}
                                                      },
                                   },
                  "coadd1d": {"sensfuncfile": "YOUR_SENSFUNC_FILE"}
                  }
    assert u.scan_nested_dict(dictionary=dictionary,
                              keys=["calibrations", "illumflatframe", "process", "use_darkimage"]) == "True"
    assert u.scan_nested_dict(dictionary=dictionary, keys=["coadd1d"]) == {"sensfuncfile": "YOUR_SENSFUNC_FILE"}


def test_get_pypeit_user_params():
    files = [os.path.join(p.project_path, "tests", "files", "test.coadd1d"),
             os.path.join(p.project_path, "tests", "files", "test.pypeit")]
    results = [{"coadd1d": {"coaddfile": "foreground_coadded.fits",
                            "sensfuncfile": "YOUR_SENSFUNC_FILE",
                            "wave_method": "linear"}},
               {"calibrations": {"traceframe": {"process": {"use_darkimage": "True"},
                                                },
                                 "illumflatframe": {"process": {"use_darkimage": "True"},
                                                    },
                                 "pixelflatframe": {"process": {"use_darkimage": "True"}
                                                    },
                                 }}]
    for i, file in enumerate(files):
        with open(file) as f:
            lines = f.readlines()
        result = u.get_pypeit_user_params(lines)
        assert result == results[i]
