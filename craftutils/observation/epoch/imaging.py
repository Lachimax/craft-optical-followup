import os
from typing import Union

from astropy.time import Time

import craftutils.observation.epoch.epoch as epoch
import craftutils.observation.field as fld
import craftutils.observation.objects as objects
import craftutils.params as p
import craftutils.retrieve as retrieve
import craftutils.utils as u

epoch_stage_dirs = ["0-data_with_raw_calibs"]


class ImagingEpoch(epoch.Epoch):
    def __init__(self,
                 name: str = None,
                 field: fld.Field = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 standard_epochs: list = None):
        super().__init__(name=name, field=field, data_path=data_path, instrument=instrument, date=date)

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        default_params.update({
            "sextractor":
                {"aperture_diameters": [7.72],
                 "dual_mode": True,
                 "threshold": 1.5,
                 "kron_factor": 3.5,
                 "kron_radius_min": 1.0
                 },
            "calibration":
                {"star_class_tolerance": 0.95,
                 },
            "background_subtraction":
                {"renormalise_centre": objects.position_dictionary.copy(),
                 "test_synths":
                     {[{"position": objects.position_dictionary.copy(),
                        "mags": {}
                        }]
                      }
                 },
            "skip":
                {"esoreflex_copy": False,
                 "sextractor_individual": False,
                 "astrometry_net": False,
                 "sextractor": False,
                 "esorex": False,
                 },
        })
        return default_params

    @classmethod
    def new_yaml(cls, name: str, path: str = None, quiet: bool = False):
        param_dict = cls.default_params()
        param_dict["name"] = name
        if path is not None:
            path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict


class ESOImagingEpoch(ImagingEpoch):

    def __init__(self,
                 name: str = None,
                 field: fld.Field = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 standard_epochs: list = None):
        super().__init__(name=name, field=field, data_path=data_path, instrument=instrument, date=date,
                         standard_epochs=standard_epochs)

    def retrieve(self):
        """
        Check ESO archive for the epoch raw frames, and download those frames and associated files.
        :return:
        """
        raw_path = os.path.join(self.data_path, epoch_stage_dirs[0])
        u.mkdir_check(raw_path)
        r = retrieve.save_eso_raw_data_and_calibs(output=raw_path, date_obs=self.date, obj=self.obj,
                                                  program_id=self.program_id, instrument=self.instrument)
        os.system(f"uncompress {raw_path}/*.Z")
        return r


class FORS2ImagingEpoch(ESOImagingEpoch):
    @classmethod
    def convert_old_params(cls, epoch_name: str):
        new_params = cls.new_yaml(name=epoch_name, path=None)
        old_params = p.object_params_fors2(epoch_name)

        new_params["instrument"] = "FORS2"
        new_params["data_path"] = old_params["data_dir"]
        new_params["field"] = epoch_name[:epoch_name.find("_")]

        new_params["sextractor"]["aperture_diameters"] = old_params["photometry_apertures"]
        new_params["sextractor"]["dual_mode"] = old_params["do_dual_mode"]
        new_params["sextractor"]["threshold"] = old_params["threshold"]
        new_params["sextractor"]["kron_factor"] = old_params["sextractor_kron_radius"]
        new_params["sextractor"]["kron_radius_min"] = old_params["sextractor_min_radius"]

        new_params["calibration"]["star_class_tolerance"] = old_params["star_class_tolerance"]

        new_params["background_subtraction"]["renormalise_centre"]["dec"] = old_params["renormalise_centre_dec"]
        new_params["background_subtraction"]["renormalise_centre"]["ra"] = old_params["renormalise_centre_ra"]
        new_params["background_subtraction"]["test_synths"] = []
        for i, _ in enumerate(old_params["test_synths"]):
            synth_dict = {}
            synth_dict["position"]["ra"] = old_params["test_synths"]["ra"][i]
            synth_dict["position"]["dec"] = old_params["test_synths"]["dec"][i]
            synth_dict["mags"]["g"] = old_params["test_synths"]["g_mag"][i]
            synth_dict["mags"]["I"] = old_params["test_synths"]["I_mag"][i]
            new_params["background_subtraction"]["test_synths"].append(synth_dict)

        new_params["skip"]["esoreflex_copy"] = old_params["skip_copy"]
        new_params["skip"]["sextractor_individual"] = not old_params["do_sextractor_individual"]
        new_params["skip"]["astrometry_net"] = old_params["skip_astrometry"]
        new_params["skip"]["sextractor"] = not old_params["do_sextractor"]
        new_params["skip"]["esorex"] = old_params["skip_esorex"]


class Image:
    frame_type = "stacked"
