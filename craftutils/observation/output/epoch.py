import os.path

import astropy.units as units

import craftutils.params as p
from .output import OutputCatalogue


class EpochCatalogue(OutputCatalogue):

    def add_epoch(self, epoch_name: str, fil_name: str, entry: dict):
        self.load_table()
        key = self.build_key(epoch_name=epoch_name, fil_name=fil_name)
        self.add_entry(key=key, entry=entry)

    def get_epoch(self, epoch_name: str, fil_name: str):
        self.load_table()
        key = self.build_key(epoch_name=epoch_name, fil_name=fil_name)
        if key in self.table:
            return self.table[key]
        else:
            return None

    @classmethod
    def build_key(cls, epoch_name: str, fil_name: str):
        return f"{epoch_name}_{fil_name}"

    @classmethod
    def column_names(cls):
        columns = super().column_names()
        columns.update({
            "epoch_name": str,
            "filter_name": str,
            "instrument": str,
            "date_utc": str,
            "mjd": units.day,
            "filter_lambda_eff": units.Angstrom,
            "n_frames": int,
            "n_frames_included": int,
            "frame_exp_time": units.second,
            "total_exp_time": units.second,
            "total_exp_time_included": units.second,
            "psf_fwhm": units.arcsec,
            "program_id": str,
            "zeropoint": units.mag,
            "zeropoint_err": units.mag,
            "zeropoint_source": str,
            "last_processed": str,
            "depth": units.mag,
            "extinction_atm": units.mag,
            "extinction_atm_err": units.mag,
        })

    @classmethod
    def required(cls):
        required = super().required()
        required += [
            "epoch_name",
            "filter_name",
            "instrument"
        ]
        return required


# if "table_dir" in p.config and isinstance(p.config["table_dir"], str):
#     os.makedirs(p.config["table_dir"], exist_ok=True)
imaging_table = EpochCatalogue(name="master_imaging_table")
