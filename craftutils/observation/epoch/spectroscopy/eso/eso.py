import shutil
import os

from typing import Union

from astropy.time import Time

import craftutils.utils as u
import craftutils.observation.image as image

from ..spectroscopy_epoch import SpectroscopyEpoch
from ...epoch import _retrieve_eso_epoch


class ESOSpectroscopyEpoch(SpectroscopyEpoch):
    def __init__(
            self,
            param_path: str = None,
            name: str = None,
            field: Union[str, 'fld.Field'] = None,
            data_path: str = None,
            instrument: str = None,
            date: Union[str, Time] = None,
            program_id: str = None,
            grism: str = None,
            **kwargs
    ):
        super().__init__(
            param_path=param_path,
            name=name,
            field=field,
            data_path=data_path,
            instrument=instrument,
            date=date,
            program_id=program_id,
            grism=grism,
            **kwargs
        )
        # Data reduction paths

    def proc_download(self, output_dir: str, **kwargs):

        # Check for alternate directory.
        alt_dir = None
        if "alternate_dir" in kwargs and isinstance(kwargs["alternate_dir"], str):
            alt_dir = kwargs["alternate_dir"]

        if alt_dir is None:
            r = self.retrieve(output_dir)
            if r:
                return True
        else:
            u.rmtree_check(output_dir)
            shutil.copytree(alt_dir, output_dir)
            return True

    def _initial_setup(self, output_dir: str, **kwargs):
        raw_dir = self.paths["download"]
        m_path = os.path.join(raw_dir, "M")
        u.mkdir_check(m_path)
        os.system(f"mv {os.path.join(self.paths['raw_dir'], 'M.')}* {m_path}")
        image.fits_table_all(
            input_path=raw_dir,
            output_path=os.path.join(self.data_path, f"{self.name}_fits_table_science.csv")
        )
        image.fits_table_all(
            input_path=raw_dir,
            output_path=os.path.join(self.data_path, f"{self.name}_fits_table_all.csv"),
            science_only=False
        )

    def retrieve(self, output_dir: str):
        """
        Check ESO archive for the epoch raw frames, and download those frames and associated files.
        :return:
        """
        r = []
        r = _retrieve_eso_epoch(self, path=output_dir)
        return r

    @classmethod
    def stages(cls):
        spec_stages = super().stages()
        stages = {
            "download": {
                "method": cls.proc_download,
                "message": "Download raw data from ESO archive?",
                "default": True,
                "keywords": {
                    "alternate_dir": None
                }
            },
            "initial_setup": spec_stages["initial_setup"],
            "pypeit_setup": spec_stages["pypeit_setup"],
            "pypeit_run": spec_stages["pypeit_run"],
            "pypeit_flux": spec_stages["pypeit_flux"],
            "pypeit_coadd": spec_stages["pypeit_coadd"],
            "convert_to_marz_format": spec_stages["convert_to_marz_format"]
        }
        return stages
