import astropy.units as units

from .objects import ObjectCatalogue


class OpticalCatalogue(ObjectCatalogue):
    @classmethod
    def column_names(cls):
        columns = super().column_names()
        columns.update({
            "epoch_position": str,
            "epoch_position_date": str,
            "a_err": units.arcsec,
            "b_err": units.arcsec,
            "theta_err": units.deg,
            "epoch_ellipse": str,
            "epoch_ellipse_date": str,
            "kron_radius": float,
            "e_b-v": units.mag,
            "class_star": float,
            "spread_model": float,
            "spread_model_err": float,
            "class_flag": int,
        })
        return columns
