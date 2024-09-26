import os

import numpy as np

import astropy.units as units
import astropy.table as table
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord

import craftutils
import craftutils.utils as u
import craftutils.observation.sed as sed

from .extragalactic import Extragalactic, cosmology


@u.export
class Galaxy(Extragalactic):
    optical = True
    radio = True

    def __init__(
            self,
            z: float = None,
            **kwargs
    ):
        super().__init__(
            z=z,
            **kwargs
        )

        self.mass = None
        if "mass" in kwargs:
            self.mass = kwargs["mass"]

        self.log_mass_stellar = None
        if "mass_stellar" in kwargs:
            self.log_mass_stellar = np.log10(u.check_quantity(kwargs["mass_stellar"], units.solMass).value)
        elif "log_mass_stellar" in kwargs:
            self.log_mass_stellar = float(kwargs["log_mass_stellar"])

        self.log_mass_stellar_err_plus = None
        self.log_mass_stellar_err_minus = None
        if "mass_stellar_err" in kwargs:
            self.log_mass_stellar_err_plus = u.uncertainty_power_2(
                x=self.log_mass_stellar,
                base=10,
                sigma_x=float(kwargs["mass_stellar_err"]),
            )
        elif "mass_stellar_err_plus" in kwargs:
            self.log_mass_stellar_err_plus = u.uncertainty_power_2(
                x=self.log_mass_stellar,
                base=10,
                sigma_x=float(kwargs["mass_stellar_err_plus"]),
            )
        elif "log_mass_stellar_err_plus" in kwargs:
            self.log_mass_stellar_err_plus = float(kwargs["log_mass_stellar_err_plus"])
        elif "log_mass_stellar_err" in kwargs:
            self.log_mass_stellar_err_plus = float(kwargs["log_mass_stellar_err"])

        if "mass_stellar_err" in kwargs:
            self.log_mass_stellar_err_minus = u.uncertainty_power_2(
                x=self.log_mass_stellar,
                base=10,
                sigma_x=float(kwargs["mass_stellar_err"]),
            )
        elif "mass_stellar_err_minus" in kwargs:
            self.log_mass_stellar_err_minus = u.uncertainty_power_2(
                x=self.log_mass_stellar,
                base=10,
                sigma_x=float(kwargs["mass_stellar_err_minus"]),
            )
        elif "log_mass_stellar_err_minus" in kwargs:
            self.log_mass_stellar_err_minus = float(kwargs["log_mass_stellar_err_minus"])
        elif "log_mass_stellar_err" in kwargs:
            self.log_mass_stellar_err_minus = float(kwargs["log_mass_stellar_err"])

        self.sfr = None
        if "sfr" in kwargs:
            self.sfr = kwargs["sfr"] * units.solMass

        self.sfr_err = None
        if "sfr_err" in kwargs:
            self.sfr_err = kwargs["sfr_err"] * units.solMass

        self.mass_halo = None
        self.log_mass_halo = None
        self.log_mass_halo_upper = None
        self.log_mass_halo_lower = None
        if "mass_halo" in kwargs:
            self.mass_halo = u.check_quantity(kwargs["mass_halo"], units.solMass)
            self.log_mass_halo = np.log10(self.mass_halo / units.solMass)

        self.halo_mnfw = None
        self.halo_yf17 = None
        self.halo_mb15 = None
        self.halo_mb04 = None

        self.sed_models = {}

        self.cigale_model_path = None
        self.cigale_model = None

        self.cigale_sfh_path = None
        self.cigale_sfh = None

        self.cigale_results_path = None
        self.cigale_results = None

        self.galfit_models = {}

        self.inclination = None
        self.position_angle = None

        self.do_galfit: bool = False
        if "do_galfit" in kwargs:
            self.do_galfit = kwargs["do_galfit"]

    def sed_model_path(self):
        path = os.path.join(self.data_path, "sed_models")
        u.mkdir_check(path)
        return path

    def add_sed_model(
            self,
            path: str,
            name: str,
            model_type: type = sed.SEDModel,
            **kwargs
    ):
        sed_path = os.path.join(self.sed_model_path(), name)
        u.mkdir_check(sed_path)
        self.sed_models[name] = model_type(
            z=self.z,
            path=path,
            output_dir=sed_path,
            name=name,
            **kwargs
        )
        return self.sed_models[name]

    def load_cigale_model(self, force: bool = False):
        # TODO: incorporate into SEDModel
        if self.cigale_model_path is None:
            print(f"Cannot load CIGALE model; {self}.cigale_model_path has not been set.")
        elif force or self.cigale_model is None:
            self.cigale_model = fits.open(self.cigale_model_path)

        if self.cigale_sfh_path is None:
            print(f"Cannot load CIGALE SFH; {self}.cigale_sfh_path has not been set.")
        elif force or self.cigale_sfh is None:
            self.cigale_sfh = fits.open(self.cigale_sfh_path)

        return self.cigale_model, self.cigale_sfh  # , self.cigale_results

    def _output_dict(self):
        output = super()._output_dict()
        output.update({
            "log_mass_stellar": self.log_mass_stellar,
            "log_mass_stellar_err_plus": self.log_mass_stellar_err_plus,
            "log_mass_stellar_err_minus": self.log_mass_stellar_err_minus,
            "sfr": self.sfr,
            "sfr_err": self.sfr_err,
            "cigale_model_path": self.cigale_model_path,
            "cigale_sfh_path": self.cigale_sfh_path,
            "cigale_results": self.cigale_results,
            "galfit_models": self.galfit_models
        })
        return output

    def load_output_file(self):
        outputs = super().load_output_file()
        if outputs is not None:
            # if "log_mass_stellar" in outputs and outputs["log_mass_stellar"] is not None:
            #     self.log_mass_stellar = outputs["log_mass_stellar"]
            # if "log_mass_stellar_err_plus" in outputs and outputs["log_mass_stellar_err_plus"] is not None:
            #     self.log_mass_stellar_err_plus = outputs["log_mass_stellar_err_plus"]
            # if "log_mass_stellar_err_minus" in outputs and outputs["log_mass_stellar_err_minus"] is not None:
            #     self.log_mass_stellar_err_minus = outputs["log_mass_stellar_err_minus"]
            if "sfr" in outputs and outputs["sfr"] is not None:
                self.sfr = outputs["sfr"]
            if "sfr_err" in outputs and outputs["sfr_err"] is not None:
                self.sfr_err = outputs["sfr_err"]
            if "cigale_model_path" in outputs and outputs["cigale_model_path"] is not None:
                self.cigale_model_path = outputs["cigale_model_path"]
            if "cigale_sfh_path" in outputs and outputs["cigale_sfh_path"] is not None:
                self.cigale_sfh_path = outputs["cigale_sfh_path"]
            if "cigale_results" in outputs and outputs["cigale_results"] is not None:
                self.cigale_results = outputs["cigale_results"]
            if "galfit_models" in outputs and outputs["galfit_models"] is not None and outputs["galfit_models"]:
                self.galfit_models = outputs["galfit_models"]
        return outputs

    def h(self):
        return cosmology.H(z=self.z) / (100 * units.km * units.second ** -1 * units.Mpc ** -1)

    def halo_mass(self):
        from frb.halos.utils import halomass_from_stellarmass
        if self.log_mass_stellar is None:
            raise ValueError(f"{self}.log_mass_stellar has not been defined.")
        self.log_mass_halo = halomass_from_stellarmass(
            log_mstar=self.log_mass_stellar,
            z=self.z
        )
        self.mass_halo = (10 ** self.log_mass_halo) * units.solMass
        if self.log_mass_stellar_err_plus is None:
            self.log_mass_stellar_err_plus = 0. * units.solMass
        self.log_mass_halo_upper = halomass_from_stellarmass(
            log_mstar=self.log_mass_stellar + self.log_mass_stellar_err_plus,
            z=self.z
        )
        if self.log_mass_stellar_err_minus is None:
            self.log_mass_stellar_err_minus = 0. * units.solMass
        self.log_mass_halo_lower = halomass_from_stellarmass(
            log_mstar=self.log_mass_stellar - self.log_mass_stellar_err_minus,
            z=self.z
        )

        return self.mass_halo, self.log_mass_halo

    def halo_concentration_parameter(self):
        if self.log_mass_halo is None:
            self.halo_mass()
        c = 4.67 * (self.mass_halo / (10 ** 14 * self.h() ** -1 * units.solMass)) ** (-0.11)
        return float(c)

    def halo_model_mnfw(self, y0=2., alpha=2., **kwargs):
        from frb.halos.models import ModifiedNFW
        if self.log_mass_halo is None:
            self.halo_mass()
        self.halo_mnfw = ModifiedNFW(
            log_Mhalo=self.log_mass_halo,
            z=self.z,
            cosmo=cosmology,
            c=self.halo_concentration_parameter(),
            y0=y0,
            alpha=alpha,
            **kwargs
        )
        self.halo_mnfw.coord = self.position
        return self.halo_mnfw

    def halo_model_yf17(self, **kwargs):
        from frb.halos.models import YF17
        if self.log_mass_halo is None:
            self.halo_mass()
        self.halo_yf17 = YF17(
            log_Mhalo=self.log_mass_halo,
            z=self.z,
            cosmo=cosmology,
            **kwargs
        )
        return self.halo_yf17

    def halo_model_mb04(self, r_c=147 * units.kpc, **kwargs):
        from frb.halos.models import MB04
        if self.log_mass_halo is None:
            self.halo_mass()
        self.halo_mb04 = MB04(
            log_Mhalo=self.log_mass_halo,
            z=self.z,
            cosmo=cosmology,
            c=self.halo_concentration_parameter(),
            Rc=r_c,
            **kwargs
        )
        return self.halo_mb04

    def halo_model_mb15(self, **kwargs):
        from frb.halos.models import MB15
        if self.log_mass_halo is None:
            self.halo_mass()
        self.halo_mb15 = MB15(
            log_Mhalo=self.log_mass_halo,
            z=self.z,
            cosmo=cosmology,
            **kwargs
        )
        return self.halo_mb15

    def halo_dm_cum(
            self,
            rmax: float = 1.,
            rperp: units.Quantity = 0. * units.kpc,
            step_size: units.Quantity = 0.1 * units.kpc
    ):
        from .frb import dm_units
        d, dm = self.halo_mnfw.Ne_Rperp(
            rperp,
            step_size=step_size,
            rmax=rmax,
            cumul=True
        )
        tbl = table.QTable({
            "d": d * units.kpc,
            "d_abs": d * units.kpc + self.comoving_distance(),
            "DM": dm * dm_units / (1 + self.z),
        })
        return tbl

    def galfit_guess_dict(self, img: 'craftutils.observation.image.ImagingImage'):
        img.load_output_file()
        fil = img.filter.name
        instrument = img.instrument.name
        photom, mean = self.select_photometry(
            fil=fil,
            instrument=instrument,
            local_output=False
        )
        if photom["ra"] < 0 * units.deg:
            photom = self.select_deepest()
        if photom["ra"] < 0 * units.deg:
            position = self.position
        else:
            position = SkyCoord(photom["ra"], photom["dec"])
        x, y = img.world_to_pixel(position)
        r_e = img.pixel(photom["a"])

        guesses = {
            "object_type": "sersic",
            "int_mag": photom["mag"].value,
            "position": position,
            "x": float(x),
            "y": float(y),
            "r_e": r_e,
            "position_angle": photom["theta"],
            "axis_ratio": photom["b"] / photom["a"],
        }
        img.extract_n_pix()
        frame_lower = int(img.pixel(photom["a"] * photom["kron_radius"] * 2).value)
        frame_upper = int(min(frame_lower * 3, img.n_x / 2, img.n_y / 2))

        kwargs = {
            "frame_lower": int(frame_lower),
            "frame_upper": int(frame_upper)
        }
        print("Using frames", frame_lower, kwargs["frame_upper"])
        return guesses, kwargs

    def assemble_row(
            self,
            **kwargs
    ):
        row, _ = super().assemble_row(**kwargs)
        if self.z is not None:
            self.set_z()
            row["z"] = self.z
            row["d_A"] = self.D_A
            row["d_L"] = self.D_L
            row["mu"] = self.mu
        if "best" in self.galfit_models and self.galfit_models["best"] is not None and "COMP_2" in self.galfit_models[
            "best"]:
            best = self.galfit_models["best"]
            galfit_model = best["COMP_2"]
            row["galfit_axis_ratio"] = galfit_model["axis_ratio"]
            row["galfit_axis_ratio_err"] = galfit_model["axis_ratio_err"]
            row["galfit_ra"] = galfit_model["ra"]
            row["galfit_ra_err"] = galfit_model["ra_err"].to("arcsec")
            row["galfit_dec"] = galfit_model["dec"]
            row["galfit_dec_err"] = galfit_model["dec_err"].to("arcsec")
            row["galfit_r_eff"] = galfit_model["r_eff_ang"]
            row["galfit_r_eff_err"] = galfit_model["r_eff_ang_err"]
            if self.z is not None and self.z > 0:
                galfit_model["r_eff_proj"] = self.projected_size(galfit_model["r_eff_ang"])
                galfit_model["r_eff_proj_err"] = self.projected_size(galfit_model["r_eff_ang_err"])
            if "r_eff_proj" in galfit_model:
                row["galfit_r_eff_proj"] = galfit_model["r_eff_proj"]
                row["galfit_r_eff_proj_err"] = galfit_model["r_eff_proj_err"]
            row["galfit_n"] = galfit_model["n"]
            row["galfit_n_err"] = galfit_model["n_err"]
            row["galfit_theta"] = galfit_model["position_angle"]
            row["galfit_theta_err"] = galfit_model["position_angle_err"]
            row["galfit_mag"] = galfit_model["mag"]
            row["galfit_mag_err"] = galfit_model["mag_err"]
            row["galfit_img"] = os.path.basename(best["image"]).replace(".fits", "")
            row["galfit_band"] = best["band"]
            row["galfit_instrument"] = best["instrument"]

        return row, "optical"

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        default_params.update({
            "type": "Galaxy"
        })
        return default_params
