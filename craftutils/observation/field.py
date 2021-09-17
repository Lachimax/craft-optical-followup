# Code by Lachlan Marnoch, 2021
import copy
import os
import warnings
from typing import Union, List
import shutil

import matplotlib.pyplot as plt
import numpy as np

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as units
import astropy.table as table
import astropy.wcs as wcs

import craftutils.astrometry as am
import craftutils.fits_files as ff
import craftutils.observation.objects as objects
import craftutils.observation.image as image
import craftutils.params as p
import craftutils.plotting as pl
import craftutils.retrieve as retrieve
import craftutils.spectroscopy as spec
import craftutils.utils as u
import craftutils.wrap.montage as montage
import craftutils.wrap.dragons as dragons

config = p.config

instruments_imaging = p.instruments_imaging
instruments_spectroscopy = p.instruments_spectroscopy
surveys = p.surveys


def _output_img_list(lst: list):
    """
    Turns a list of images into a YAML-able list of paths.
    :param lst:
    :return:
    """
    out_list = []
    for img in lst:
        out_list.append(img.path)
    out_list.sort()
    return out_list


def _output_img_dict_single(dictionary: dict):
    """
    Turns a dict of images into a YAML-able dict of paths.
    :param dictionary:
    :return:
    """
    out_dict = {}
    for fil in dictionary:
        img = dictionary[fil]
        if isinstance(img, image.Image):
            out_dict[fil] = img.path
        elif isinstance(img, str):
            out_dict[fil] = img
    return out_dict


def _output_img_dict_list(dictionary: dict):
    """
    Turns a dict of lists of images into a YAML-able dict of lists of paths.
    :param dictionary:
    :return:
    """
    out_dict = {}
    for fil in dictionary:
        out_dict[fil] = list(map(lambda f: f.path, dictionary[fil]))
        out_dict[fil].sort()
    return out_dict


def select_instrument(mode: str):
    if mode == "imaging":
        options = instruments_imaging
    elif mode == "spectroscopy":
        options = instruments_spectroscopy
    else:
        raise ValueError("Mode must be 'imaging' or 'spectroscopy'.")
    _, instrument = u.select_option("Select an instrument:", options=options)
    return instrument


def list_fields():
    print("Searching for field param files...")
    param_path = os.path.join(config['param_dir'], 'fields')
    fields = list(filter(lambda d: os.path.isdir(os.path.join(param_path, d)) and os.path.isfile(
        os.path.join(param_path, d, f"{d}.yaml")), os.listdir(param_path)))
    fields.sort()
    return fields


def list_fields_old():
    print("Searching for old-format field param files...")
    param_path = os.path.join(config['param_dir'], 'FRBs')
    fields = filter(lambda d: os.path.isfile(os.path.join(param_path, d)) and d.endswith('.yaml'),
                    os.listdir(param_path))
    fields = list(map(lambda f: f.split(".")[0], fields))
    fields.sort()
    return fields


def _retrieve_eso_epoch(epoch: Union['ESOImagingEpoch', 'ESOSpectroscopyEpoch'], path: str):
    if epoch.program_id is None:
        epoch.set_program_id(input("A program ID is required to retrieve ESO data. Enter here:\n"))
    if epoch.date is None:
        epoch.set_date(Time(
            input("An observation date is required to retrieve ESO data. Enter here, in iso or isot format:\n")))
    if isinstance(epoch, ESOImagingEpoch):
        mode = "imaging"
    elif isinstance(epoch, ESOSpectroscopyEpoch):
        mode = "spectroscopy"
    else:
        raise TypeError("epoch must be either an ESOImagingEpoch or an ESOSpectroscopyEpoch.")
    if epoch.target is None:
        obj = u.user_input(
            "Specifying an object might help find the correct observation. Enter here, as it appears in the archive "
            "OBJECT field, or leave blank:\n")
        if obj not in ["", " "]:
            epoch.set_target(obj)

    u.mkdir_check(path)
    instrument = epoch.instrument.split('-')[-1]
    r = retrieve.save_eso_raw_data_and_calibs(output=path, date_obs=epoch.date,
                                              program_id=epoch.program_id, instrument=instrument, mode=mode,
                                              obj=epoch.target)
    if r:
        os.system(f"uncompress {path}/*.Z -f")
    return r


def _check_do_list(do: Union[list, str]):
    if type(do) is str:
        try:
            do = [int(do)]
        except ValueError:
            if " " in do:
                char = " "
            elif "," in do:
                char = ","
            else:
                raise ValueError("do string is not correctly formatted.")
            do = list(map(int, do.split(char)))
    return do


class Field:
    def __init__(self,
                 name: str = None,
                 centre_coords: Union[SkyCoord, str] = None,
                 param_path: str = None,
                 data_path: str = None,
                 objs: Union[List[objects.Object], dict] = None,
                 extent: units.Quantity = None
                 ):
        """

        :param centre_coords:
        :param name:
        :param param_path:
        :param data_path:
        :param objs: a list of objects of interest in the field. The primary object of interest should be first in
        the list.
        """

        # Input attributes

        self.objects = []

        if type(objs) is dict:
            for obj_name in objs:
                if obj_name != "<name>":
                    obj_dict = objs[obj_name]
                    if "position" not in obj_dict:
                        obj_dict = {"position": obj_dict}
                    if "name" not in obj_dict:
                        obj_dict["name"] = obj_name
                    obj = objects.Object.from_dict(obj_dict)
                    self.add_object(obj)

        if centre_coords is None:
            if objs is not None:
                centre_coords = objs[0].coords
        if centre_coords is not None:
            self.centre_coords = am.attempt_skycoord(centre_coords)

        self.name = name
        self.param_path = param_path
        self.param_dir = None
        if self.param_path is not None:
            self.param_dir = os.path.split(self.param_path)[0]
        self.mkdir_params()
        self.data_path = data_path
        self.mkdir()
        self.output_file = None

        # Derived attributes

        self.epochs_spectroscopy = {}
        self.epochs_spectroscopy_loaded = {}
        self.epochs_imaging = {}
        self.epochs_imaging_loaded = {}

        self.paths = {}

        self.cats = {}
        self.cat_gaia = None

        self.load_output_file()

        self.extent = extent

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return self.__str__()

    def mkdir(self):
        if self.data_path is not None:
            u.mkdir_check(self.data_path)
            u.mkdir_check(os.path.join(self.data_path, "objects"))

    def mkdir_params(self):
        if self.param_dir is not None:
            u.mkdir_check(os.path.join(self.param_dir, "spectroscopy"))
            u.mkdir_check(os.path.join(self.param_dir, "imaging"))
            u.mkdir_check(os.path.join(self.param_dir, "objects"))
        else:
            warnings.warn(f"param_dir is not set for this {type(self)}.")

    def _gather_epochs(self, mode: str = "imaging"):
        print(f"Searching for {mode} epoch param files...")
        epochs = {}
        if self.param_dir is not None:
            mode_path = os.path.join(self.param_dir, mode)
            print(f"Looking in {mode_path}")
            for instrument in filter(lambda d: os.path.isdir(os.path.join(mode_path, d)), os.listdir(mode_path)):
                instrument_path = os.path.join(mode_path, instrument)
                print(f"Looking in {instrument_path}")
                for epoch_param in filter(lambda f: f.endswith(".yaml"), os.listdir(instrument_path)):
                    epoch_name = epoch_param[:epoch_param.find(".yaml")]
                    param_path = os.path.join(instrument_path, epoch_param)
                    epoch = p.load_params(file=param_path)
                    epoch["format"] = "current"
                    epoch["param_path"] = param_path
                    epochs[epoch_name] = epoch

        return epochs

    def gather_epochs_spectroscopy(self):
        epochs = self._gather_epochs(mode="spectroscopy")
        self.epochs_spectroscopy.update(epochs)
        return epochs

    def gather_epochs_imaging(self):
        epochs = self._gather_epochs(mode="imaging")
        self.epochs_imaging.update(epochs)
        return epochs

    def epoch_from_params(self, epoch_name: str, instrument: str, old_format: bool = False):
        epoch = ImagingEpoch.from_params(name=epoch_name, field=self, instrument=instrument, old_format=old_format)
        self.epochs_imaging[epoch_name] = epoch
        return epoch

    def select_epoch_imaging(self):
        options = {}
        for epoch in self.epochs_imaging:
            epoch = self.epochs_imaging[epoch]
            date_string = ""
            if epoch["format"] == 'old':
                date_string = " (old format)           "
            elif "date" in epoch and epoch["date"] is not None:
                if isinstance(epoch["date"], str):
                    date_string = f" {epoch['date']}"
                else:
                    date_string = f" {epoch['date'].to_datetime().date()}"
            options[f'{epoch["name"]}\t{date_string}\t{epoch["instrument"]}'] = epoch
        for epoch in self.epochs_imaging_loaded:
            # If epoch is already instantiated.
            epoch = self.epochs_spectroscopy_loaded[epoch]
            options[f'*{epoch.name}\t{epoch.date.isot}\t{epoch.instrument}'] = epoch
        options["New epoch"] = "new"
        j, epoch = u.select_option(message="Select epoch.", options=options, sort=True)
        if epoch == "new":
            epoch = self.new_epoch_imaging()
        elif not isinstance(epoch, Epoch):
            old_format = False
            if epoch["format"] == "old":
                old_format = True
            epoch = ImagingEpoch.from_file(epoch, old_format=old_format, field=self)
            self.epochs_imaging_loaded[epoch.name] = epoch
        return epoch

    def select_epoch_spectroscopy(self):
        options = {}
        for epoch in self.epochs_spectroscopy:
            epoch = self.epochs_spectroscopy[epoch]
            options[f"{epoch['name']}\t{epoch['date'].to_datetime().date()}\t{epoch['instrument']}"] = epoch
        for epoch in self.epochs_spectroscopy_loaded:
            epoch = self.epochs_spectroscopy_loaded[epoch]
            options[f'*{epoch.name}\t{epoch.date.isot}\t{epoch.instrument}'] = epoch
        options["New epoch"] = "new"
        j, epoch = u.select_option(message="Select epoch.", options=options)
        if epoch == "new":
            epoch = self.new_epoch_spectroscopy()
        elif not isinstance(epoch, Epoch):
            epoch = SpectroscopyEpoch.from_file(epoch, field=self)
            self.epochs_spectroscopy_loaded[epoch.name] = epoch
        return epoch

    def new_epoch_imaging(self):
        return self._new_epoch(mode="imaging")

    def new_epoch_spectroscopy(self):
        return self._new_epoch(mode="spectroscopy")

    def _new_epoch(self, mode: str):
        """
        Helper method for generating a new epoch.
        :param mode:
        :return:
        """
        instrument = select_instrument(mode=mode)
        if mode == "imaging":
            cls = ImagingEpoch.select_child_class(instrument=instrument)
        elif mode == "spectroscopy":
            cls = SpectroscopyEpoch.select_child_class(instrument=instrument)
        else:
            raise ValueError("mode must be 'imaging' or 'spectroscopy'.")
        new_params = cls.default_params()
        if instrument in surveys:
            new_params["name"] = instrument.upper()
            new_params["date"] = None
            new_params["program_id"] = None
            survey = True
        else:
            survey = False
            new_params["name"] = u.user_input("Please enter a name for the epoch.")
            new_params["date"] = u.enter_time(message="Enter UTC observation date, in iso or isot format:")
            new_params["program_id"] = input("Enter the programmme ID for the observation:\n")
        new_params["instrument"] = instrument
        new_params["data_path"] = self._epoch_data_path(mode=mode, instrument=instrument, date=new_params["date"],
                                                        epoch_name=new_params["name"], survey=survey)
        new_params["field"] = self.name
        param_path = self._epoch_param_path(mode=mode, instrument=instrument, epoch_name=new_params["name"])

        p.save_params(file=param_path, dictionary=new_params)
        epoch = cls.from_file(param_file=param_path, field=self)

        return epoch

    def _mode_param_path(self, mode: str):
        if self.param_dir is not None:
            path = os.path.join(self.param_dir, mode)
            u.mkdir_check(path)
            return path
        else:
            raise ValueError(f"param_dir is not set for {self}.")

    def _mode_data_path(self, mode: str):
        if self.data_path is not None:
            path = os.path.join(self.data_path, mode)
            u.mkdir_check(path)
            return path
        else:
            raise ValueError(f"data_path is not set for {self}.")

    def _cat_data_path(self, cat: str):
        if self.data_path is not None:
            filename = f"{cat}_{self.name}.csv"
            path = os.path.join(self.data_path, filename)
            return path
        else:
            raise ValueError(f"data_path is not set for {self}.")

    def _instrument_param_path(self, mode: str, instrument: str):
        path = os.path.join(self._mode_param_path(mode=mode), instrument)
        u.mkdir_check(path)
        return path

    def _instrument_data_path(self, mode: str, instrument: str):
        path = os.path.join(self._mode_data_path(mode=mode), instrument)
        u.mkdir_check(path)
        return path

    def _epoch_param_path(self, mode: str, instrument: str, epoch_name: str):
        return os.path.join(self._instrument_param_path(mode=mode, instrument=instrument), f"{epoch_name}.yaml")

    def _epoch_data_path(self, mode: str, instrument: str, date: Time, epoch_name: str, survey: bool = False):
        if survey:
            path = self._instrument_data_path(mode=mode, instrument=instrument)
        else:
            path = os.path.join(self._instrument_data_path(mode=mode, instrument=instrument),
                                f"{date.to_datetime().date()}-{epoch_name}")
        u.mkdir_check(path)
        return path

    def retrieve_catalogues(self, force_update: bool = False):
        for cat_name in retrieve.photometry_catalogues:
            print(f"Checking for photometry in {cat_name}")
            self.retrieve_catalogue(cat_name=cat_name, force_update=force_update)

    def retrieve_catalogue(self, cat_name: str, force_update: bool = False):
        if isinstance(self.extent, units.Quantity):
            radius = self.extent
        else:
            radius = 0.2 * units.deg
        output = self._cat_data_path(cat=cat_name)
        ra = self.centre_coords.ra.value
        dec = self.centre_coords.dec.value
        if force_update or f"in_{cat_name}" not in self.cats:
            response = retrieve.save_catalogue(ra=ra, dec=dec, output=output, cat=cat_name.lower(),
                                               radius=radius)
            # Check if a valid response was received; if not, we don't want to erroneously report that
            # the field doesn't exist in the catalogue.
            if isinstance(response, str) and response == "ERROR":
                pass
            else:
                if response is not None:
                    self.cats[f"in_{cat_name}"] = True
                    self.set_path(f"cat_csv_{cat_name}", output)
                else:
                    self.cats[f"in_{cat_name}"] = False
                self.update_output_file()
            return response
        elif self.cats[f"in_{cat_name}"] is True:
            print(f"There is already {cat_name} data present for this field.")
            return True
        else:
            print(f"This field is not present in {cat_name}.")

    def load_catalogue(self, cat_name: str):
        if self.retrieve_catalogue(cat_name):
            return retrieve.load_catalogue(cat_name=cat_name, cat=self.get_path(f"cat_csv_{cat_name}"))
        else:
            print("Could not load catalogue; field is outside footprint.")

    def get_photometry(self):
        for obj in self.objects:
            pass

    def generate_astrometry_indices(self, cat_name: str = "gaia"):
        self.retrieve_catalogue(cat_name=cat_name)
        if not self.check_cat(cat_name=cat_name):
            print(f"Field is not in {cat_name}; index file could not be created.")
        else:
            cat_path = self.get_path(f"cat_csv_{cat_name}")
            index_path = os.path.join(config["top_data_dir"], "astrometry_index_files")
            u.mkdir_check(index_path)
            cat_index_path = os.path.join(index_path, cat_name)
            prefix = f"{cat_name}_index_{self.name}"
            am.generate_astrometry_indices(cat_name=cat_name,
                                           cat=cat_path,
                                           fits_cat_output=cat_path.replace(".csv", ".fits"),
                                           output_file_prefix=prefix,
                                           index_output_dir=cat_index_path
                                           )

    def get_path(self, key):
        if key in self.paths:
            return self.paths[key]
        else:
            raise KeyError(f"{key} has not been set.")

    def check_cat(self, cat_name: str):
        if f"in_{cat_name}" in self.cats:
            return self.cats[f"in_{cat_name}"]
        else:
            return None

    def set_path(self, key, value):
        self.paths[key] = value

    def load_output_file(self, **kwargs):
        outputs = p.load_output_file(self)
        if outputs is not None:
            if "cats" in outputs:
                self.cats.update(outputs["cats"])
        return outputs

    def _output_dict(self):
        return {
            "paths": self.paths,
            "cats": self.cats,
        }

    def update_output_file(self):
        p.update_output_file(self)

    def add_object(self, obj: objects.Object):
        self.objects.append(obj)
        obj.field = self

    @classmethod
    def default_params(cls):
        default_params = {"name": None,
                          "type": "Field",
                          "centre": objects.position_dictionary.copy(),
                          "objects": [objects.Object.default_params()],
                          "extent": 0.2 * units.deg
                          }
        return default_params

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):
        name, param_file, param_dict = p.params_init(param_file)
        if param_file is None:
            return None
        # Check data_dir path for relevant .yamls (output_values, etc.)

        field_type = param_dict["type"]
        centre_ra, centre_dec = p.select_coords(param_dict["centre"])

        if "extent" in param_dict:
            extent = param_dict["extent"]
        else:
            extent = None

        if field_type == "Field":
            return cls(name=name,
                       centre_coords=f"{centre_ra} {centre_dec}",
                       param_path=param_file,
                       data_path=os.path.join(config["top_data_dir"], param_dict["data_path"]),
                       objs=param_dict["objects"],
                       extent=extent
                       )
        elif field_type == "FRBField":
            return FRBField.from_file(param_file)

    @classmethod
    def from_params(cls, name):
        path = cls.build_param_path(field_name=name)
        return cls.from_file(param_file=path)

    @classmethod
    def new_yaml(cls, name: str, path: str = None, quiet: bool = False):
        param_dict = cls.default_params()
        param_dict["name"] = name
        param_dict["data_path"] = os.path.join(p.data_path, name, "")
        if path is not None:
            path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict

    @classmethod
    def build_param_path(cls, field_name: str):
        return os.path.join(p.param_dir, "fields", field_name, field_name)


class StandardField(Field):
    pass


class FRBField(Field):
    def __init__(self,
                 name: str = None,
                 centre_coords: Union[SkyCoord, str] = None,
                 param_path: str = None,
                 data_path: str = None,
                 objs: List[objects.Object] = None,
                 frb: objects.FRB = None,
                 extent: units.Quantity = None):
        if centre_coords is None:
            if frb is not None:
                centre_coords = frb.position

        # Input attributes
        super().__init__(name=name,
                         centre_coords=centre_coords,
                         param_path=param_path,
                         data_path=data_path,
                         objs=objs,
                         extent=extent
                         )

        self.frb = frb
        if self.frb is not None:
            self.frb.field = self
            if self.frb.host_galaxy is not None:
                self.add_object(self.frb.host_galaxy)
        self.epochs_imaging_old = {}

    def plot_host(self, epoch: 'ImagingEpoch', fig: plt.Figure, fil: str,
                  centre: SkyCoord = None,
                  show_frb: bool = True, frame: units.Quantity = 30 * units.pix,
                  n: int = 1, n_x: int = 1, n_y: int = 1,
                  cmap: str = 'viridis', show_cbar: bool = False,
                  stretch: str = 'sqrt',
                  vmin: float = None,
                  vmax: float = None,
                  show_grid: bool = False,
                  ticks: int = None, interval: str = 'minmax',
                  show_coords: bool = True,
                  font_size: int = 12,
                  reverse_y=False,
                  frb_kwargs: dict = {},
                  **kwargs):

        if not isinstance(self.frb, objects.FRB):
            raise TypeError("self.frb has not been set properly for this FRBField.")
        if centre is None:
            centre = self.frb.host_galaxy.position

        subplot, hdu_cut = epoch.plot_object(fig=fig, frame=frame, fil=fil, img="coadded",
                                             centre=centre,
                                             n=n, n_x=n_x, n_y=n_y,
                                             cmap=cmap, show_cbar=show_cbar, stretch=stretch,
                                             vmin=vmin, vmax=vmax,
                                             show_grid=show_grid,
                                             ticks=ticks, interval=interval,
                                             show_coords=show_coords,
                                             font_size=font_size,
                                             reverse_y=reverse_y,
                                             **kwargs
                                             )

        position = self.frb.position
        ra = position.ra.value
        dec = position.dec.value
        uncertainty = self.frb.position_err

        if show_frb:
            show_frb = 'quadrature'

        theta = uncertainty.theta.to(units.deg)

        if show_frb in ['all', 'statistical']:
            # Statistical
            pl.plot_gal_params(hdu=hdu_cut,
                               ras=[ra],
                               decs=[dec],
                               a=[uncertainty.a_stat.to(units.deg).value],
                               b=[uncertainty.b_stat.to(units.deg).value],
                               theta=[-theta.value], ls='.', **frb_kwargs)
        if show_frb in ['all', 'statistical']:
            # Systematic
            pl.plot_gal_params(hdu=hdu_cut,
                               ras=[ra],
                               decs=[dec],
                               a=[uncertainty.a_sys.to(units.deg).value],
                               b=[uncertainty.b_sys.to(units.deg).value],
                               theta=[-theta.value],
                               ls='--', **frb_kwargs)
        if show_frb in ['all', 'quadrature']:
            a_quad = np.sqrt(uncertainty.a_stat ** 2 + uncertainty.a_sys ** 2).to(units.deg).value
            b_quad = np.sqrt(uncertainty.b_stat ** 2 + uncertainty.b_sys ** 2).to(units.deg).value
            pl.plot_gal_params(hdu=hdu_cut,
                               ras=[ra],
                               decs=[dec],
                               a=[a_quad],
                               b=[b_quad],
                               theta=[-theta.value],
                               ls='-', **frb_kwargs,
                               label="FRB uncertainty")

        return subplot, hdu_cut

    @classmethod
    def default_params(cls):
        default_params = super().default_params()

        default_params.update({
            "type": "FRBField",
            "frb": objects.FRB.default_params(),
            "subtraction":
                {"template_epochs":
                     {"des": None,
                      "fors2": None,
                      "xshooter": None,
                      "sdss": None
                      }
                 }
        })

        return default_params

    @classmethod
    def new_yaml(cls, name: str, path: str = None, quiet: bool = False):
        param_dict = super().new_yaml(name=name, path=None)
        param_dict["frb"]["name"] = name
        param_dict["frb"]["type"] = "FRB"
        param_dict["frb"]["host_galaxy"]["name"] = name.replace("FRB", "HG")
        if path is not None:
            path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict

    @classmethod
    def from_file(cls, param_file: Union[str, dict]):
        name, param_file, param_dict = p.params_init(param_file)

        # Check data_dir path for relevant .yamls (output_values, etc.)

        centre_ra, centre_dec = p.select_coords(param_dict["centre"])
        frb = objects.FRB.from_dict(param_dict["frb"])
        if "extent" in param_dict:
            extent = param_dict["extent"]
        else:
            extent = None
        return cls(name=name,
                   centre_coords=f"{centre_ra} {centre_dec}",
                   param_path=param_file,
                   data_path=os.path.join(config["top_data_dir"], param_dict["data_path"]),
                   objs=param_dict["objects"],
                   frb=frb,
                   extent=extent
                   )

    @classmethod
    def convert_old_param(cls, frb: str):

        new_frb = f"FRB20{frb[3:]}"
        new_params = cls.new_yaml(name=new_frb, path=None)
        old_params = p.object_params_frb(frb)

        new_params["name"] = new_frb

        new_params["centre"]["dec"]["decimal"] = old_params["burst_dec"]
        new_params["centre"]["dec"]["dms"] = old_params["burst_dec_str"]

        new_params["centre"]["ra"]["decimal"] = old_params["burst_ra"]
        new_params["centre"]["ra"]["hms"] = old_params["burst_ra_str"]

        old_data_dir = old_params["data_dir"]
        if isinstance(old_data_dir, str):
            new_params["data_path"] = old_data_dir.replace(frb, new_frb)

        new_params["frb"]["name"] = new_frb

        new_params["frb"]["host_galaxy"]["position"]["dec"]["decimal"] = old_params["hg_dec"]
        new_params["frb"]["host_galaxy"]["position"]["ra"]["decimal"] = old_params["hg_ra"]

        new_params["frb"]["host_galaxy"]["position_err"]["dec"]["stat"] = old_params["hg_err_y"]
        new_params["frb"]["host_galaxy"]["position_err"]["ra"]["stat"] = old_params["hg_err_x"]

        new_params["frb"]["host_galaxy"]["z"] = old_params["z"]

        new_params["frb"]["mjd"] = old_params["mjd_burst"]

        new_params["frb"]["position"]["dec"]["decimal"] = old_params["burst_dec"]
        new_params["frb"]["position"]["dec"]["dms"] = old_params["burst_dec_str"]

        new_params["frb"]["position"]["ra"]["decimal"] = old_params["burst_ra"]
        new_params["frb"]["position"]["ra"]["hms"] = old_params["burst_ra_str"]

        new_params["frb"]["position_err"]["a"]["stat"] = old_params["burst_err_stat_a"]
        new_params["frb"]["position_err"]["a"]["sys"] = old_params["burst_err_sys_a"]

        new_params["frb"]["position_err"]["b"]["stat"] = old_params["burst_err_stat_b"]
        new_params["frb"]["position_err"]["b"]["sys"] = old_params["burst_err_sys_b"]

        new_params["frb"]["position_err"]["dec"]["stat"] = old_params["burst_err_stat_dec"]
        new_params["frb"]["position_err"]["dec"]["sys"] = old_params["burst_err_sys_dec"]

        new_params["frb"]["position_err"]["ra"]["stat"] = old_params["burst_err_stat_ra"]
        new_params["frb"]["position_err"]["ra"]["sys"] = old_params["burst_err_sys_ra"]

        new_params["frb"]["position_err"]["theta"] = old_params["burst_err_theta"]

        if "other_objects" in old_params and type(old_params["other_objects"]) is dict:
            for obj in old_params["other_objects"]:
                if obj != "<name>":
                    obj_dict = objects.Object.default_params()
                    obj_dict["name"] = obj
                    obj_dict["position"] = objects.position_dictionary.copy()
                    obj_dict["position"]["dec"]["decimal"] = old_params["other_objects"][obj]["dec"]
                    obj_dict["position"]["ra"]["decimal"] = old_params["other_objects"][obj]["ra"]
                    new_params["objects"].append(obj_dict)

        new_params["subtraction"]["template_epochs"]["des"] = old_params["template_epoch_des"]
        new_params["subtraction"]["template_epochs"]["fors2"] = old_params["template_epoch_fors2"]
        new_params["subtraction"]["template_epochs"]["sdss"] = old_params["template_epoch_sdss"]
        new_params["subtraction"]["template_epochs"]["xshooter"] = old_params["template_epoch_xshooter"]

        param_path_upper = os.path.join(p.param_dir, "fields", new_frb)
        u.mkdir_check(param_path_upper)
        p.save_params(file=os.path.join(param_path_upper, f"{new_frb}.yaml"), dictionary=new_params,
                      quiet=False)

    def gather_epochs_old(self):
        print("Searching for old-format imaging epoch param files...")
        epochs = {}
        param_dir = p.param_dir
        for instrument_path in filter(lambda d: d.startswith("epochs_"), os.listdir(param_dir)):
            instrument = instrument_path.split("_")[-1]
            instrument_path = os.path.join(param_dir, instrument_path)
            gathered = list(filter(
                lambda f: (f.startswith(self.name) or f.startswith(f"FRB{self.name[5:]}")) and f.endswith(".yaml"),
                os.listdir(instrument_path)))
            gathered.sort()
            for epoch_param in gathered:
                epoch_name = epoch_param[:epoch_param.find('.yaml')]
                if f"{self.name}_{epoch_name[-1]}" not in self.epochs_imaging:
                    param_path = os.path.join(instrument_path, epoch_param)
                    epoch = p.load_params(file=param_path)
                    epoch["format"] = "old"
                    epoch["name"] = epoch_name
                    epoch["instrument"] = instrument
                    epoch["param_path"] = param_path
                    epochs[epoch_name] = epoch
        self.epochs_imaging.update(epochs)
        return epochs


epoch_stage_dirs = {"0-download": "0-data_with_raw_calibs",
                    "2-pypeit": "2-pypeit",
                    }


class Epoch:
    def __init__(self,
                 param_path: str = None,
                 name: str = None,
                 field: Union[str, Field] = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 program_id: str = None,
                 target: str = None,
                 do: Union[list, str] = None
                 ):

        # Input attributes
        self.param_path = param_path
        self.name = name
        self.field = field
        self.data_path = data_path
        if data_path is not None:
            u.mkdir_check_nested(data_path)
        self.instrument = instrument
        self.date = date
        if type(self.date) is str:
            self.date = Time(date, out_subfmt="date")
        self.program_id = program_id
        self.target = target

        self.do = do

        # Written attributes
        self.output_file = None  # This will be set during the load_output_file call
        self.stages_complete = self.stages()
        self.log = {}

        self.binning = None
        self.binning_std = None

        # Data reduction paths
        self.paths = {}
        self._path_0_raw()

        # Frames
        self.frames_raw = []
        self.frames_bias = []
        self.frames_standard = []
        self.frames_science = []
        self.frames_dark = []
        self.frames_flat = []

        self.frames_reduced = []

        self.coadded = {}

        self.load_output_file()

    def pipeline(self, **kwargs):
        self._pipeline_init()

    def _pipeline_init(self, ):
        if self.data_path is not None:
            u.mkdir_check(self.data_path)
        else:
            raise ValueError(f"data_path has not been set for {self}")
        self.do = _check_do_list(self.do)

    def proc_1_initial_setup(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do initial setup of files?", stage='1-initial_setup'):
            self._initial_setup()
            self.stages_complete['1-initial_setup'] = Time.now()
            self.update_output_file()
            return True

    def _initial_setup(self):
        pass

    def _path_0_raw(self):
        if self.data_path is not None and "raw_dir" not in self.paths:
            self.paths["raw_dir"] = os.path.join(self.data_path, epoch_stage_dirs["0-download"])

    def load_output_file(self, **kwargs):
        outputs = p.load_output_file(self)
        if type(outputs) is dict:
            self.stages_complete.update(outputs["stages"])
            if "coadded" in outputs:
                for fil in outputs["coadded"]:
                    if outputs["coadded"][fil] is not None:
                        self.add_coadded_image(img=outputs["coadded"][fil], key=fil, **kwargs)
        return outputs

    def _output_dict(self):

        return {
            "stages": self.stages_complete,
            "paths": self.paths,
            "frames_science": _output_img_dict_list(self.frames_science),
            "frames_std": _output_img_dict_list(self.frames_standard),
            "coadded": _output_img_dict_list(self.coadded)
        }

    def update_output_file(self):
        p.update_output_file(self)

    def check_done(self, stage: str):
        if stage not in self.stages_complete:
            raise ValueError(f"{stage} is not a valid stage for this Epoch.")
        return self.stages_complete[stage]

    def query_stage(self, message: str, stage: str):
        n = float(stage[:stage.find("-")])
        # Check if n is an integer, and if so cast to int.
        if n == int(n):
            n = int(n)
        if self.do is not None:
            if n in self.do:
                return True
        else:
            message = f"{n}. {message}"
            done = self.check_done(stage=stage)
            if done is not None:
                time_since = (Time.now() - done).sec * units.second
                time_since = u.relevant_timescale(time_since)
                message += f" (last performed at {done.isot}, {time_since.round(1)} ago)"
            options = ["No", "Yes", "Exit"]
            opt, _ = u.select_option(message=message, options=options)
            if opt == 0:
                return False
            if opt == 1:
                return True
            if opt == 2:
                exit(0)

    def set_program_id(self, program_id: str):
        self.program_id = program_id
        self.update_param_file("program_id")

    def set_date(self, date: Union[str, Time]):
        self.date = date
        self.update_param_file("date")

    def set_target(self, target: str):
        self.target = target
        self.update_param_file("target")

    def get_binning(self):
        return self.binning

    def set_binning(self, binning: str):
        self.binning = binning
        return binning

    def get_binning_std(self):
        return self.binning_std

    def set_binning_std(self, binning: str):
        self.binning_std = binning
        return binning

    def get_path(self, key):
        if key in self.paths:
            return self.paths[key]
        else:
            raise KeyError(f"{key} has not been set.")

    def set_path(self, key, value):
        self.paths[key] = value

    def update_param_file(self, param: str):
        p_dict = {"program_id": self.program_id,
                  "date": self.date,
                  "target": self.target}
        if param not in p_dict:
            raise ValueError(f"Either {param} is not a valid parameter, or it has not been configured.")
        if self.param_path is None:
            raise ValueError("param_path has not been set.")
        else:
            params = p.load_params(self.param_path)
        params[param] = p_dict[param]
        p.save_params(file=self.param_path, dictionary=params)

    def add_frame_raw(self, raw_frame: Union[image.ImagingImage, str]):
        self.frames_raw.append(raw_frame)
        self.sort_frame(raw_frame)

    def add_frame_reduced(self, reduced_frame: image.Image):
        if reduced_frame not in self.frames_reduced:
            self.frames_reduced.append(reduced_frame)

    def add_coadded_image(self, img: Union[str, image.Image], key: str, **kwargs):
        if isinstance(img, str):
            img = image.CoaddedImage(path=img)
        img.epoch = self
        self.coadded[key] = img
        return img

    def sort_frame(self, frame: image.Image, sort_key: str = None):
        if frame.frame_type == "bias" and frame not in self.frames_bias:
            self.frames_bias.append(frame)
        elif frame.frame_type == "science" and frame not in self.frames_science:
            if isinstance(self.frames_science, list):
                self.frames_science.append(frame)
            elif isinstance(self.frames_science, dict):
                self.frames_science[sort_key].append(frame)
        elif frame.frame_type == "standard" and frame not in self.frames_standard:
            self.frames_standard.append(frame)
        elif frame.frame_type == "dark" and frame not in self.frames_dark:
            self.frames_dark.append(frame)
        elif frame.frame_type == "flat" and frame not in self.frames_flat:
            self.frames_flat.append(frame)

    @classmethod
    def stages(cls):
        return {"1-initial_setup": None}

    @classmethod
    def default_params(cls):
        default_params = {
            "name": None,
            "field": None,
            "data_path": None,
            "instrument": None,
            "date": None,
            "target": None,
            "program_id": None
        }
        return default_params

    @classmethod
    def new_yaml(cls, name: str, path: str = None, quiet: bool = False):
        param_dict = cls.default_params()
        param_dict["name"] = name
        if path is not None:
            path = os.path.join(path, name)
            p.save_params(file=path, dictionary=param_dict, quiet=quiet)
        return param_dict

    @classmethod
    def _from_params_setup(cls, name: str, field: Union[Field, str] = None):
        field_name = None
        if isinstance(field, Field):
            field_name = field.name
        elif isinstance(field, str):
            field_name = field
            field = None
        elif field is not None:
            raise TypeError(f"field must be str or Field, not {type(field)}")
        if field_name is None:
            field_name = name.split("_")[0]
        return field_name, field


class ImagingEpoch(Epoch):
    def __init__(self,
                 name: str = None,
                 field: Union[str, Field] = None,
                 param_path: str = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 program_id: str = None,
                 target: str = None,
                 source_extractor_config: dict = None,
                 standard_epochs: list = None):
        super().__init__(name=name, field=field, param_path=param_path, data_path=data_path, instrument=instrument,
                         date=date, program_id=program_id, target=target)
        self.guess_data_path()
        self.source_extractor_config = source_extractor_config
        if self.source_extractor_config is None:
            self.source_extractor_config = {}

        self.filters = []
        self.deepest = None
        self.deepest_filter = None

        self.exp_time_mean = {}
        self.exp_time_err = {}
        self.airmass_mean = {}
        self.airmass_err = {}

        self.flats = {}

        self.frames_science = {}
        self.frames_reduced = {}
        self.frames_astrometry = {}

        self.std_pointings = {}
        self.std_objects = {}

        self.coadded_trimmed = {}

        # self.load_output_file(mode="imaging")

    # TODO: Make output_path keyword standard across all proc methods

    def proc_5_correct_astrometry_frames(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage(stage="5-correct_astrometry_frames",
                                        message="Correct astrometry of individual frames?"):
            self.generate_gaia_astrometry_indices()

            if "output_path" in kwargs:
                astrometry_path = kwargs["output_path"]
                kwargs.pop("output_path")
            else:
                astrometry_path = os.path.join(self.data_path, "5-astrometry_frames")
            u.mkdir_check(astrometry_path)

            self.correct_astrometry_frames(output_path=astrometry_path, **kwargs)

            self.paths['astrometry_dir'] = astrometry_path
            self.stages_complete["5-correct_astrometry_frames"] = Time.now()
            self.update_output_file()

    def proc_6_coadd(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage(stage="6-coadd",
                                        message="Coadd astrometry-corrected frames with Montage?"):
            if "output_path" in kwargs:
                coadd_path = kwargs["output_path"]
            else:
                coadd_path = os.path.join(self.data_path, "6-coadded_with_montage")
            self.coadd(coadd_path)
            self.paths['coadd_dir'] = coadd_path
            self.stages_complete["6-coadd"] = Time.now()
            self.update_output_file()

    def proc_7_trim_coadded(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage(stage="7-trim_coadded",
                                        message="Trim / reproject coadded images to same footprint?"):
            trimmed_path = os.path.join(self.data_path, "7-trimmed_again")
            self.trim_coadded(trimmed_path)
            self.paths['trimmed_coadded_dir'] = trimmed_path
            self.stages_complete["7-trim_coadded"] = Time.now()
            self.update_output_file()

    def proc_8_source_extraction(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do source extraction?", stage='8-source_extraction'):
            source_extraction_path = os.path.join(self.data_path, "8-source_extraction")
            u.mkdir_check(source_extraction_path)
            for fil in self.coadded_trimmed:
                img = self.coadded_trimmed[fil]
                self.set_path("source_extraction_dir", source_extraction_path)
                configs = self.source_extractor_config
                img.source_extraction_psf(output_dir=source_extraction_path,
                                          phot_autoparams=f"{configs['kron_factor']},{configs['kron_radius_min']}")
            self.stages_complete['8-source_extraction'] = Time.now()
            self.update_output_file()

    def proc_9_photometric_calibration(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do photometric calibration?", stage="9-photometric_calibration"):
            calib_dir = os.path.join(self.data_path, "9-photometric_calibration")
            self.photometric_calibration(calib_dir)
            self.paths['calib_dir'] = calib_dir
            self.stages_complete["9-photometric_calibration"] = Time.now()
            self.update_output_file()

    def proc_10_dual_mode_source_extraction(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do source extraction in dual-mode, using deepest image as footprint?",
                                        stage="10-dual_mode_source_extraction"):
            source_extraction_path = os.path.join(self.data_path, "10-dual_mode_source_extraction")
            self.dual_mode_source_extraction(source_extraction_path)
            self.stages_complete["10-dual_mode_source_extraction"] = Time.now()
            self.update_output_file()

    def proc_11_get_photometry(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Get photometry?", stage="11-get_photometry"):
            object_property_path = os.path.join(self.data_path, "11-object_properties")
            self.get_photometry(object_property_path)
            self.stages_complete["11-object_properties"] = Time.now()
            self.update_output_file()

    def correct_astrometry_frames(self, output_path: str):
        self.frames_astrometry = {}
        for fil in self.frames_reduced:
            astrometry_fil_path = os.path.join(output_path, fil)
            for frame in self.frames_reduced[fil]:
                new_frame = frame.correct_astrometry(output_dir=astrometry_fil_path)
                self.add_frame_astrometry(new_frame)

    def photometric_calibration(self, output_path: str):
        u.mkdir_check(output_path)

        deepest = self.coadded_trimmed[self.filters[0]]
        for fil in self.filters:
            img = self.coadded_trimmed[fil]
            for cat_name in retrieve.photometry_catalogues:
                if cat_name == "gaia":
                    continue
                fil_path = os.path.join(output_path, fil)
                u.mkdir_check(fil_path)
                if f"in_{cat_name}" in self.field.cats and self.field.cats[f"in_{cat_name}"]:
                    img.zeropoint(cat_path=self.field.get_path(f"cat_csv_{cat_name}"),
                                  output_path=os.path.join(fil_path, cat_name),
                                  cat_name=cat_name,
                                  dist_tol=0.2 * units.arcsec,
                                  show=False
                                  )

            zeropoint, cat = img.select_zeropoint()

            img.estimate_depth(zeropoint_name=cat)

            if img.depth > deepest.depth:
                deepest = img

        for fil in self.filters:
            img = self.coadded[fil]
            print(img.filter, img.depth)

        self.deepest_filter = deepest.filter
        self.deepest = deepest

        print("DEEPEST FILTER:", self.deepest_filter, self.deepest.depth)

    def trim_coadded(self, output_dir: str):
        u.mkdir_check(output_dir)
        template = None
        for fil in self.coadded:
            img = self.coadded[fil]
            output_path = os.path.join(output_dir, img.filename.replace(".fits", "_trimmed.fits"))
            trimmed = img.trim_from_area(output_path=output_path)
            if template is None:
                template = trimmed
            else:
                # Using the first image as a template, reproject this one into the pixel space (for alignment)
                trimmed = trimmed.reproject(other_image=template, output_path=output_path)
            self.add_coadded_trimmed_image(trimmed, key=fil)

    def coadd(self, output_dir: str, frames: str = "astrometry"):
        """
        Use Montage to coadd individual frames.
        :param output_dir: Directory in which to write data products.
        :param frames: Name of frames list to coadd.
        :return:
        """
        u.mkdir_check(output_dir)
        if frames == "astrometry":
            input_directory = self.paths['astrometry_dir']
        else:
            raise ValueError(f"{frames} not recognised as frame type.")
        for fil in self.filters:
            input_directory_fil = os.path.join(input_directory, fil)
            output_directory_fil = os.path.join(output_dir, fil)
            u.mkdir_check(output_directory_fil)
            coadded_path = montage.standard_script(
                input_directory=input_directory_fil,
                output_directory=output_directory_fil,
                output_file_name=f"{self.name}_{self.date.strftime('%Y-%m-%d')}_{fil}_coadded.fits")
            self.add_coadded_image(coadded_path, key=fil, mode="imaging")

    def _get_images(self, image_type: str):
        if image_type == "coadded_trimmed":
            image_dict = self.coadded_trimmed
        elif image_type == "coadded":
            image_dict = self.coadded
        else:
            raise ValueError(f"Images type '{image_type}' not recognised.")
        return image_dict

    def dual_mode_source_extraction(self, path: str, image_type: str = "coadded_trimmed"):
        image_dict = self._get_images(image_type=image_type)
        u.mkdir_check(path)
        for fil in image_dict:
            img = image_dict[fil]
            self.set_path("source_extraction_dual_dir", path)
            configs = self.source_extractor_config
            img.source_extraction_psf(output_dir=path,
                                      phot_autoparams=f"{configs['kron_factor']},{configs['kron_radius_min']}",
                                      template=self.deepest)

    def get_photometry(self, path: str, image_type: str = "coadded_trimmed"):
        """
        Retrieve photometric properties of key objects and write to disk.
        :param path: Path to which to write the data products.
        :return:
        """
        image_dict = self._get_images(image_type=image_type)
        u.mkdir_check(path)
        # Loop through filters
        for fil in image_dict:
            fil_output_path = os.path.join(path, fil)
            u.mkdir_check(fil_output_path)
            img = image_dict[fil]
            img.calibrate_magnitudes(zeropoint_name="best", dual=True)
            rows = []
            for obj in self.field.objects:
                plt.close()
                # Get nearest Source-Extractor object:
                nearest = img.find_object(obj.position)
                rows.append(nearest)
                err = nearest[f'MAGERR_AUTO_ZP_best']
                print("FILTER:", fil)
                print(f"MAG_AUTO = {nearest['MAG_AUTO_ZP_best']} +/- {err}")
                print(f"A = {nearest['A_WORLD'].to(units.arcsec)}; B = {nearest['B_WORLD'].to(units.arcsec)}")
                img.plot_source_extractor_object(nearest, output=os.path.join(fil_output_path, f"{obj.name}.png"),
                                                 show=False,
                                                 title=f"{obj.name}, {fil}-band, {nearest['MAG_AUTO_ZP_best'].round(3).value} ± {err.round(3)}")
                obj.cat_row = nearest
                obj.photometry[f"{fil}_{self.instrument}"] = {"mag": nearest['MAG_AUTO_ZP_best'],
                                                              "mag_err": err,
                                                              "a": nearest['A_WORLD'],
                                                              "b": nearest['B_WORLD'],
                                                              "ra": nearest['ALPHA_SKY'],
                                                              "ra_err": np.sqrt(nearest["ERRX2_WORLD"]),
                                                              "dec": nearest['DELTA_SKY'],
                                                              "dec_err": np.sqrt(nearest["ERRY2_WORLD"]),
                                                              "kron_radius": nearest["KRON_RADIUS"]}
                obj.update_output_file()
            tbl = table.hstack(rows)
            tbl.write(os.path.join(fil_output_path, f"{self.field.name}_{self.name}_{fil}.ecsv"),
                      format="ascii.ecsv")

    def _initial_setup(self):
        data_dir = self.data_path
        raw_dir = epoch_stage_dirs["0-download"]
        data_title = self.name

        # Write tables of fits files to main directory; firstly, science images only:
        tbl = image.fits_table(input_path=raw_dir,
                               output_path=os.path.join(data_dir, data_title + "_fits_table_science.csv"),
                               science_only=True)
        # Then including all calibration files
        tbl_full = image.fits_table(input_path=raw_dir,
                                    output_path=data_dir + data_title + "_fits_table_all.csv",
                                    science_only=False)

        image.fits_table_all(input_path=raw_dir,
                             output_path=data_dir + data_title + "_fits_table_detailed.csv",
                             science_only=False)

        for row in tbl_full:
            path = os.path.join(self.paths["raw_dir"], row["identifier"])
            cls = image.ImagingImage.select_child_class(instrument=self.instrument, mode="imaging")
            img = cls(path)
            img.extract_frame_type()
            img.extract_filter()
            # The below will also update the filter list.
            self.add_frame_raw(path)

        # Collect pointings of standard-star observations.
        for img in self.frames_standard:
            pointing = img.extract_pointing
            fil = img.extract_filter
            if pointing not in self.std_pointings[fil]:
                self.std_pointings[fil].append(pointing)

        # Collect and save some stats on those filters:

        template = self.frames_science[0]
        if self.target is None:
            self.target = template.object
        if self.date is None:
            self.set_date(self.frames_science[0].extract_date_obs())

        std_dir = os.path.join(data_dir, 'standards')

        for i, fil in enumerate(self.filters):

            exp_times = list(map(lambda frame: frame.exposure_time, self.frames_science[fil]))
            self.exp_time_mean[fil] = np.nanmean(exp_times)
            self.exp_time_err[fil] = np.nanstd(exp_times)

            airmasses = np.array(map(lambda frame: frame.airmass, self.frames_science[fil]))
            self.airmass_mean[fil] = np.nanmean(airmasses)
            self.airmass_err[fil] = max(np.nanmax(airmasses) - self.airmass_mean,
                                        self.airmass_mean - np.nanmin(airmasses))

            std_filter_dir = os.path.join(std_dir, fil)
            self.set_path(f"standard_dir_{fil}", std_filter_dir)
            u.mkdir_check(std_filter_dir)
            print(f'Copying {fil} calibration data to standard folder...')

            # Sort the STD files by filter, and within that by pointing.
            for j, pointing in enumerate(self.std_pointings):
                pointing_dir = os.path.join(std_filter_dir, f"RA{pointing.ra.value}_DEC{pointing.dec.value}")
                u.mkdir_check(pointing_dir)
                for std in self.frames_standard[fil]:
                    path_dest = os.path.join(pointing_dir, std.filename)
                    shutil.move(std.path, path_dest)
                    std.path = path_dest
                    std.update_output_file()

        self.update_output_file()

    def guess_data_path(self):
        if self.data_path is None and self.field is not None and self.field.data_path is not None and \
                self.instrument is not None and self.date is not None:
            self.data_path = os.path.join(self.field.data_path, "imaging", self.instrument,
                                          f"{self.date.isot}-{self.name}")
        return self.data_path

    def _output_dict(self):
        output_dict = super()._output_dict()
        if self.deepest is not None:
            deepest = self.deepest.path
        else:
            deepest = None

        output_dict.update({
            "filters": self.filters,
            "deepest": deepest,
            "deepest_filter": self.deepest_filter,
            "coadded": _output_img_dict_single(self.coadded),
            "coadded_trimmed": _output_img_dict_single(self.coadded_trimmed),
            "frames_raw": _output_img_list(self.frames_raw),
            "frames_reduced": _output_img_dict_list(self.frames_reduced),
            "frames_astrometry": _output_img_dict_list(self.frames_astrometry),
            "exp_time_mean": self.exp_time_mean,
            "exp_time_err": self.exp_time_err,
            "airmass_mean": self.airmass_mean,
            "airmass_err": self.airmass_err,
            "flats": _output_img_dict_list(self.flats)
        })
        return output_dict

    def load_output_file(self, **kwargs):
        outputs = super().load_output_file(**kwargs)
        if type(outputs) is dict:
            cls = image.Image.select_child_class(instrument=self.instrument, mode='imaging')
            if "filters" in outputs:
                self.filters = outputs["filters"]
            if "deepest" in outputs and outputs["deepest"] is not None:
                self.deepest = cls(path=outputs["deepest"])
            if "deepest_filter" in outputs:
                self.deepest_filter = outputs["deepest_filter"]
            if "exp_time_mean" in outputs:
                self.exp_time_mean = outputs["exp_time_mean"]
            if "exp_time_err" in outputs:
                self.exp_time_err = outputs["exp_time_err"]
            if "airmass_mean" in outputs:
                self.airmass_mean = outputs["airmass_mean"]
            if "airmass_err" in outputs:
                self.airmass_err = outputs["airmass_err"]
            if "frames_raw" in outputs:
                for frame in outputs["frames_raw"]:
                    self.add_frame_raw(raw_frame=frame)
            if "frames_reduced" in outputs:
                for fil in outputs["frames_reduced"]:
                    if outputs["frames_reduced"][fil] is not None:
                        for frame in outputs["frames_reduced"][fil]:
                            self.add_frame_reduced(reduced_frame=frame)
            if "frames_astrometry" in outputs:
                for fil in outputs["frames_astrometry"]:
                    if outputs["frames_astrometry"][fil] is not None:
                        for frame in outputs["frames_astrometry"][fil]:
                            self.add_frame_astrometry(astrometry_frame=frame)
            if "coadded" in outputs:
                for fil in outputs["coadded"]:
                    if outputs["coadded"][fil] is not None:
                        self.add_coadded_image(img=outputs["coadded"][fil], key=fil, **kwargs)
            if "coadded_trimmed" in outputs:
                for fil in outputs["coadded_trimmed"]:
                    if outputs["coadded_trimmed"][fil] is not None:
                        self.add_coadded_trimmed_image(img=outputs["coadded_trimmed"][fil], key=fil, **kwargs)

        return outputs

    def generate_gaia_astrometry_indices(self, correct_to_epoch: bool = True):
        if not isinstance(self.field, Field):
            raise ValueError("field has not been set for this observation.")
        self.field.retrieve_catalogue(cat_name="gaia")
        index_path = os.path.join(config["top_data_dir"], "astrometry_index_files")
        u.mkdir_check(index_path)
        cat_index_path = os.path.join(index_path, "gaia")
        gaia_csv_path = self.field.get_path(f"cat_csv_gaia")

        if correct_to_epoch:
            gaia_cat = am.correct_gaia_to_epoch(gaia_cat=gaia_csv_path,
                                                new_epoch=self.date)
        else:
            gaia_cat = retrieve.load_catalogue(cat_name="gaia", cat=gaia_csv_path)

        unique_id_prefix = int(self.field.name.replace("FRB", ""))

        am.generate_astrometry_indices(cat_name="gaia",
                                       cat=gaia_cat,
                                       output_file_prefix=f"gaia_index_{self.field.name}",
                                       index_output_dir=cat_index_path,
                                       fits_cat_output=gaia_csv_path.replace(".csv", ".fits"),
                                       p_lower=-1,
                                       p_upper=2,
                                       unique_id_prefix=unique_id_prefix,
                                       )
        return gaia_cat

    def add_frame_raw(self, raw_frame: Union[image.ImagingImage, str]):
        if isinstance(raw_frame, str):
            cls = image.ImagingImage.select_child_class(instrument=self.instrument)
            raw_frame = cls(path=raw_frame, frame_type="raw", instrument=self.instrument)
        fil = raw_frame.extract_filter()
        if self.check_filter(fil=fil):
            self.frames_raw[fil].append(raw_frame)
        self.sort_frame(raw_frame)

    def add_frame_reduced(self, reduced_frame: image.ImagingImage):
        if isinstance(reduced_frame, str):
            cls = image.ImagingImage.select_child_class(instrument=self.instrument)
            reduced_frame = cls(path=reduced_frame, frame_type="reduced")
        fil = reduced_frame.extract_filter()
        if self.check_filter(fil=fil) and reduced_frame not in self.frames_reduced[fil]:
            self.frames_reduced[fil].append(reduced_frame)

    def add_frame_astrometry(self, astrometry_frame: Union[str, image.ImagingImage]):
        if isinstance(astrometry_frame, str):
            cls = image.ImagingImage.select_child_class(instrument=self.instrument)
            astrometry_frame = cls(path=astrometry_frame, frame_type="astrometry")
        try:
            fil = astrometry_frame.extract_filter()
            if self.check_filter(fil=fil) and astrometry_frame not in self.frames_astrometry[fil]:
                self.frames_astrometry[fil].append(astrometry_frame)
        except FileNotFoundError:
            print(f"Astrometry file {astrometry_frame.path} not found.")

    def add_coadded_trimmed_image(self, img: Union[str, image.Image], key: str, **kwargs):
        if isinstance(img, str):
            img = image.CoaddedImage(path=img)
        img.epoch = self
        self.coadded_trimmed[key] = img
        return img

    def check_filter(self, fil: str):
        """
        If a filter name is not present in the various lists and dictionaries that use it, adds it.
        :param fil:
        :return: False if None, True if not.
        """
        if fil is not None:
            if fil not in self.filters:
                self.filters.append(fil)
            if fil not in self.frames_science:
                if isinstance(self.frames_science, dict):
                    self.frames_science[fil] = []
            if fil not in self.frames_reduced:
                if isinstance(self.frames_reduced, dict):
                    self.frames_reduced[fil] = []
            if fil not in self.frames_astrometry:
                self.frames_astrometry[fil] = []
            if fil not in self.coadded:
                self.coadded[fil] = None
            if fil not in self.coadded_trimmed:
                self.coadded_trimmed[fil] = None
            if fil not in self.exp_time_mean:
                self.exp_time_mean[fil] = None
            if fil not in self.exp_time_err:
                self.exp_time_err[fil] = None
            if fil not in self.airmass_mean:
                self.airmass_mean[fil] = None
            if fil not in self.airmass_err:
                self.airmass_err[fil] = None
            if fil not in self.std_pointings:
                self.std_pointings[fil] = None
            if fil not in self.flats:
                self.flats[fil] = []
            return True
        else:
            return False

    def plot_object(self, img: str, fil: str, fig: plt.Figure,
                    centre: SkyCoord,
                    frame: units.Quantity = 30 * units.pix,
                    n: int = 1, n_x: int = 1, n_y: int = 1,
                    cmap: str = 'viridis', show_cbar: bool = False,
                    stretch: str = 'sqrt',
                    vmin: float = None,
                    vmax: float = None,
                    show_grid: bool = False,
                    ticks: int = None, interval: str = 'minmax',
                    show_coords: bool = True,
                    font_size: int = 12,
                    reverse_y=False,
                    **kwargs):
        if img == "coadded":
            to_plot = self.coadded[fil]
        else:
            raise ValueError(f"img type {img} not recognised.")
        subplot, hdu_cut = to_plot.plot_subimage(fig=fig, frame=frame,
                                                 centre=centre,
                                                 n=n, n_x=n_x, n_y=n_y,
                                                 cmap=cmap, show_cbar=show_cbar, stretch=stretch,
                                                 vmin=vmin, vmax=vmax,
                                                 show_grid=show_grid,
                                                 ticks=ticks, interval=interval,
                                                 show_coords=show_coords,
                                                 font_size=font_size,
                                                 reverse_y=reverse_y,
                                                 **kwargs
                                                 )
        return subplot, hdu_cut

    @classmethod
    def stages(cls):
        stages = super().stages()
        stages.update({"5-correct_astrometry_frames": None,
                       "6-coadd": None,
                       "7-trim_coadded": None,
                       "8-source_extraction": None,
                       "9-photometric_calibration": None,
                       "10-dual_mode_source_extraction": None,
                       "11-get_photometry": None
                       })
        return stages

    @classmethod
    def from_params(cls, name: str, instrument: str, field: Union[Field, str] = None, old_format: bool = False):
        instrument = instrument.lower()
        field_name, field = cls._from_params_setup(name=name, field=field)
        if old_format:
            instrument = instrument.split("-")[-1]
            path = os.path.join(p.param_dir, f"epochs_{instrument}", name)
        else:
            path = cls.build_param_path(instrument_name=instrument,
                                        field_name=field_name,
                                        epoch_name=name)
        return cls.from_file(param_file=path, field=field)

    @classmethod
    def build_param_path(cls, instrument_name: str, field_name: str, epoch_name: str):
        return os.path.join(p.param_dir, "fields", field_name, "imaging", instrument_name, epoch_name)

    @classmethod
    def from_file(cls, param_file: Union[str, dict], old_format: bool = False, field: Field = None):

        name, param_file, param_dict = p.params_init(param_file)

        if param_dict is None:
            raise FileNotFoundError("Param file missing!")

        if old_format:
            instrument = "vlt-fors2"
        else:
            instrument = param_dict["instrument"].lower()

        if field is None:
            field = param_dict["field"]

        sub_cls = cls.select_child_class(instrument=instrument)
        if sub_cls is ImagingEpoch:
            return cls(name=name,
                       field=field,
                       param_path=param_file,
                       data_path=os.path.join(config["top_data_dir"], param_dict['data_path']),
                       instrument=instrument,
                       date=param_dict['date'],
                       program_id=param_dict["program_id"],
                       target=param_dict["target"],
                       source_extractor_config=param_dict['sextractor'],
                       )
        elif sub_cls is FORS2ImagingEpoch:
            return sub_cls.from_file(param_dict, name=name, old_format=old_format, field=field)
        else:
            return sub_cls.from_file(param_dict, name=name, field=field)

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
            "background_subtraction":
                {"renormalise_centre": objects.position_dictionary.copy(),
                 "test_synths":
                     [{"position": objects.position_dictionary.copy(),
                       "mags": {}
                       }]

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
    def select_child_class(cls, instrument: str):
        instrument = instrument.lower()
        if instrument == "vlt-fors2":
            return FORS2ImagingEpoch
        if instrument == "panstarrs1":
            return PanSTARRS1ImagingEpoch
        if instrument == "gs-aoi":
            return GSAOIImagingEpoch
        elif instrument in instruments_imaging:
            return ImagingEpoch
        else:
            raise ValueError(f"Unrecognised instrument {instrument}")


class GSAOIImagingEpoch(ImagingEpoch):
    """
    This class works a little differently to the other epochs; instead of keeping track of the files internally, we let
    DRAGONS do that for us. Thus, many of the dictionaries and lists of files used in other Epoch classes
    will be empty even if the files are actually being tracked correctly. See eg science_table instead.
    """
    instrument = "gs-aoi"

    def __init__(self,
                 name: str = None,
                 field: Union[str, Field] = None,
                 param_path: str = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 program_id: str = None,
                 target: str = None,
                 source_extractor_config: dict = None,
                 ):
        super().__init__(name=name,
                         field=field,
                         param_path=param_path,
                         data_path=data_path,
                         instrument=instrument,
                         date=date,
                         program_id=program_id,
                         target=target,
                         source_extractor_config=source_extractor_config)
        self.science_table = None
        self.flats_lists = {}
        self.std_lists = {}

    @classmethod
    def stages(cls):
        stages = super().stages()
        stages.update({"0-download": None,
                       "2-reduce_flats": None,
                       "3-reduce_science": None
                       })
        return stages

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)
        self.proc_0_download(**kwargs)
        self.proc_1_initial_setup(**kwargs)
        self.proc_2_reduce_flats(**kwargs)
        self.proc_3_reduce_science(**kwargs)

    def proc_0_download(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Download raw data from Gemini archive?", stage='0-download'):
            if 'overwrite_download' in kwargs:
                overwrite = kwargs['overwrite_download']
            else:
                overwrite = False
            self.retrieve(overwrite=overwrite)
            self.stages_complete['0-download'] = Time.now()
            self.update_output_file()

    def retrieve(self, overwrite: bool = False):
        raw_dir_full = self.paths["raw_dir"]
        u.mkdir_check(raw_dir_full)

        # Get the calibration files
        retrieve.save_gemini_calibs(output=raw_dir_full,
                                    obs_date=self.date,
                                    overwrite=overwrite)

        # Get the science files
        retrieve.save_gemini_epoch(output=raw_dir_full,
                                   program_id=self.program_id,
                                   coord=self.field.centre_coords,
                                   overwrite=overwrite)

    def _initial_setup(self):
        data_dir = self.data_path
        raw_dir = self.paths["raw_dir"]
        self.paths["redux_dir"] = redux_dir = os.path.join(data_dir, "1-reduced")
        u.mkdir_check(redux_dir)
        # DO the initial database setup for DRAGONS.
        dragons.caldb_init(redux_dir=redux_dir)

        # Get a list of science files from the raw directory, using DRAGONS.
        science_list_name = "science.list"
        science_list = dragons.data_select(
            redux_dir=redux_dir,
            raw_dir=raw_dir,
            expression="observation_class==\"science\"",
            output=science_list_name
        ).splitlines(False)[3:]
        self.paths["science_list"] = os.path.join(redux_dir, science_list_name)

        science_tbl_name = "science.csv"
        science_tbl = dragons.showd(
            input_filenames=science_list,
            descriptors="filter_name,exposure_time,object",
            output=science_tbl_name,
            csv=True,
            working_dir=redux_dir
        )
        # Set up filters.
        for img in science_tbl:
            fil = img["filter_name"]
            fil = fil[:fil.find("_")]
            self.check_filter(fil)
        print(self.filters)
        self.science_table = science_tbl

        # Get lists of flats for each filter.
        for fil in self.flats_lists:
            flats_list_name = f"flats_{fil}.list"
            flats_list = dragons.data_select(
                redux_dir=redux_dir,
                raw_dir=raw_dir,
                tags=["FLAT"],
                expression=f"filter_name==\"{fil}\"",
                output=flats_list_name
            ).splitlines(False)[3:]
            self.flats_lists[fil] = os.path.join(redux_dir, flats_list_name)
            self.flats[fil] = flats_list

        # Get list of standard observations:
        std_tbl_name = "std_objects.csv"
        self.paths["std_tbl"] = os.path.join(redux_dir, std_tbl_name)
        std_list_name = "std_objects.list"
        self.paths["std_list"] = os.path.join(redux_dir, std_list_name)
        std_list = dragons.data_select(
            redux_dir=redux_dir,
            raw_dir=raw_dir,
            expression=f"observation_class==\"partnerCal\"",
            output=std_list_name
        )
        std_tbl = dragons.showd(
            input_filenames=std_list,
            descriptors="object",
            output=std_tbl_name,
            csv=True,
            working_dir=redux_dir
        )

        # Set up dictionary of standard objects
        # TODO: ACCOUNT FOR MULTIPLE FILTERS.
        for std in std_tbl:
            if std["object"] not in self.std_objects:
                self.std_objects[std["object"]] = None

        for obj in self.std_objects:
            # And get the individual objects imaged like so:
            std_list_obj_name = f"std_{obj}.list"
            std_list_obj = dragons.data_select(
                redux_dir=redux_dir,
                raw_dir=raw_dir,
                expression=f"object==\"{obj}\"",
                output=std_list_obj_name
            ).splitlines(False)[3:]
            self.std_objects[obj] = std_list_obj
            self.std_lists[obj] = os.path.join(redux_dir, std_list_obj_name)

    def proc_2_reduce_flats(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage(stage="2-reduce_flats",
                                        message="Reduce flat-field images?"):
            for fil in self.flats_lists:
                dragons.reduce(self.flats_lists[fil], redux_dir=self.paths["redux_dir"])
            flat_dir = os.path.join(self.paths["redux_dir"], "calibrations", "processed_flat")
            for flat in os.listdir(flat_dir):
                os.system(f"caldb add {flat}")
            self.stages_complete['2-reduce_flats'] = Time.now()
            self.update_output_file()

    def proc_3_reduce_science(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage(stage="3-reduce_science",
                                        message="Reduce science images?"):
            dragons.reduce(self.paths["science_list"], redux_dir=self.paths["redux_dir"])
            self.stages_complete['3-reduce_science'] = Time.now()
            self.update_output_file()

    def check_filter(self, fil: str):
        not_none = super().check_filter(fil)
        if not_none:
            self.flats_lists[fil] = None
        return not_none

    def _output_dict(self):
        output_dict = super()._output_dict()
        output_dict.update({
            "science_table": self.science_table,
            "flats_lists": self.flats_lists,
            "std_lists": self.std_lists
        })
        return output_dict

    # def load_sc

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        # default_params.update({})
        return default_params

    @classmethod
    def from_file(cls, param_file: Union[str, dict], name: str = None, field: Field = None):

        name, param_file, param_dict = p.params_init(param_file)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")

        if field is None:
            field = param_dict["field"]
        if 'target' in param_dict:
            target = param_dict['target']
        else:
            target = None

        return cls(name=name,
                   field=field,
                   param_path=param_file,
                   data_path=os.path.join(config["top_data_dir"], param_dict['data_path']),
                   instrument='gs-aoi',
                   program_id=param_dict['program_id'],
                   date=param_dict['date'],
                   target=target,
                   source_extractor_config=param_dict['sextractor'])

    @classmethod
    def sort_files(cls, input_dir: str, output_dir: str = None, tolerance: units.Quantity = 3 * units.arcmin):
        """
        A routine to sort through a directory containing an arbitrary number of GSAOI observations and assign epochs to
        them.

        :param input_dir:
        :param tolerance: Images will be grouped if they are < tolerance from each other (or, specifically, from the
        first encountered in that cluster).
        :return:
        """
        pointings = {}
        if output_dir is None:
            output_dir = input_dir
        u.mkdir_check(output_dir)
        files = os.listdir(input_dir)
        files.sort()
        for file in filter(lambda f: f.endswith(".fits"), files):
            # Since GSAOI science files cannot be relied upon to include the object/target in the header, we group
            # images by RA and Dec.
            path = os.path.join(input_dir, file)
            img = image.GSAOIImage(path=path)
            pointing = img.extract_pointing()
            associated = False
            for pointing_str in pointings:
                pointings_list = pointings[pointing_str]
                other_pointing = pointings_list[0]
                if pointing.separation(other_pointing) <= tolerance:
                    pointings_list.append(pointing)
                    associated = True
                    shutil.move(path, pointing_str)
                    break

            if not associated:
                pointing_str = os.path.join(output_dir, pointing.to_string())
                u.mkdir_check(pointing_str)
                pointings[pointing_str] = [pointing]
                shutil.move(path, pointing_str)

        return pointings


class PanSTARRS1ImagingEpoch(ImagingEpoch):
    instrument = "panstarrs1"

    def __init__(self,
                 name: str = None,
                 field: Union[str, Field] = None,
                 param_path: str = None,
                 data_path: str = None,
                 source_extractor_config: dict = None
                 ):
        super().__init__(name=name,
                         field=field,
                         param_path=param_path,
                         data_path=data_path,
                         source_extractor_config=source_extractor_config,
                         instrument="panstarrs1"
                         )
        # self.load_output_file()
        if isinstance(field, Field):
            self.field.retrieve_catalogue(cat_name="panstarrs1")

    # TODO: Automatic cutout download; don't worry for now.

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)
        self.proc_0_download(**kwargs)
        self.proc_1_initial_setup(**kwargs)
        self.proc_2_source_extraction(**kwargs)
        self.proc_3_photometric_calibration(**kwargs)
        self.proc_4_dual_mode_source_extraction(**kwargs)
        self.proc_5_get_photometry(**kwargs)

    def proc_0_download(self, **kwargs):
        """
        Automatically download PanSTARRS1 cutout.
        @param kwargs:
        @return:
        """
        pass

    def proc_2_source_extraction(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do source extraction?", stage='2-source_extraction'):
            source_extraction_path = os.path.join(self.data_path, "2-source_extraction")
            u.mkdir_check(source_extraction_path)
            for fil in self.coadded:
                img = self.coadded[fil]
                self.set_path("source_extraction_dir", source_extraction_path)
                configs = self.source_extractor_config
                img.source_extraction_psf(output_dir=source_extraction_path,
                                          phot_autoparams=f"{configs['kron_factor']},{configs['kron_radius_min']}")
            self.stages_complete['2-source_extraction'] = Time.now()
            self.update_output_file()

    def proc_3_photometric_calibration(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do photometric calibration?", stage="3-photometric_calibration"):
            calib_dir = os.path.join(self.data_path, "3-photometric_calibration")
            self.photometric_calibration(calib_dir)
            self.stages_complete['3-photometric_calibration'] = Time.now()
            self.update_output_file()

    def proc_4_dual_mode_source_extraction(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do source extraction in dual-mode, using deepest image as footprint?",
                                        stage="4-dual_mode_source_extraction"):
            source_extraction_path = os.path.join(self.data_path, "4-dual_mode_source_extraction")
            self.dual_mode_source_extraction(source_extraction_path, image_type="coadded")
            self.stages_complete["4-dual_mode_source_extraction"] = Time.now()
            self.update_output_file()

    def proc_5_get_photometry(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Get photometry?", stage="5-get_photometry"):
            object_property_path = os.path.join(self.data_path, "5-object_properties")
            self.get_photometry(object_property_path, image_type="coadded")

            self.stages_complete["5-get_photometry"] = Time.now()
            self.update_output_file()

    def _initial_setup(self):
        imaging_dir = os.path.join(self.data_path, "0-imaging")
        u.mkdir_check(imaging_dir)
        for file in filter(lambda f: f.endswith(".fits"), os.listdir(self.data_path)):
            shutil.move(os.path.join(self.data_path, file), imaging_dir)
        self.set_path("imaging_dir", imaging_dir)
        # Write a table of fits files from the 0-imaging directory.
        table_path_all = os.path.join(self.data_path, f"{self.name}_fits_table_all.csv")
        self.set_path("fits_table", table_path_all)
        image.fits_table_all(input_path=imaging_dir, output_path=table_path_all, science_only=False)
        for file in filter(lambda f: f.endswith(".fits"), os.listdir(imaging_dir)):
            path = os.path.join(imaging_dir, file)
            img = image.PanSTARRS1Cutout(path=path)

            # img.open(mode="update")
            # print(img.hdu_list.info())
            # if len(img.hdu_list) == 2:
            #     img.hdu_list[0] = img.hdu_list[1]
            #     img.hdu_list.pop(1)
            # img.close()

            img.extract_filter()
            self.coadded[img.filter] = img
            self.check_filter(img.filter)

    def guess_data_path(self):
        if self.data_path is None and self.field is not None and self.field.data_path is not None:
            self.data_path = os.path.join(self.field.data_path, "imaging", "panstarrs1")
        return self.data_path

    def photometric_calibration(self, output_path: str):
        u.mkdir_check(output_path)

        deepest = self.coadded[self.filters[0]]
        for fil in self.filters:
            img = self.coadded[fil]
            img.zeropoint(cat_path=self.field.get_path("cat_csv_panstarrs1"),
                          output_path=os.path.join(output_path, img.name),
                          cat_name="PanSTARRS1",
                          image_name="PanSTARRS Cutout",
                          )
            img.zeropoint_best = img.zeropoints["panstarrs1"]
            img.estimate_depth(zeropoint_name="panstarrs1")

            if img.depth > deepest.depth:
                deepest = img

        for fil in self.filters:
            img = self.coadded[fil]
            print(img.filter, img.depth)

        self.deepest_filter = deepest.filter
        self.deepest = deepest

        print("DEEPEST FILTER:", self.deepest_filter, self.deepest.depth)

    def add_coadded_image(self, img: Union[str, image.Image], key: str, **kwargs):
        if isinstance(img, str):
            img = image.PanSTARRS1Cutout(path=img)
        img.epoch = self
        self.coadded[key] = img
        return img

    @classmethod
    def stages(cls):
        param_dict = super().stages()
        param_dict.update({"0-download": None,
                           "2-source_extraction": None,
                           "3-photometric_calibration": None,
                           "4-dual_mode_source_extraction": None,
                           "5-get_photometry": None
                           })
        return param_dict

    @classmethod
    def from_file(cls, param_file: Union[str, dict], name: str = None, field: Field = None):
        name, param_file, param_dict = p.params_init(param_file)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")

        if field is None:
            field = param_dict["field"]

        epoch = cls(name=name,
                    field=field,
                    param_path=param_file,
                    data_path=os.path.join(config["top_data_dir"], param_dict['data_path']),
                    source_extractor_config=param_dict['sextractor'])
        epoch.instrument = cls.instrument
        return epoch


class ESOImagingEpoch(ImagingEpoch):

    def __init__(self,
                 name: str = None,
                 field: Field = None,
                 param_path: str = None,
                 data_path: str = None,
                 instrument: str = None,
                 program_id: str = None,
                 date: Union[str, Time] = None,
                 target: str = None,
                 standard_epochs: list = None,
                 source_extractor_config: dict = None):
        super().__init__(name=name, field=field, param_path=param_path, data_path=data_path, instrument=instrument,
                         date=date, program_id=program_id, target=target,
                         standard_epochs=standard_epochs, source_extractor_config=source_extractor_config)

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)
        self.proc_0_download(**kwargs)
        self.proc_1_initial_setup(**kwargs)

    def proc_0_download(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Download raw data from ESO archive?", stage='0-download'):
            r = self.retrieve()
            if r:
                self.stages_complete['0-download'] = Time.now()
                self.update_output_file()

    def retrieve(self):
        """
        Check ESO archive for the epoch raw frames, and download those frames and associated files.
        :return:
        """
        r = []
        if "raw_dir" in self.paths:
            r = _retrieve_eso_epoch(self, path=self.paths["raw_dir"])
        else:
            warnings.warn("raw_dir has not been set. Retrieve could not be run.")
        return r

    @classmethod
    def stages(cls):
        param_dict = super().stages()
        param_dict.update({"0-download": None,
                           })
        return param_dict


class FORS2ImagingEpoch(ESOImagingEpoch):

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)
        self.proc_5_correct_astrometry_frames(**kwargs)
        self.proc_6_coadd(**kwargs)
        self.proc_7_trim_coadded(**kwargs)
        self.proc_8_source_extraction(**kwargs)
        self.proc_9_photometric_calibration(**kwargs)
        self.proc_10_dual_mode_source_extraction(**kwargs)
        self.proc_11_get_photometry(**kwargs)

    def proc_1_initial_setup(self, no_query: bool = False, **kwargs):
        if super().proc_1_initial_setup(no_query, **kwargs):
            for fil in self.filters:
                self.pair_files(self.frames_science[fil])

    def correct_astrometry_frames(self, output_path: str, **kwargs):
        self.frames_astrometry = {}
        method = True
        if "method" in kwargs:
            method = kwargs["method"]

        for fil in self.frames_reduced:
            astrometry_fil_path = os.path.join(output_path, fil)
            if method == "pairwise":
                pairs = self.pair_files(self.frames_reduced[fil])
                reverse_pair = False
                for pair in pairs:
                    if isinstance(pair, tuple):
                        img_1, img_2 = pair
                        success = False
                        failed_first = False
                        while not success:  # The SystemError should stop this from looping indefinitely.
                            if not reverse_pair:
                                new_img_1 = img_1.correct_astrometry(output_dir=astrometry_fil_path)
                                # Check if the first astrometry run was successful.
                                # If it wasn't, we need to be running on the second image of the pair.
                                if new_img_1 is None:
                                    reverse_pair = True
                                    failed_first = True
                                    print(
                                        f"Astrometry.net failed to solve {img_1}, trying on opposite chip {img_2}.")
                                else:
                                    self.add_frame_astrometry(new_img_1)
                                    new_img_2 = img_2.correct_astrometry_from_other(new_img_1,
                                                                                    output_dir=astrometry_fil_path)
                                    self.add_frame_astrometry(new_img_2)
                                    success = True
                            # We don't use an else statement here because reverse_pair can change within the above block,
                            # and if it does the block below needs to execute.
                            if reverse_pair:
                                new_img_2 = img_2.correct_astrometry(output_dir=astrometry_fil_path)
                                if new_img_2 is None:
                                    if failed_first:
                                        raise SystemError(
                                            f"Astrometry.net failed to solve both chips of this pair ({img_1}, {img_2})")
                                    else:
                                        reverse_pair = False
                                else:
                                    self.add_frame_astrometry(new_img_2)
                                    new_img_1 = img_1.correct_astrometry_from_other(new_img_2,
                                                                                    output_dir=astrometry_fil_path)
                                    self.add_frame_astrometry(new_img_1)
                                    success = True
                    else:
                        new_img = pair.correct_astrometry(output_dir=astrometry_fil_path)
                        self.add_frame_astrometry(new_img)

            else:
                for img in self.frames_reduced[fil]:
                    new_img = img.correct_astrometry(output_dir=astrometry_fil_path)
                    if new_img is not None:
                        self.add_frame_astrometry(new_img)

    @classmethod
    def pair_files(cls, images: list):
        pairs = []
        images.sort(key=lambda im: im.name)
        is_paired = True
        for i, img_1 in enumerate(images):
            # If the images are in pairs, it's sufficient to check only the even-numbered ones.
            # If not, is_paired=False should be triggered by the case below.
            if i % 2 == 0 or not is_paired:
                chip_this = img_1.extract_chip_number()
                # If we are at the end of the list and still checking, this must be unpaired.
                if i + 1 == len(images):
                    pair = img_1
                else:
                    # Get the next image in the list.
                    img_2 = images[i + 1]
                    chip_other = img_2.extract_chip_number()
                    # If we have chip
                    if (chip_this == 1 and chip_other == 2) or (chip_this == 2 and chip_other == 1):
                        img_1.other_chip = img_2
                        img_1.update_output_file()
                        img_2.other_chip = img_1
                        img_2.update_output_file()
                        if chip_this == 1:
                            pair = (img_1, img_2)
                        elif chip_this == 2:
                            pair = (img_2, img_1)
                        else:
                            raise ValueError("Image is missing chip.")
                        is_paired = True
                    else:
                        is_paired = False
                        pair = img_1
                print("PAIR:")
                if isinstance(pair, tuple):
                    print(str(pair[0]), ",", str(pair[1]))
                else:
                    print(pair)
                pairs.append(pair)

        return pairs

    @classmethod
    def from_file(cls, param_file: Union[str, dict], name: str = None, old_format: bool = False, field: Field = None):

        if old_format:
            if name is None:
                raise ValueError("name must be provided for old_format=True.")
            param_file = cls.convert_old_params(old_epoch_name=name)
        name, param_file, param_dict = p.params_init(param_file)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")

        if field is None:
            field = param_dict["field"]
        if 'target' in param_dict:
            target = param_dict['target']
        else:
            target = None

        return cls(name=name,
                   field=field,
                   param_path=param_file,
                   data_path=os.path.join(config["top_data_dir"], param_dict['data_path']),
                   instrument='vlt-fors2',
                   program_id=param_dict['program_id'],
                   date=param_dict['date'],
                   target=target,
                   source_extractor_config=param_dict['sextractor'])

    @classmethod
    def convert_old_params(cls, old_epoch_name: str):
        new_params = cls.new_yaml(name=old_epoch_name, path=None)
        old_params = p.object_params_fors2(old_epoch_name)

        new_epoch_name = f"FRB20{old_epoch_name[3:]}"

        old_field = old_epoch_name[:old_epoch_name.find('_')]
        new_field = new_epoch_name[:new_epoch_name.find('_')]

        old_data_dir = old_params["data_dir"]
        i = old_data_dir.find("MJD")
        mjd = old_data_dir[i + 3:i + 8]
        t = Time(mjd, format="mjd")
        date = t.strftime("%Y-%m-%d")
        new_data_dir = os.path.join(p.config['top_data_dir'], new_field, "imaging", "vlt-fors2",
                                    f"{date}-{new_epoch_name}")

        new_params["instrument"] = "vlt-fors2"
        new_params["data_path"] = new_data_dir
        new_params["field"] = new_field
        new_params["name"] = new_epoch_name
        new_params["date"] = date

        new_params["sextractor"]["aperture_diameters"] = old_params["photometry_apertures"]
        new_params["sextractor"]["dual_mode"] = old_params["do_dual_mode"]
        new_params["sextractor"]["threshold"] = old_params["threshold"]
        new_params["sextractor"]["kron_factor"] = old_params["sextractor_kron_radius"]
        new_params["sextractor"]["kron_radius_min"] = old_params["sextractor_min_radius"]

        filters = filter(lambda k: k.endswith("_star_class_tol"), old_params)
        filters = list(map(lambda f: f[0], filters))

        new_params["background_subtraction"]["renormalise_centre"]["dec"] = old_params["renormalise_centre_dec"]
        new_params["background_subtraction"]["renormalise_centre"]["ra"] = old_params["renormalise_centre_ra"]
        new_params["background_subtraction"]["test_synths"] = []
        if old_params["test_synths"]["ra"]:
            for i, _ in enumerate(old_params["test_synths"]):
                synth_dict = {"position": {}}
                synth_dict["position"]["ra"] = old_params["test_synths"]["ra"][i]
                synth_dict["position"]["dec"] = old_params["test_synths"]["dec"][i]
                synth_dict["mags"] = {}
                synth_dict["mags"]["g"] = old_params["test_synths"]["g_mag"][i]
                synth_dict["mags"]["I"] = old_params["test_synths"]["I_mag"][i]
                new_params["background_subtraction"]["test_synths"].append(synth_dict)

        new_params["skip"]["esoreflex_copy"] = old_params["skip_copy"]
        new_params["skip"]["sextractor_individual"] = not old_params["do_sextractor_individual"]
        new_params["skip"]["astrometry_net"] = old_params["skip_astrometry"]
        new_params["skip"]["sextractor"] = not old_params["do_sextractor"]
        new_params["skip"]["esorex"] = old_params["skip_esorex"]

        instrument_path = os.path.join(p.param_dir, "fields", new_field, "imaging", "vlt-fors2")
        u.mkdir_check(instrument_path)
        output_path = os.path.join(instrument_path, new_epoch_name)
        p.save_params(file=output_path,
                      dictionary=new_params, quiet=False)

        return output_path


class SpectroscopyEpoch(Epoch):
    grisms = {}

    def __init__(self,
                 param_path: str = None,
                 name: str = None,
                 field: Union[str, Field] = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 program_id: str = None,
                 target: str = None,
                 grism: str = None,
                 decker: str = None
                 ):
        super().__init__(param_path=param_path, name=name, field=field, data_path=data_path, instrument=instrument,
                         date=date, program_id=program_id, target=target)

        self.decker = decker
        self.decker_std = decker
        self.grism = grism
        if grism is None or grism not in self.grisms:
            warnings.warn("grism not configured.")

        self.obj = target

        self._path_2_pypeit()
        self.standards_raw = []
        self._instrument_pypeit = self.instrument.replace('-', '_')

        self._pypeit_file = None
        self._pypeit_sorted_file = None
        self._pypeit_coadd1d_file = None

    def proc_4_pypeit_flux(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do fluxing with PypeIt?", stage='4-pypeit_flux_calib'):
            self._pypeit_flux()
            self.stages_complete['4-pypeit_flux_calib'] = Time.now()
            self.update_output_file()

    def _pypeit_flux(self):
        pypeit_run_dir = self.get_path("pypeit_run_dir")
        pypeit_science_dir = os.path.join(pypeit_run_dir, "Science")
        std_reduced_filename = filter(lambda f: "spec1d" in f and "STD,FLUX" in f and f.endswith(".fits"),
                                      os.listdir(pypeit_science_dir)).__next__()
        std_reduced_path = os.path.join(pypeit_science_dir, std_reduced_filename)
        print(f"Using {std_reduced_path} for fluxing.")

        sensfunc_path = os.path.join(pypeit_run_dir, "sens.fits")
        # Generate sensitivity function from standard observation
        spec.pypeit_sensfunc(spec1dfile=std_reduced_path, outfile=sensfunc_path)
        # Generate flux setup file.
        spec.pypeit_flux_setup(sci_path=pypeit_science_dir, run_dir=pypeit_run_dir)
        flux_setup_path = os.path.join(pypeit_run_dir, self.pypeit_flux_title())
        # Insert name of sensitivity file to flux setup file.
        with open(flux_setup_path, "r") as flux_setup:
            flux_lines = flux_setup.readlines()
        file_first = flux_lines.index("flux read\n") + 1
        flux_lines[file_first] = flux_lines[file_first][:-1] + " " + sensfunc_path + "\n"
        # Write back to file.
        u.write_list_to_file(path=flux_setup_path, file=flux_lines)
        # Run pypeit_flux_calib
        os.system(f"pypeit_flux_calib {flux_setup_path}")

        self.set_path("pypeit_sensitivity_file", sensfunc_path)
        self.set_path("pypeit_std_reduced", std_reduced_path)
        self.set_path("pypeit_science_dir", pypeit_science_dir)
        self.set_path("pypeit_flux_setup", flux_setup_path)

    def pypeit_flux_title(self):
        return f"{self._instrument_pypeit}.flux"

    def read_pypeit_sorted_file(self):
        if "pypeit_setup_dir" in self.paths and self.paths["pypeit_setup_dir"] is not None:
            setup_files = self.paths["pypeit_setup_dir"]
            sorted_path = os.path.join(setup_files,
                                       filter(lambda f: f.endswith(".sorted"), os.listdir(setup_files)).__next__())
            with open(sorted_path) as sorted_file:
                self._pypeit_sorted_file = sorted_file.readlines()
        else:
            raise KeyError("pypeit_setup_dir has not been set.")

    def setup_info(self, setup: str):
        """
        Pulls setup info from a pypeit .sorted file.
        :param setup:
        :return:
        """
        file = self._get_pypeit_sorted_file()
        # Find start of setup description
        setup_start = file.index(f"Setup {setup}\n")
        setup_dict = {}
        i = setup_start + 1
        line = file[i]
        # Assemble a dictionary of the setup parameters.
        while line != "#---------------------------------------------------------\n":
            while line[0] == " ":
                line = line[1:]
            line = line[:-1]
            key, value = line.split(": ")
            setup_dict[key] = value
            i += 1
            line = file[i]
        return setup_dict

    def _read_pypeit_file(self, filename):
        pypeit_run_dir = os.path.join(self.paths['pypeit_dir'], filename)
        self.set_path("pypeit_run_dir", pypeit_run_dir)
        self.set_path("pypeit_file", os.path.join(pypeit_run_dir, f"{filename}.pypeit"))
        # Retrieve text from .pypeit file
        with open(self.get_path("pypeit_file"), 'r') as pypeit_file:
            pypeit_lines = pypeit_file.readlines()
            self._set_pypeit_file(pypeit_lines)
        f_start = pypeit_lines.index("data read\n") + 3
        f_end = pypeit_lines.index("data end\n")
        for line in pypeit_lines[f_start:f_end]:
            raw = image.SpecRaw.from_pypeit_line(line=line, pypeit_raw_path=self.paths["raw_dir"])
            self.add_frame_raw(raw)
        return pypeit_lines

    def read_pypeit_file(self, setup: str):
        if "pypeit_dir" in self.paths and self.paths["pypeit_dir"] is not None:
            filename = f"{self._instrument_pypeit}_{setup}"
            self._read_pypeit_file(filename=filename)
            return self._pypeit_file
        else:
            raise KeyError("pypeit_run_dir has not been set.")

    def write_pypeit_file_science(self):
        """
        Rewrites the stored .pypeit file to disk at its original path.
        :return: path of .pypeit file.
        """
        pypeit_lines = self._get_pypeit_file()
        if pypeit_lines is not None:
            pypeit_file_path = self.get_path("pypeit_file")
            u.write_list_to_file(path=pypeit_file_path, file=pypeit_lines)
        else:
            raise ValueError("pypeit_file has not yet been read.")
        return pypeit_file_path

    def add_pypeit_user_param(self, param: list, value: str, file_type: str = "pypeit"):
        """
        Inserts a parameter for the PypeIt run at the correct point in the stored .pypeit file.
        :param param: For m
        :param value:
        :return:
        """
        if file_type == "pypeit":
            pypeit_file = self._get_pypeit_file()
        elif file_type == "coadd1d":
            pypeit_file = self._get_pypeit_coadd1d_file()
        else:
            raise ValueError(f"file_type {file_type} not recognised.")

        if pypeit_file is not None:
            # Build the final line of the setting specially.
            setting = "\t" * (len(param) - 1) + f"{param.pop()} = {value}\n"
            p_start = pypeit_file.index("# User-defined execution parameters\n") + 1
            insert_here = False
            # For each level of the param list, look to see if it's already there.
            for i, line in pypeit_file[p_start]:
                if param[0] in line:
                    p_start = i
                    break

            for i, par in enumerate(param):
                # Encase each level of the parameter in the correct number of square brackets and tabs.
                par = "\t" * i + "[" * (i + 1) + par + "]" * (i + 1) + "\n"
                # First, check if param sub-headings are already there:
                if par in pypeit_file and not insert_here:
                    p_start = pypeit_file.index(par) + 1
                else:
                    # Insert the line at correct position.
                    pypeit_file.insert(p_start, par)
                    p_start += 1
                    insert_here = True
            # Insert the final line.
            pypeit_file.insert(p_start, setting)

            if file_type == "pypeit":
                self._set_pypeit_file(lines=pypeit_file)
            elif file_type == "coadd1d":
                self._set_pypeit_coadd1d_file(lines=pypeit_file)

        else:
            raise ValueError("pypeit_file has not yet been read.")

    def add_pypeit_file_lines(self, lines: list):
        if self._get_pypeit_file() is not None:
            # Remove last two lines of file ("data end")
            pypeit_lines = self._pypeit_file[:-2]
            # Insert desired lines
            pypeit_lines += lines
            # Reinsert last two lines.
            pypeit_lines += ["data end\n", "\n"]
            self._pypeit_file = pypeit_lines
        else:
            raise ValueError("pypeit_file has not yet been read.")

    def _output_dict(self):
        output_dict = super()._output_dict()
        output_dict.update({"binning": self.binning,
                            "decker": self.decker})
        return output_dict

    def proc_2_pypeit_setup(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do PypeIt setup?", stage='2-pypeit_setup'):
            pass

    def proc_3_pypeit_run(self, no_query: bool = False, do_not_reuse_masters=False, **kwargs):
        if no_query or self.query_stage("Run PypeIt?", stage='3-pypeit_run'):
            pass

    def proc_5_pypeit_coadd(self, no_query: bool = False, **kwargs):
        pass

    def proc_6_convert_to_marz_format(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Convert co-added 1D spectra to Marz format?", stage='6-marz-format'):
            pass

    def _path_2_pypeit(self):
        if self.data_path is not None and "pypeit_dir" not in self.paths:
            self.paths["pypeit_dir"] = os.path.join(self.data_path, epoch_stage_dirs["2-pypeit"])

    def find_science_attributes(self):
        frames = self.get_frames_science()
        if frames:
            frame = frames[0]
            self.set_binning(frame.binning)
            self.set_decker(frame.decker)
        else:
            raise ValueError(f"Science frames list is empty.")
        return frame

    def find_std_attributes(self):
        frames = self.get_frames_standard()
        if frames:
            frame = frames[0]
            self.set_binning_std(frame.binning)
            self.set_decker_std(frame.decker)
        else:
            raise ValueError(f"Standard frames list is empty.")
        return frame

    def get_frames_science(self):
        return self.frames_science

    def get_frames_standard(self):
        return self.frames_standard

    def get_decker(self):
        return self.decker

    def set_decker(self, decker: str):
        self.decker = decker
        return decker

    def get_decker_std(self):
        return self.decker_std

    def set_decker_std(self, decker: str):
        self.decker_std = decker
        return decker

    def _set_pypeit_file(self, lines: list):
        self._pypeit_file = lines

    def _get_pypeit_file(self):
        return self._pypeit_file

    def _set_pypeit_coadd1d_file(self, lines: list):
        self._pypeit_coadd1d_file = lines

    def _get_pypeit_coadd1d_file(self):
        return self._pypeit_coadd1d_file

    def _get_pypeit_sorted_file(self):
        return self._pypeit_sorted_file

    @classmethod
    def stages(cls):
        param_dict = super().stages()
        param_dict.update({"2-pypeit_setup": None,
                           "3-pypeit_run": None,
                           "4-pypeit_flux_calib": None})
        return param_dict

    @classmethod
    def select_child_class(cls, instrument: str):
        instrument = instrument.lower()
        if instrument == "vlt-fors2":
            return FORS2SpectroscopyEpoch
        elif instrument == "vlt-xshooter":
            return XShooterSpectroscopyEpoch
        elif instrument in instruments_spectroscopy:
            return SpectroscopyEpoch
        else:
            raise ValueError(f"Unrecognised instrument {instrument}")

    @classmethod
    def from_file(cls, param_file: Union[str, dict], field: Field = None):
        name, param_file, param_dict = p.params_init(param_file)
        if param_dict is None:
            raise FileNotFoundError(f"No parameter file found at {param_file}.")
        instrument = param_dict["instrument"].lower()
        if field is None:
            field = param_dict["field"]
        if 'target' in param_dict:
            target = param_dict['target']
        else:
            target = None
        sub_cls = cls.select_child_class(instrument=instrument)
        # if sub_cls is SpectroscopyEpoch:
        return sub_cls(name=name,
                       field=field,
                       param_path=param_file,
                       data_path=os.path.join(config["top_data_dir"], param_dict['data_path']),
                       instrument=instrument,
                       date=param_dict["date"],
                       program_id=param_dict["program_id"],
                       target=target
                       )
        # else:
        # return sub_cls.from_file(param_file=param_file, field=field)

    @classmethod
    def from_params(cls, name, field: Union[Field, str] = None, instrument: str = None):
        instrument = instrument.lower()
        field_name, field = cls._from_params_setup(name=name, field=field)
        path = cls.build_param_path(field_name=field_name,
                                    instrument_name=instrument,
                                    epoch_name=name)
        return cls.from_file(param_file=path, field=field)

    @classmethod
    def build_param_path(cls, field_name: str, instrument_name: str, epoch_name: str):
        return os.path.join(p.param_dir, "fields", field_name, "spectroscopy", instrument_name, epoch_name)


class ESOSpectroscopyEpoch(SpectroscopyEpoch):
    def __init__(self,
                 param_path: str = None,
                 name: str = None,
                 field: Union[str, Field] = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 program_id: str = None,
                 grism: str = None
                 ):
        super().__init__(param_path=param_path,
                         name=name,
                         field=field,
                         data_path=data_path,
                         instrument=instrument,
                         date=date,
                         program_id=program_id,
                         grism=grism)
        # Data reduction paths

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)
        if "do_not_reuse_masters" in kwargs:
            do_not_reuse_masters = kwargs["do_not_reuse_masters"]
        else:
            do_not_reuse_masters = False
        self.proc_0_raw()
        self.proc_1_initial_setup()
        self.proc_2_pypeit_setup()
        self.proc_3_pypeit_run(do_not_reuse_masters=do_not_reuse_masters)
        self.proc_4_pypeit_flux()
        self.proc_5_pypeit_coadd()
        self.proc_6_convert_to_marz_format()

    def proc_0_raw(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Download raw data from ESO archive?", stage='0-download'):
            self._path_0_raw()
            r = self.retrieve()
            if r:
                self.stages_complete['0-download'] = Time.now()
                self.update_output_file()

    def _initial_setup(self):
        self._path_0_raw()
        m_path = os.path.join(self.paths["raw_dir"], "M")
        u.mkdir_check(m_path)
        os.system(f"mv {os.path.join(self.paths['raw_dir'], 'M.')}* {m_path}")
        image.fits_table_all(input_path=self.paths["raw_dir"],
                             output_path=os.path.join(self.data_path, f"{self.name}_fits_table_science.csv"))
        image.fits_table_all(input_path=self.paths["raw_dir"],
                             output_path=os.path.join(self.data_path, f"{self.name}_fits_table_all.csv"),
                             science_only=False)

    def retrieve(self):
        """
        Check ESO archive for the epoch raw frames, and download those frames and associated files.
        :return:
        """
        r = []
        if "raw_dir" in self.paths:
            r = _retrieve_eso_epoch(self, path=self.paths["raw_dir"])
        else:
            warnings.warn("raw_dir has not been set. Retrieve could not be run.")
        return r

    @classmethod
    def stages(cls):
        param_dict = super().stages()
        param_dict.update({"0-download": None})
        return param_dict


class FORS2SpectroscopyEpoch(ESOSpectroscopyEpoch):
    _instrument_pypeit = "vlt_fors2"
    grisms = {
        "GRIS_300I": {
            "lambda_min": 6000 * units.angstrom,
            "lambda_max": 11000 * units.angstrom
        }}

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)

    def proc_2_pypeit_setup(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do PypeIt setup?", stage='2-pypeit_setup'):
            self._path_2_pypeit()
            setup_files = os.path.join(self.paths["pypeit_dir"], 'setup_files', '')
            self.paths["pypeit_setup_dir"] = setup_files
            os.system(f"rm {setup_files}*")
            # Generate .sorted file and others
            spec.pypeit_setup(root=self.paths['raw_dir'], output_path=self.paths['pypeit_dir'],
                              spectrograph=self._instrument_pypeit)
            # Generate files to use for run. Set cfg_split to "A" because that corresponds to Chip 1, which is the only
            # one we need to worry about.
            spec.pypeit_setup(root=self.paths['raw_dir'], output_path=self.paths['pypeit_dir'],
                              spectrograph=self._instrument_pypeit, cfg_split="A")
            # Read .sorted file
            self.read_pypeit_sorted_file()
            # Retrieve bias files from .sorted file.
            bias_lines = list(filter(lambda s: "bias" in s and "CHIP1" in s, self._pypeit_sorted_file))
            # Find line containing information for standard observation.
            std_line = filter(lambda s: "standard" in s and "CHIP1" in s, self._pypeit_sorted_file).__next__()
            std_raw = image.SpecRaw.from_pypeit_line(std_line, pypeit_raw_path=self.paths['raw_dir'])
            self.standards_raw.append(std_raw)
            std_start_index = self._pypeit_sorted_file.index(std_line)
            # Find last line of the std-obs configuration (encapsulating the required calibration files)
            std_end_index = self._pypeit_sorted_file[std_start_index:].index(
                "##########################################################\n") + std_start_index
            std_lines = self._pypeit_sorted_file[std_start_index:std_end_index]
            # Read in .pypeit file
            self.read_pypeit_file(setup="A")
            # Add lines to set slit prediction to "nearest" in .pypeit file.
            self.add_pypeit_user_param(param=["calibrations", "slitedges", "sync_predict"], value="nearest")
            # Insert bias lines from .sorted file
            self.add_pypeit_file_lines(lines=bias_lines + std_lines)
            # Write modified .pypeit file back to disk.
            self.write_pypeit_file_science()

            self.stages_complete['2-pypeit_setup'] = Time.now()
            self.update_output_file()

    def proc_3_pypeit_run(self, no_query: bool = False, do_not_reuse_masters: bool = False, **kwargs):
        if no_query or self.query_stage("Run PypeIt?", stage='3-pypeit_run'):
            spec.run_pypeit(pypeit_file=self.paths['pypeit_file'],
                            redux_path=self.paths['pypeit_run_dir'],
                            do_not_reuse_masters=do_not_reuse_masters)
            self.stages_complete['3-pypeit_run'] = Time.now()
            self.update_output_file()

    def proc_5_pypeit_coadd(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage(
                "Do coaddition with PypeIt?\nYou should first inspect the 2D spectra to determine which objects to co-add.",
                stage='5-pypeit_coadd'):
            for file in filter(lambda f: "spec1d" in f, os.listdir(self.paths["pypeit_science_dir"])):
                path = os.path.join(self.paths["pypeit_science_dir"], file)
                os.system(f"pypeit_show_1dspec {path}")


class XShooterSpectroscopyEpoch(ESOSpectroscopyEpoch):
    _instrument_pypeit = "vlt_xshooter"
    grisms = {'uvb': {"lambda_min": 300 * units.nm,
                      "lambda_max": 550 * units.nm},
              "vis": {"lambda_min": 550 * units.nm,
                      "lambda_max": 1000 * units.nm},
              "nir": {"lambda_min": 1000 * units.nm,
                      "lambda_max": 2500 * units.nm}}

    def __init__(self,
                 param_path: str = None,
                 name: str = None,
                 field: Union[str, Field] = None,
                 data_path: str = None,
                 instrument: str = None,
                 date: Union[str, Time] = None,
                 program_id: str = None,
                 ):

        super().__init__(param_path=param_path,
                         name=name,
                         field=field,
                         data_path=data_path,
                         instrument=instrument,
                         date=date,
                         program_id=program_id
                         )

        self.frames_raw = {"uvb": [],
                           "vis": [],
                           "nir": []}
        self.frames_bias = {"uvb": [],
                            "vis": [],
                            "nir": []}
        self.frames_standard = {"uvb": [],
                                "vis": [],
                                "nir": []}
        self.frames_science = {"uvb": [],
                               "vis": [],
                               "nir": []}
        self.frames_dark = {"uvb": [],
                            "vis": [],
                            "nir": []}
        self._pypeit_file = {"uvb": None,
                             "vis": None,
                             "nir": None}
        self._pypeit_file_std = {"uvb": None,
                                 "vis": None,
                                 "nir": None}
        self._pypeit_sorted_file = {"uvb": None,
                                    "vis": None,
                                    "nir": None}
        self._pypeit_coadd1d_file = {"uvb": None,
                                     "vis": None,
                                     "nir": None}
        self._pypeit_user_param_start = {"uvb": None,
                                         "vis": None,
                                         "nir": None}
        self._pypeit_user_param_end = {"uvb": None,
                                       "vis": None,
                                       "nir": None}

        self.binning = {"uvb": None,
                        "vis": None,
                        "nir": None}
        self.binning_std = {"uvb": None,
                            "vis": None,
                            "nir": None}
        self.decker = {"uvb": None,
                       "vis": None,
                       "nir": None}
        self.decker_std = {"uvb": None,
                           "vis": None,
                           "nir": None}

        self._cfg_split_letters = {"uvb": None,
                                   "vis": None,
                                   "nir": None}

        self.load_output_file()
        self._current_arm = None

    def pipeline(self, **kwargs):
        super().pipeline(**kwargs)
        # self.proc_5_pypeit_coadd()

    def proc_2_pypeit_setup(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Do PypeIt setup?", stage='2-pypeit_setup'):
            self._path_2_pypeit()
            setup_files = os.path.join(self.paths["pypeit_dir"], 'setup_files', '')
            self.paths["pypeit_setup_dir"] = setup_files
            os.system(f"rm {setup_files}*")
            for arm in self.grisms:
                self._current_arm = arm
                spec.pypeit_setup(root=self.paths['raw_dir'], output_path=self.paths['pypeit_dir'],
                                  spectrograph=f"{self._instrument_pypeit}_{arm}")
                # Read .sorted file
                self.read_pypeit_sorted_file()
                setup = self._cfg_split_letters[arm]
                spec.pypeit_setup(root=self.paths['raw_dir'], output_path=self.paths['pypeit_dir'],
                                  spectrograph=f"{self._instrument_pypeit}_{arm}", cfg_split=setup)
                # Retrieve text from .pypeit file
                self.read_pypeit_file(setup=setup)
                # Add parameter to use dark frames for NIR reduction.
                if arm == "nir":
                    self.add_pypeit_user_param(param=["calibrations", "pixelflatframe", "process", "use_darkimage"],
                                               value="True")
                    self.add_pypeit_user_param(param=["calibrations", "illumflatframe", "process", "use_darkimage"],
                                               value="True")
                    self.add_pypeit_user_param(param=["calibrations", "traceframe", "process", "use_darkimage"],
                                               value="True")
                self.find_science_attributes()
                # For X-Shooter, we need to reduce the standards separately due to the habit of observing them with
                # different decker (who knows)
                self.find_std_attributes()
                # Remove incompatible binnings and frametypes
                print(f"\nRemoving incompatible files for {arm} arm:")
                pypeit_file = self._get_pypeit_file()
                # pypeit_file_std = pypeit_file.copy()
                decker = self.get_decker()
                binning = self.get_binning()
                decker_std = self.get_decker_std()
                for raw_frame in self.frames_raw[arm]:
                    # Remove all frames with frame_type "None" from both science and standard lists.
                    if raw_frame.frame_type == "None":
                        pypeit_file.remove(raw_frame.pypeit_line)
                        # pypeit_file_std.remove(raw_frame.pypeit_line)
                    else:
                        # Remove files with incompatible binnings from science reduction list.
                        if raw_frame.binning != binning \
                                or raw_frame.decker not in ["Pin_row", decker, decker_std]:
                            pypeit_file.remove(raw_frame.pypeit_line)
                    # Special behaviour for NIR arm
                    if arm == "nir":
                        # For the NIR arm, PypeIt only works if you use the Science frames for the arc, tilt calib.
                        if raw_frame.frame_type in ["arc,tilt", "tilt,arc"]:
                            pypeit_file.remove(raw_frame.pypeit_line)
                            # pypeit_file_std.remove(raw_frame.pypeit_line)
                        elif raw_frame.frame_type == "science":
                            raw_frame.frame_type = "science,arc,tilt"
                            # Find original line in PypeIt file
                            to_replace = pypeit_file.index(raw_frame.pypeit_line)
                            # Rewrite pypeit line.
                            raw_frame.pypeit_line = raw_frame.pypeit_line.replace("science", "science,arc,tilt")
                            pypeit_file[to_replace] = raw_frame.pypeit_line
                        elif raw_frame.frame_type == "standard":
                            raw_frame.frame_type = "standard,arc,tilt"
                            # Find original line in PypeIt file
                            to_replace = pypeit_file.index(raw_frame.pypeit_line)
                            # Rewrite pypeit line.
                            raw_frame.pypeit_line = raw_frame.pypeit_line.replace("standard", "standard,arc,tilt")
                            pypeit_file[to_replace] = raw_frame.pypeit_line
                self._set_pypeit_file(pypeit_file)
                # self._set_pypeit_file_std(pypeit_file_std)
                self.write_pypeit_file_science()
                # std_path = os.path.join(self.paths["pypeit_dir"], self.get_path("pypeit_run_dir"), "Flux_Standards")
                # u.mkdir_check(std_path)
                # self.set_path("pypeit_dir_std", std_path)
                # self.set_path("pypeit_file_std",
                #              os.path.join(self.get_path("pypeit_run_dir"), f"vlt_xshooter_{arm}_std.pypeit"))
                # self.write_pypeit_file_std()
            self._current_arm = None
            self.stages_complete['2-pypeit_setup'] = Time.now()
            self.update_output_file()

    def proc_3_pypeit_run(self, no_query: bool = False, do_not_reuse_masters: bool = False, **kwargs):
        for i, arm in enumerate(self.grisms):
            # UVB not yet implemented in PypeIt, so we skip.
            if arm == "uvb":
                continue
            self._current_arm = arm
            if no_query or self.query_stage(f"Run PypeIt for {arm.upper()} arm?", stage=f'3.{i + 1}-pypeit_run_{arm}'):
                spec.run_pypeit(pypeit_file=self.get_path('pypeit_file'),
                                redux_path=self.get_path('pypeit_run_dir'),
                                do_not_reuse_masters=do_not_reuse_masters)
                self.stages_complete[f'3.{i + 1}-pypeit_run_{arm}'] = Time.now()
                self.update_output_file()
            # if arm != "nir" and self.query_stage(f"Run PypeIt on flux standards for {arm.upper()} arm?",
            #                                      stage=f'3.{i + 1}-pypeit_run_{arm}_std'):
            #     print(self.get_path('pypeit_file_std'))
            #     spec.run_pypeit(pypeit_file=self.get_path('pypeit_file_std'),
            #                     redux_path=self.get_path('pypeit_dir_std'),
            #                     do_not_reuse_masters=do_not_reuse_masters)
            #     self.stages_complete[f'3.{i + 1}-pypeit_run_{arm}'] = Time.now()
            #     self.update_output_file()
        self._current_arm = None

    def proc_4_pypeit_flux(self, no_query: bool = False, **kwargs):
        for i, arm in enumerate(self.grisms):
            # UVB not yet implemented in PypeIt, so we skip.
            if arm == "uvb":
                continue
            self._current_arm = arm
            if no_query or self.query_stage(f"Do PypeIt fluxing for {arm.upper()} arm?",
                                            stage=f'4.{i + 1}-pypeit_flux_calib_{arm}'):
                self._current_arm = arm
                self._pypeit_flux()
            self.stages_complete[f'4.{i + 1}-pypeit_flux_calib_{arm}'] = Time.now()
        self._current_arm = None
        self.update_output_file()

    def proc_5_pypeit_coadd(self, no_query: bool = False, **kwargs):
        for i, arm in enumerate(self.grisms):
            # UVB not yet implemented in PypeIt, so we skip.
            if arm == "uvb":
                continue
            self._current_arm = arm
            if no_query or self.query_stage(f"Do PypeIt coaddition for {arm.upper()} arm?",
                                            stage=f'5.{i + 1}-pypeit_coadd_{arm}'):
                run_dir = self.get_path("pypeit_run_dir")
                coadd_file_path = os.path.join(run_dir, f"{self._instrument_pypeit}_{arm}.coadd1d")
                self.set_path("pypeit_coadd1d_file", coadd_file_path)
                with open(coadd_file_path) as file:
                    coadd_file_lines = file.readlines()
                output_path = os.path.join(run_dir, f"{self.name}_{arm}_coadded.fits")
                sensfunc_path = self.get_path("pypeit_sensitivity_file")
                # Remove non-science files
                for line in coadd_file_lines[coadd_file_lines.index("coadd1d read\n"):]:
                    if "STD,FLUX" in line or "STD,TELLURIC" in line:
                        coadd_file_lines.remove(line)

                self._set_pypeit_coadd1d_file(coadd_file_lines)
                # Re-insert parameter lines
                self.add_pypeit_user_param(param=["coadd1d", "coaddfile"], value=output_path, file_type="coadd1d")
                self.add_pypeit_user_param(param=["coadd1d", "sensfuncfile"], value=sensfunc_path, file_type="coadd1d")
                self.add_pypeit_user_param(param=["coadd1d", "wave_method"], value="velocity", file_type="coadd1d")
                u.write_list_to_file(coadd_file_path, self._get_pypeit_coadd1d_file())
                spec.pypeit_coadd_1dspec(coadd1d_file=coadd_file_path)
                self.add_coadded_image(coadd_file_path, key=arm)

            self.stages_complete[f'5.{i + 1}-pypeit_coadd_{arm}'] = Time.now()

        self._current_arm = None

    def proc_6_convert_to_marz_format(self, no_query: bool = False, **kwargs):
        if no_query or self.query_stage("Convert co-added 1D spectra to Marz format?",
                                        stage='6-convert_to_marz_format'):
            for arm in self.coadded:
                self.coadded[arm].convert_to_marz_format()
            self.stages_complete[f'6-convert_to_marz_format'] = Time.now()

    def add_coadded_image(self, img: Union[str, image.Spec1DCoadded], **kwargs):
        arm = kwargs["key"]
        if isinstance(img, str):
            img = image.Spec1DCoadded(path=img, grism=arm)
        img.epoch = self
        self.coadded[arm] = img
        return img

    def add_frame_raw(self, raw_frame: image.Image):
        arm = self._get_current_arm()
        self.frames_raw[arm].append(raw_frame)
        self.sort_frame(raw_frame)

    def sort_frame(self, frame: image.Image):
        arm = self._get_current_arm()
        if frame.frame_type == "bias":
            self.frames_bias[arm].append(frame)
        elif frame.frame_type == "science":
            self.frames_science[arm].append(frame)
        elif frame.frame_type == "standard":
            self.frames_standard[arm].append(frame)
        elif frame.frame_type == "dark":
            self.frames_dark[arm].append(frame)

    def read_pypeit_sorted_file(self):
        arm = self._get_current_arm()
        if "pypeit_setup_dir" in self.paths and self.paths["pypeit_setup_dir"] is not None:
            setup_files = self.paths["pypeit_setup_dir"]
            sorted_path = os.path.join(setup_files,
                                       filter(lambda f: f"vlt_xshooter_{arm}" in f and f.endswith(".sorted"),
                                              os.listdir(setup_files)).__next__())
            with open(sorted_path) as sorted_file:
                file = sorted_file.readlines()
            self._pypeit_sorted_file[arm] = file
            for setup in ["A", "B", "C"]:
                info = self.setup_info(setup=setup)
                arm_this = info["arm"].lower()
                self._cfg_split_letters[arm_this] = setup
        else:
            raise KeyError("pypeit_setup_dir has not been set.")

    def read_pypeit_file(self, setup: str):
        if "pypeit_dir" in self.paths and self.paths["pypeit_dir"] is not None:
            arm = self._get_current_arm()
            filename = f"{self._instrument_pypeit}_{arm}_{setup}"
            self._read_pypeit_file(filename=filename)
            return self._pypeit_file[arm]
        else:
            raise KeyError("pypeit_run_dir has not been set.")

    def pypeit_flux_title(self):
        return f"{self._instrument_pypeit}_{self._get_current_arm()}.flux"

    def get_path(self, key):
        key = self._get_key_arm(key)
        return self.paths[key]

    def set_path(self, key: str, value: str):
        key = self._get_key_arm(key)
        self.paths[key] = value

    def get_frames_science(self):
        return self.frames_science[self._get_current_arm()]

    def get_frames_standard(self):
        return self.frames_standard[self._get_current_arm()]

    def get_binning(self):
        return self.binning[self._get_current_arm()]

    def set_binning(self, binning: str):
        self.binning[self._get_current_arm()] = binning
        return binning

    def get_binning_std(self):
        return self.binning_std[self._get_current_arm()]

    def set_binning_std(self, binning: str):
        self.binning_std[self._get_current_arm()] = binning
        return binning

    def get_decker(self):
        return self.decker[self._get_current_arm()]

    def set_decker(self, decker: str):
        self.decker[self._get_current_arm()] = decker
        return decker

    def get_decker_std(self):
        return self.decker_std[self._get_current_arm()]

    def set_decker_std(self, decker: str):
        self.decker_std[self._get_current_arm()] = decker
        return decker

    def _get_current_arm(self):
        if self._current_arm is not None:
            return self._current_arm
        else:
            raise ValueError("self._current_arm is not set (no arm currently active).")

    def _set_pypeit_file(self, lines: list):
        self._pypeit_file[self._get_current_arm()] = lines

    def _get_pypeit_file(self):
        return self._pypeit_file[self._get_current_arm()]

    def _get_pypeit_sorted_file(self):
        return self._pypeit_sorted_file[self._get_current_arm()]

    def _set_pypeit_file_std(self, lines: list):
        self._pypeit_file_std[self._get_current_arm()] = lines

    def _get_pypeit_file_std(self):
        return self._pypeit_file_std[self._get_current_arm()]

    def _set_pypeit_coadd1d_file(self, lines: list):
        self._pypeit_coadd1d_file[self._get_current_arm()] = lines

    def _get_pypeit_coadd1d_file(self):
        return self._pypeit_coadd1d_file[self._get_current_arm()]

    def _get_key_arm(self, key):
        arm = self._get_current_arm()
        key = f"{arm}_{key}"
        return key

    @classmethod
    def stages(cls):
        param_dict = super().stages()
        param_dict.update({"2-pypeit_setup": None,
                           "3.1-pypeit_run_uvb": None,
                           "3.2-pypeit_run_vis": None,
                           "3.3-pypeit_run_nir": None,
                           "4.1-pypeit_flux_calib_uvb": None,
                           "4.2-pypeit_flux_calib_vis": None,
                           "4.3-pypeit_flux_calib_nir": None,
                           "5.1-pypeit_coadd_uvb": None,
                           "5.2-pypeit_coadd_vis": None,
                           "5.3-pypeit_coadd_nir": None,
                           "6-convert_to_marz_format": None
                           })
        return param_dict

    def _output_dict(self):
        return {
            "stages": self.stages_complete,
            "paths": self.paths,
        }

    def load_output_file(self, **kwargs):
        outputs = super().load_output_file(mode="spectroscopy", **kwargs)
        if outputs not in [None, True, False]:
            self.stages_complete.update(outputs["stages"])
            if "paths" in outputs:
                self.paths.update(outputs[f"paths"])
        return outputs

    def write_pypeit_file_std(self):
        """
        Rewrites the stored .pypeit file to disk.
        :return: path of .pypeit file.
        """
        pypeit_lines = self._get_pypeit_file_std()
        if pypeit_lines is not None:
            pypeit_file_path = os.path.join(self.get_path("pypeit_file_std"), )
            u.write_list_to_file(path=pypeit_file_path, file=pypeit_lines)
        else:
            raise ValueError("pypeit_file_std has not yet been read.")
        return pypeit_file_path

# def test_frbfield_from_params():
#     frb_field = FRBField.from_file("FRB181112")
#     assert frb_field.frb.position_err.a_stat ==
