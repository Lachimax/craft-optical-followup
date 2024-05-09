#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-

# Code by Lachlan Marnoch, 2021
import os

import craftutils.observation.field as fld
import craftutils.observation.epoch as ep
import craftutils.params as p
import craftutils.utils as u

config = p.config


def main(
        field_name: str,
        epoch_name: str,
        mode: str,
        instrument: str,
        furby_path: str,
        do: str,
        debug_level: int,
        skip_cats: bool
):
    if not p.param_dir:
        param_dir = u.user_input(
            message="Please enter a path to use as the param directory. This will be created if it does not exist.",
            input_type=str,
            default=p.param_dir_project
        )
        p.set_param_dir(param_dir, write=True)

    if not p.data_dir:
        param_dir = u.user_input(
            message="Please enter a path to use as the top-level data directory. This will be created if it does "
                    "not exist. It should be on a drive with plenty of space!",
            input_type=str,
            default=f"{os.path.expanduser('~')}/data"
        )
        p.set_data_dir(param_dir, write=True)

    if not config["table_dir"]:
        table_dir = u.user_input(
            message="Please enter a directory in which to write tables describing observations etc. This will be created if it does "
                    "not exist.",
            input_type=str,
            default=os.path.join(p.data_dir, "tables")
        )
        p.set_table_dir(table_dir, write=True)

    u.debug_level = debug_level

    new_field = False

    directory = ep.load_epoch_directory()

    if epoch_name is not None:
        print(f"Looking for {epoch_name} in directory...")
        if epoch_name in directory:
            epoch_dict = directory[epoch_name]
            field_name = epoch_dict["field_name"]
            instrument = epoch_dict["instrument"]
            mode = epoch_dict["mode"]
        else:
            print(f"{epoch_name} not found.")

    # Do automated FURBY process.
    if furby_path is not None:

        imaging = True

        healpix_path = furby_path.replace(".json", "_hp.fits")
        if not os.path.isfile(healpix_path):
            healpix_path = None

        params = fld.FRBField.param_from_furby_json(
            json_path=furby_path,
            healpix_path=healpix_path,
        )
        field_name = params["name"]
        field = fld.Field.from_params(name=field_name)

        instrument = "vlt-fors2"

        epoch_name = f"{field_name}_FORS2_1"
        ep.FORS2ImagingEpoch.new_yaml(
            name=epoch_name,
            path=ep.FORS2ImagingEpoch.build_param_path(
                instrument_name=instrument,
                field_name=field_name,
                epoch_name=epoch_name
            ),
            field=field.name,
            instrument=instrument,
            data_path=os.path.join(field_name, "imaging", instrument, epoch_name, "")
        )

    else:
        if field_name is None:
            fields = ["New field"]
            fields += fld.list_fields()
            opt, field_name = u.select_option("No field specified. Please select one:", options=fields, sort=False)
            if opt == 0:
                new_field = True
                field_name = input("Please enter the name of the new field:\n")
        # Check for field param file
        if not new_field:
            field = fld.Field.from_params(name=field_name)
        else:
            field = None
        # If this field has no parameter file, ask to create one.
        if field is None:
            param_path = os.path.join(p.param_dir, "fields", "")
            print()
            field_param_path = os.path.join(param_path, field_name)
            u.mkdir_check(field_param_path)
            field_param_path_yaml = os.path.join(field_param_path, f"{field_name}.yaml")
            if not new_field:
                print(f"{field_name} not found in the param directory.")
            if u.select_yn(f"Create a new param file at '{field_param_path_yaml}'?"):
                fld.Field.new_params_from_input(field_name=field_name, field_param_path=field_param_path)
            else:
                print("Exiting.")
                exit(0)
            field = fld.Field.from_params(name=field_name)

    if mode is None:
        _, mode = u.select_option(
            message="Please select a mode.",
            options=["Imaging", "Spectroscopy", "Objects"],
            include_exit=True
        )

    mode = mode.lower()

    if mode == "objects":
        field.pipeline()
        exit()
    else:
        if epoch_name is None:
            # Build a list of relevant epochs from that field.
            if mode == "imaging":
                field.gather_epochs_imaging(instrument=instrument)
            elif mode == "spectroscopy":
                field.gather_epochs_spectroscopy(instrument=instrument)
            else:
                raise ValueError(f"mode must be objects, imaging or spectroscopy; received {mode}.")
            # Let the user select an epoch.
            epoch = field.select_epoch(instrument=instrument, mode=mode)
        else:
            if instrument is None:
                instrument = fld.select_instrument(mode="imaging")
            epoch = ep.ImagingEpoch.from_params(epoch_name, instrument=instrument, field=field)
            epoch.field = field

    u.debug_print(2, "pipeline.py: type(epoch) ==", type(epoch))
    epoch.do = do
    epoch.pipeline(skip_cats=skip_cats)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="General pipeline for optical/infrared data reduction."
    )
    parser.add_argument(
        "--field",
        help='Name of field, eg FRB20180924',
        type=str,
        default=None)
    parser.add_argument(
        "--epoch",
        help='Name of epoch, eg FRB20181112_1',
        type=str,
        default=None)
    parser.add_argument(
        "--instrument",
        help="Name of instrument on which epoch was observed, eg 'vlt-fors2'",
        type=str,
        default=None)
    parser.add_argument(
        "--mode",
        help="imaging, spectroscopy or objects",
        default=None,
    )
    parser.add_argument(
        "--do",
        help="Epoch processing stages to perform (overrides manual selection if provided). "
             "Numbers separated by space or comma.",
        type=str
    )
    parser.add_argument(
        "-d",
        help="Debug level for verbosity purposes.",
        type=int,
        default=0
    )
    parser.add_argument(
        "--furby_path",
        help="Path to FURBY json file. If specified, will create a new FRB param file & FORS2 epoch param file, and use"
             "those for FORS2 data retrieval and reduction. Overrides --furby, --epoch, --field, -i and -s.",
        type=str
    )
    parser.add_argument(
        "--furby",
        help="FURBY (TNS) name; if this is specified, will look for the FURBY .json file in furby_dir."
             "Will create a new FRB param file & FORS2 epoch param file, and use"
             "those for FORS2 data retrieval and reduction. Overrides --epoch, --field, -i and -s.",
        type=str
    )
    parser.add_argument(
        "--skip_cats",
        help="Skips initial catalogue retrieval.",
        action="store_true"
    )

    # Load arguments

    args = parser.parse_args()

    furby = args.furby
    fp = args.furby_path
    if furby is not None and fp is None:
        fp = os.path.join(p.config["furby_dir"], "craco_fu", "data", furby, f"{furby}.json")

    main(
        field_name=args.field,
        epoch_name=args.epoch,
        mode=args.mode,
        instrument=args.instrument,
        do=args.do,
        debug_level=args.d,
        furby_path=fp,
        skip_cats=args.skip_cats
    )


if __name__ == "__main__":
    parse_args()
