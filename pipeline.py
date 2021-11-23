# Code by Lachlan Marnoch, 2021
import json
import os

import craftutils.observation.objects as obj
import craftutils.observation.field as fld
import craftutils.params as p
import craftutils.utils as u

config = p.config


def main(
        field_name: str,
        epoch_name: str,
        imaging: bool,
        spectroscopy: bool,
        instrument: str,
        furby_path: str,
        furby_test: bool,
        do: str,
        do_not_reuse_masters: bool,
        overwrite_download: bool,
        distance_tolerance: float,
        snr_tolerance: float,
        class_star_tolerance: float,
        debug_level: int,
):
    u.debug_level = debug_level

    new_field = False

    # Do automated FURBY process.
    if furby_path is not None:

        new_field = True
        furby = True

        healpix_path = furby_path.replace(".json", "_hp.fits")
        if not os.path.isfile(healpix_path):
            healpix_path = None

        fld.FRBField.param_from_furby_json(
            json_path=furby_path,
            healpix_path=healpix_path,
        )
        field = fld.Field.from_params(name=field_name)

        instrument = "vlt-fors2"

        epoch_name = f"{field_name}_FORS2_1"
        fld.FORS2ImagingEpoch.new_yaml(
            name=epoch_name,
            path=fld.FORS2ImagingEpoch.build_param_path(
                instrument_name=instrument,
                field_name=field_name,
                epoch_name=epoch_name
            ),

        )
        epoch = fld.FORS2ImagingEpoch.from_params(
            name=epoch_name,
            instrument=instrument,
            field=field,
            old_format=False,
        )

    else:
        if field_name is None:
            fields = ["New field"]
            fields += fld.list_fields()
            old_fields = fld.list_fields_old()
            for old_field in old_fields:
                if old_field not in fields and f"FRB20{old_field[3:]}" not in fields:
                    fields.append(old_field)
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
            # Check for old format param file, and ask to convert if found.
            old_field_name = f"FRB{field_name[-8:]}"
            old_params = p.object_params_frb(obj=old_field_name)
            print()
            field_param_path = os.path.join(param_path, field_name)
            u.mkdir_check(field_param_path)
            field_param_path_yaml = os.path.join(field_param_path, f"{field_name}.yaml")
            if old_params is None:
                if not new_field:
                    print(f"{field_name} not found in the param directory.")
                if u.select_yn(f"Create a new param file at '{field_param_path_yaml}'?"):
                    _, field_class = u.select_option(
                        message="Which type of field would you like to create?",
                        options={"FRB field": fld.FRBField,
                                 "Standard (calibration) field": fld.StandardField,
                                 "Normal field": fld.Field
                                 })
                    field_class.new_yaml(name=field_name, path=field_param_path)
                    print(f"Template parameter file created at '{field_param_path_yaml}'")
                    input("Please edit this file before proceeding, then press Enter to continue.")
                else:
                    print("Exiting.")
                    exit(0)
            else:
                print("Old format param file detected.")
                if u.select_yn("Convert to new format?"):
                    fld.FRBField.convert_old_param(frb=old_field_name)
                else:
                    print("Exiting...")
                    exit(0)
            field = fld.Field.from_params(name=field_name)

        field.retrieve_catalogues()
        if spectroscopy:
            mode = "Spectroscopy"
        elif imaging:
            mode = "Imaging"
        else:
            _, mode = u.select_option(message="Please select a mode.", options=["Imaging", "Spectroscopy"])

        if mode == "Spectroscopy":
            if epoch_name is None:
                # Build a list of imaging epochs from that field.
                field.gather_epochs_spectroscopy()
                # Let the user select an epoch.
                epoch = field.select_epoch_spectroscopy()
            else:
                if instrument is None:
                    instrument = fld.select_instrument(mode="spectroscopy")
                epoch = fld.SpectroscopyEpoch.from_params(epoch_name, instrument=instrument, field=field)

        else:  # if mode == "Imaging"
            if epoch_name is None:
                # Build a list of imaging epochs from that field.
                if type(field) is fld.FRBField:
                    field.gather_epochs_old()
                field.gather_epochs_imaging()
                # Let the user select an epoch.
                epoch = field.select_epoch_imaging()
            else:
                if instrument is None:
                    instrument = fld.select_instrument(mode="imaging")
                epoch = fld.ImagingEpoch.from_params(epoch_name, instrument=instrument, field=field)
                epoch.field = field

    epoch.do = do
    epoch.pipeline(
        do_not_reuse_masters=do_not_reuse_masters,
        overwrite_download=overwrite_download,
        distance_tolerance=distance_tolerance,
        snr_tolerance=snr_tolerance,
        class_star_tolerance=class_star_tolerance
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="General pipeline for optical/infrared data reduction.")
    parser.add_argument(
        "--field", help='Name of field, eg FRB180924', type=str, default=None)
    parser.add_argument(
        "--epoch", help='Name of epoch, eg FRB181112_1', type=str, default=None)
    parser.add_argument(
        "--instrument", help="Name of instrument on which epoch was observed, eg 'vlt-fors2'", type=str,
        default=None)
    parser.add_argument(
        "-i", help="Imaging pipeline", action="store_true")
    parser.add_argument(
        "-s", help="Spectroscopy pipeline. Overrides -i.", action="store_true")
    parser.add_argument(
        "--do", help="Epoch processing stages to perform (overrides manual selection if provided). "
                     "Numbers separated by space or comma.",
        type=str)
    parser.add_argument(
        "--do_not_reuse_masters",
        help="If provided, PypeIt is asked to re-make master frames for every"
             "science reduction (including within a run). DRASTICALLY "
             "increases PypeIt runtime, especially for epochs with many "
             "science frames.",
        action='store_true')
    parser.add_argument(
        "-o",
        help="Overwrite existing files during download.",
        action='store_true')
    parser.add_argument(
        "--distance_tolerance",
        help="Distance tolerance for object-matching.",
        type=float
    )
    parser.add_argument(
        "--snr_tolerance",
        help="Minimum SNR for photometric calibration",
        type=float)
    parser.add_argument(
        "--class_star_tolerance",
        help="Minimum class_star for object inclusion in photometric calibration.",
        type=float)
    parser.add_argument(
        "-d",
        help="Debug level for verbosity purposes.",
        type=int,
        default=0)
    parser.add_argument(
        "--furby",
        help="Path to FURBY json file. If specified, will create a new FRB param file & FORS2 epoch param file, and use"
             "those for FORS2 data retrieval and reduction. Overrides --epoch, --field, -i and -s.",
        type=str
    )
    parser.add_argument(
        "--furby_test",
        help="Puts the automated FURBY pipeline in test mode, retrieving I-band data instead of R-band and using old "
             "programme IDs.",
        action="store_true",
    )

    # Load arguments

    args = parser.parse_args()

    main(
        field_name=args.field,
        epoch_name=args.epoch,
        imaging=args.i,
        spectroscopy=args.s,
        instrument=args.instrument,
        do=args.do,
        do_not_reuse_masters=args.do_not_reuse_masters,
        overwrite_download=args.o,
        distance_tolerance=args.distance_tolerance,
        snr_tolerance=args.snr_tolerance,
        class_star_tolerance=args.class_star_tolerance,
        debug_level=args.d,
        furby_path=args.furby,
        furby_test=args.furby_test
    )
