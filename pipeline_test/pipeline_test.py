# Code by Lachlan Marnoch, 2021

import os

import craftutils.observation.objects as obj
import craftutils.observation.field as fld
import craftutils.params as p
import craftutils.utils as u

config = p.config


def main(field_name):
    new_field = False
    print("Refreshing parameter files from templates...")
    p.refresh_params_all(quiet=True)
    if field_name is None:
        fields = ["New field"]
        fields += fld.list_fields()
        old_fields = fld.list_fields_old()
        for old_field in old_fields:
            if old_field not in fields:
                fields.append(old_field)
        opt, field_name = u.select_option("No field specified. Please select one:", options=fields)
        if opt == 0:
            new_field = True
            field_name = input("Please enter the name of the new field:\n")
    # Check for field param file
    print(field_name)
    if not new_field:
        field = fld.Field.from_params(name=field_name)
    else:
        field = None
    # If this field has no parameter file, ask to create one.
    if field is None:
        param_path = os.path.join(p.param_path, "fields", "")
        # Check for old format param file, and ask to convert if found.
        old_params = p.object_params_frb(obj=field_name)
        print()
        field_param_path = os.path.join(param_path, field_name)
        print(field_param_path)
        u.mkdir_check(field_param_path)
        field_param_path_yaml = os.path.join(field_param_path, f"{field_name}.yaml")
        if old_params is None:
            if not new_field:
                print(f"{field_name} not found in the param directory.")
            if u.select_yn(f"Create a new param file at '{field_param_path_yaml}'?"):
                fld.FRBField.new_yaml(name=field_name, path=field_param_path)
                print(f"Template parameter file created at '{field_param_path_yaml}'")
                print("Please edit this file before proceeding.")
            else:
                print("Exiting.")
                exit(0)
        else:
            print("Old format param file detected.")
            if u.select_yn("Convert to new format?"):
                fld.FRBField.convert_old_param(frb=field_name)
            else:
                exit(0)
        field = fld.Field.from_params(name=field_name)
    field.mkdir_params()
    field.mkdir()
    _, mode = u.select_option(message="Please select a mode.", options=["Imaging", "Spectroscopy"])
    if mode == "Spectroscopy":
        field.gather_epochs_spectroscopy()
        print("This doesn't do anything yet.")
        print("Exiting.")
        exit(0)
    else:
        # Build a list of imaging epochs from that field.
        field.gather_epochs_imaging()
        if type(field) is fld.FRBField:
            field.gather_epochs_old()
        epoch = field.select_epoch_imaging()

        # Data retrieval
        if isinstance(epoch, fld.ESOImagingEpoch):
            if epoch.query_stage("Download raw data from ESO archive?", stage='download'):
                epoch.retrieve()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Test general pipeline.")
    parser.add_argument("--field", help='Name of field, eg FRB180924', type=str)

    # Load arguments

    args = parser.parse_args()

    main(field_name=args.field)
