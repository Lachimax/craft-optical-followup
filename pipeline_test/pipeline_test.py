# Code by Lachlan Marnoch, 2021

import craftutils.astronobjects as objects
import craftutils.observation as obs
import craftutils.params as p
import craftutils.utils as u

config = p.config


def main(field_name):
    print("Refreshing parameter files from templates...")
    p.refresh_params_all(quiet=True)
    field = obs.Field.from_params(name=field_name)
    # If this field has no parameter file, ask to create one.
    if field is None:
        param_path = f"{p.param_path}fields/"
        old_params = p.object_params_frb(obj=field_name)
        print()
        if old_params is None:
            print(f"{field_name} not found in the param directory.")
            if u.select_yn(f"Create a new param file at '{param_path}{field_name}.yaml'?"):
                obs.FRBField.new_yaml(name=field_name, path=param_path)
                print(f"Template parameter file created at '{param_path}{field_name}.yaml'")
                print("Please edit this file before proceeding.")
            else:
                print("Exiting.")
                exit(0)
        else:
            print("Old format param file detected.")
            if u.select_yn("Convert to new format?"):
                obs.FRBField.convert_old_param(frb=field_name)

    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Test general pipeline.")
    parser.add_argument("--field", help='Name of field, eg FRB180924', type=str)

    # Load arguments

    args = parser.parse_args()

    main(field_name=args.field)
