# Code by Lachlan Marnoch, 2021

import craftutils.astronobjects as objects
import craftutils.observation as obs
import craftutils.params as p

config = p.config


def main(field):

    field = obs.Field.from_params(name=field)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Test general pipeline.")
    parser.add_argument("--field", help='Name of field, eg FRB180924', type=str)

    # Load arguments

    args = parser.parse_args()

    main(field=args.field)
