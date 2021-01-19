# Code by Lachlan Marnoch, 2020

import craftutils.params as p


def main(frb: str):
    data_dir = p.config['top_data_dir']
    p.add_frb_param(obj=frb, params={"data_dir": data_dir + frb + "/"})


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Set up a new epoch.")
    parser.add_argument('--op', help='Name of frb, eg FRB180924')

    # Load arguments

    args = parser.parse_args()

    main(frb=args.op)
