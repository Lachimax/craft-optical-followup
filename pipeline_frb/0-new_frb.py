# Code by Lachlan Marnoch, 2020

from astropy.coordinates import SkyCoord
from astropy import units

import craftutils.params as p


def main(frb: str):
    data_dir = p.config['top_data_dir']
    p.add_frb_param(obj=frb, params={"data_dir": data_dir + frb + "/"})
    print()
    print(f"Enter the FRB coordinates. These can be changed later by editing the file {p.param_path}FRBs/{frb}.yaml")
    ra = input(f"Please enter the right ascension of {frb}:")
    try:
        ra = float(ra)
        ra = units.Quantity(ra, unit="deg")
    except ValueError:
        print()
    dec = input(f"Please enter the declination of {frb}:")
    try:
        dec = float(dec)
        dec = units.Quantity(dec, unit="deg")
    except ValueError:
        print()
    coords = SkyCoord(ra, dec)
    ra = coords.ra.value
    dec = coords.dec.value
    p.add_frb_param(obj=frb, params={"burst_ra": ra, "burst_dec": dec})


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Set up a new epoch.")
    parser.add_argument('--op', help='Name of frb, eg FRB180924')

    # Load arguments

    args = parser.parse_args()

    main(frb=args.op)
