# Code by Lachlan Marnoch, 2020

from astropy.coordinates import SkyCoord
from astropy import units

import craftutils.params as p


def main(frb: str, new: bool):
    data_dir = p.config['top_data_dir']
    ra = dec = None
    if new:
        p.add_frb_param(obj=frb, params={"data_dir": data_dir + frb + "/"})
        print()
        print(
            f"Enter the FRB coordinates. These can be changed later by editing the file {p.param_path}FRBs/{frb}.yaml")
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
    else:
        params = p.object_params_frb(obj=frb)

        na = [0.0, None]

        if params["burst_ra"] in na and params["burst_dec"] in na and params["burst_ra_str"] is not None and params[
            "burst_dec_str"] is not None:
            ra = params["burst_ra_str"]
            dec = params["burst_dec_str"]

    coords = SkyCoord(ra, dec)
    ra = float(coords.ra.value)
    dec = float(coords.dec.value)
    ra_str = str(coords.ra.to_string())
    dec_str = str(coords.dec.to_string())
    p.add_frb_param(obj=frb,
                    params={"burst_ra": ra, "burst_dec": dec, "burst_ra_str": ra_str, "burst_dec_str": dec_str})


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Set up a new epoch.")

    parser.add_argument('--op', help='Name of frb, eg FRB180924')
    parser.add_argument('--new', action='store_true', help='Name of frb, eg FRB180924')

    # Load arguments

    args = parser.parse_args()

    main(frb=args.op, new=args.new)
