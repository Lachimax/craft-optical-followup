import craftutils.retrieve as r
import craftutils.params as p


def main(frb: str):
    r.update_frb_des_photometry(frb=frb)
    r.update_frb_sdss_photometry(frb=frb)
    r.update_frb_skymapper_photometry(frb=frb)
    r.update_frb_irsa_extinction(frb=frb)

    r.update_frb_des_cutout(frb=frb)

    outputs = p.frb_output_params(obj=frb)
    for key in filter(lambda x: "in_" in x, outputs):
        print(f"{key}: {outputs[key]}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Set up a new epoch.")
    parser.add_argument('--op', help='Name of frb, eg FRB180924')

    # Load arguments

    args = parser.parse_args()

    main(frb=args.op)
