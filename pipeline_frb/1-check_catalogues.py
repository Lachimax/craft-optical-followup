import craftutils.retrieve as r

def main(frb:str):
    r.update_frb_des_photometry(frb=frb)
    r.update_frb_sdss_photometry(frb=frb)
    r.update_frb_irsa_extinction(frb=frb)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Set up a new epoch.")
    parser.add_argument('--op', help='Name of frb, eg FRB180924')

    # Load arguments

    args = parser.parse_args()

    main(frb=args.op)