# Code by Lachlan Marnoch, 2020

import craftutils.params as p


def main(epoch: str):
    param_dir = p.config['param_dir']
    data_dir = p.config['top_data_dir']
    p.add_params(file=param_dir + "epochs_fors2/" + epoch + ".yaml",
                 params={"data_dir": data_dir + epoch[:-2] + "/FORS2/new_epoch/"})

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Set up a new epoch.")
    parser.add_argument('--op', help='Name of frb, eg FRB180924')

    # Load arguments

    args = parser.parse_args()

    main(epoch=args.op)
