# Code by Lachlan Marnoch, 2019
from craftutils import params as p
from craftutils.utils import check_trailing_slash

from matplotlib import pyplot as plt


def main(path):
    path = check_trailing_slash(path)
    outputs = p.tabulate_output_values(path=path, output=path + "output_values.csv")
    plt.plot(outputs["_fwhm_arcsec"], range(len(outputs)))
    plt.savefig(path + "fwhm.png")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Add a path to this dataset's output_paths.yaml")

    parser.add_argument('--path',
                        help='Path to tabulate.',
                        type=str)

    args = parser.parse_args()

    main(path=args.path)
