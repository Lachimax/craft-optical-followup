# Code by Lachlan Marnoch, 2019
# This script trims the edges of the Montage-produced coaddition, in order to remove the regions with sub-nominal
# overlap.

import craftutils.fits_files as f
import craftutils.params as p
import os
import craftutils.utils as u
from shutil import copyfile


# TODO: Refactor all script inputs to match argparse inputs, for readability.

def main(comb_path, output_dir, obj, sextractor_path):

    print("\nExecuting Python script pipeline_fors2/7-trim_combined.py, with:")
    print(f"\tepoch {obj}")
    print(f"\toutput directory {output_dir}")
    print(f"\tsextractor directory {sextractor_path}")
    print()

    if sextractor_path is not None:
        if not os.path.isdir(sextractor_path):
            os.mkdir(sextractor_path)
        do_sextractor = True
    else:
        do_sextractor = False

    # Build a list of the filter prefixes used.
    fils = []
    files = list(filter(lambda x: x[-4:] == '.tbl', os.listdir(comb_path)))
    for file in files:
        if file[0] not in fils:
            fils.append(str(file[0]))

    for fil in fils:
        if do_sextractor:
            u.mkdir_check(sextractor_path)
        area_file = fil + "_coadded_area.fits"
        comb_file = fil + "_coadded.fits"

        left, right, bottom, top = f.detect_edges_area(comb_path + area_file)
        # Trim a little extra to be safe.
        left = left + 5
        right = right - 5
        top = top - 5
        bottom = bottom + 5

        f.trim_file(comb_path + comb_file, left=left, right=right, top=top, bottom=bottom,
                    new_path=output_dir + "/" + comb_file)
        # Keep a trimmed version of the area file, it comes in handy later.
        f.trim_file(comb_path + area_file, left=left, right=right, top=top, bottom=bottom,
                    new_path=output_dir + "/" + area_file)
        if do_sextractor:
            copyfile(output_dir + "/" + comb_file, sextractor_path + comb_file)

        p.add_output_path(obj=obj, instrument='fors2', key=fil[0] + '_trimmed_image', path=output_dir + "/" + comb_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Trim the edges off combined image.")
    parser.add_argument('--directory', help='Directory containing co-added images.')
    parser.add_argument('--destination', help='Output directory.')
    parser.add_argument('--object', help='Object name, eg FRB-181112--Host')
    parser.add_argument('--sextractor_directory', default=None,
                        help='Directory for sextractor scripts to be moved to.')

    args = parser.parse_args()

    main(comb_path=args.directory, output_dir=args.destination, obj=args.object,
         sextractor_path=args.sextractor_directory)
