# Code by Lachlan Marnoch, 2019

import os
import shutil as sh
from astropy.io import fits

import craftutils.fits_files as ff
import craftutils.params as p
import craftutils.utils as u


def main(data_title: 'str'):
    eso_dir = p.config['esoreflex_output_dir']
    if os.path.isdir(eso_dir):
        obj_params = p.object_params_fors2(data_title)
        data_dir = obj_params['data_dir']
        destination = data_dir + "2-sorted/"
        date = None
        output_values = p.object_output_params(obj=data_title, instrument='FORS2')
        mjd = int(output_values['mjd_obs'])
        obj = output_values['object']

        print(f"Looking for data with object '{obj}' and MJD of observation {mjd} inside {eso_dir}")
        # Look for files with the appropriate object and MJD, as recorded in output_values

        # List directories in eso_output_dir; these are dates on which data was reduced using ESOReflex.
        eso_dirs = filter(lambda d: os.path.isdir(eso_dir + "/" + d), os.listdir(eso_dir))
        for directory in eso_dirs:
            directory += "/"
            # List directories within 'reduction date' directories.
            # These should represent individual images reduced.
            print(f"Searching {eso_dir + directory}")
            eso_subdirs = filter(lambda d: os.path.isdir(eso_dir + directory + d),
                                 os.listdir(eso_dir + directory))
            for subdirectory in eso_subdirs:
                subdirectory += "/"
                subpath = eso_dir + "/" + directory + subdirectory
                print(f"\tSearching {subpath}")
                # Get the files within the image directory.
                files = filter(lambda d: os.path.isfile(subpath + d),
                               os.listdir(subpath))
                for file_name in files:
                    # Retrieve the target object name from the fits file.
                    file_path = subpath + file_name
                    file = fits.open(file_path)
                    file_obj = ff.get_object(file)
                    file_mjd = int(ff.get_header_attribute(file, 'MJD-OBS'))
                    file_filter = ff.get_filter(file)
                    # Check the object name and observation date against those of the epoch we're concerned with.
                    if file_obj == obj and file_mjd == mjd:
                        # Check which type of file we have.
                        date = directory[:-1]
                        if file_name[-28:] == "PHOT_BACKGROUND_SCI_IMG.fits":
                            file_destination = f"{destination}/backgrounds/"
                            suffix = "PHOT_BACKGROUND_SCI_IMG.fits"
                        elif file_name[-25:] == "OBJECT_TABLE_SCI_IMG.fits":
                            file_destination = f"{destination}/obj_tbls/"
                            suffix = "OBJECT_TABLE_SCI_IMG.fits"
                        elif file_name[-24:] == "SCIENCE_REDUCED_IMG.fits":
                            file_destination = f"{destination}/science/"
                            suffix = "SCIENCE_REDUCED_IMG.fits"
                        else:
                            file_destination = f"{destination}/sources/"
                            suffix = "SOURCES_SCI_IMG.fits"
                        # Make this directory, if it doesn't already exist.
                        u.mkdir_check(file_destination)
                        # Make a subdirectory by filter.
                        file_destination += file_filter + "/"
                        u.mkdir_check(file_destination)
                        # Title new file.
                        file_destination += f"{data_title}_{subdirectory[:-1]}_{suffix}"
                        # Copy file to new location.
                        print(f"Copying: {file_path} to \n\t {file_destination}")
                        file.writeto(file_destination, overwrite=True)

        if not os.path.isfile(f"{destination}/{data_title}.log"):
            sh.copy(f"{data_dir}0-data_with_raw_calibs/{data_title}.log", f"{destination}/{data_title}.log")
        u.write_log(path=destination + data_title + ".log", action=f'Data reduced with ESOReflex.', date=date)
        u.write_log(path=destination + data_title + ".log", action='Files sorted using 2-sort_after_esoreflex.sh')

    else:
        print(f"ESO output directory '{eso_dir}' not found.")
        exit(1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='After reduction with ESOReflex, sorts fits files by their filters.')
    parser.add_argument('--op',
                        help='Name of epoch parameter file without .yaml, eg FRB180924_1',
                        type=str)
    args = parser.parse_args()
    main(data_title=args.op)
