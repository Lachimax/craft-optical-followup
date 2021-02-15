# Code by Lachlan Marnoch, 2019 - 2020

import os
from shutil import copyfile
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy

import craftutils.utils as u
import craftutils.fits_files as f
import craftutils.params as p
from craftutils.photometry import fit_background_fits

from astropy.wcs import WCS
from astropy.io import fits


def main(data_dir, data_title, origin, destination):
    print("\nExecuting Python script pipeline_fors2/5-background_subtract.py, with:")
    print(f"\tepoch {data_title}")
    print(f"\torigin directory {origin}")
    print(f"\tdestination directory {destination}")
    print()

    frame = 100

    methods = ["ESO backgrounds only", "SExtractor backgrounds only", "polynomial fit", "Gaussian fit", "median value"]

    eso_back = False

    method = u.select_option(message="Please select the background subtraction method.", options=methods,
                             default="polynomial fit")
    degree = None
    if method == "polynomial fit":
        degree = u.user_input(message=f"Please enter the degree of {method} to use:", typ=int, default=3)
    elif method == "ESO backgrounds only":
        eso_back = True
    if method not in ["ESO backgrounds only", "SExtractor backgrounds only"]:
        local = u.select_yn(message="Use a local fit?", default=True)
    else:
        local = False
    global_sub = False
    trim_image = False
    if local:
        global_sub = u.select_yn(message="Subtract local fit from entire image?", default="n")
        if not global_sub:
            trim_image = u.select_yn(message="Trim images to subtracted region?", default="y")

    # if not eso_back and method != "SExtractor backgrounds only":
    #     eso_back = u.select_yn(message="Subtract ESO Reflex fitted backgrounds first?", default=False)

    outputs = p.object_output_params(data_title, instrument='FORS2')

    data_dir = u.check_trailing_slash(data_dir)

    destination = u.check_trailing_slash(destination)
    destination = data_dir + destination
    u.mkdir_check_nested(destination)

    origin = u.check_trailing_slash(origin)
    science_origin = data_dir + origin + "science/"
    print(science_origin)

    filters = outputs['filters']

    background_origin_eso = ""
    if eso_back:
        background_origin_eso = data_dir + "/" + origin + "/backgrounds/"

    if method == "SExtractor backgrounds only":
        background_origin = f"{data_dir}{origin}backgrounds_sextractor/"
    elif method == "polynomial fit":
        background_origin = f"{destination}backgrounds_{method.replace(' ', '')}_degree_{degree}_local_{local}_globalsub_{global_sub}/"
    else:
        background_origin = f"{destination}backgrounds_{method.replace(' ', '')}_local_{local}_globalsub_{global_sub}/"
    frb_params = p.object_params_frb(obj=data_title[:-2])

    trimmed_path = ""
    if trim_image:
        trimmed_path = f"{data_dir}{origin}trimmed_to_background/"
        u.mkdir_check_nested(trimmed_path)

    ra = frb_params["burst_ra"]
    dec = frb_params["burst_dec"]

    for fil in filters:
        trimmed_path_fil = ""
        if trim_image:
            trimmed_path_fil = f"{trimmed_path}{fil}/"
            u.mkdir_check(trimmed_path_fil)
        background_fil_dir = f"{background_origin}{fil}/"
        u.mkdir_check_nested(background_fil_dir)
        science_destination_fil = f"{destination}science/{fil}/"
        u.mkdir_check_nested(science_destination_fil)
        files = os.listdir(science_origin + fil + "/")
        for file_name in files:
            if file_name.endswith('.fits'):
                new_file = file_name.replace("norm", "bg_sub")
                new_path = f"{science_destination_fil}/{new_file}"
                print("NEW_PATH:", new_path)
                science = science_origin + fil + "/" + file_name
                # First subtract ESO Reflex background images
                if eso_back:
                    background_eso = background_origin_eso + fil + "/" + file_name.replace("SCIENCE_REDUCED",
                                                                                           "PHOT_BACKGROUND_SCI")

                    f.subtract_file(file=science, sub_file=background_eso, output=new_path)
                    science = new_path

                if method != "ESO backgrounds only":
                    print("Science image:", science)
                    science = fits.open(science)
                    print("Science file:", science)
                    wcs_this = WCS(header=science[0].header)
                    x, y = wcs_this.all_world2pix(ra, dec, 0)
                    if method == "SExtractor backgrounds only":
                        background = background_origin + fil + "/" + file_name + "_back.fits"
                        print("Background image:", background)
                    else:
                        if method == "median value":
                            background_value = np.nanmedian(science[0].data)
                            background = deepcopy(science)
                            background[0].data = np.array(background_value, shape=science[0].data.shape)
                            background_path = background_origin + fil + "/" + file_name.replace("SCIENCE_REDUCED",
                                                                                                "PHOT_BACKGROUND_MEDIAN")

                        # Next do background fitting.
                        else:
                            background = fit_background_fits(image=science, model_type=method[:method.find(" ")],
                                                             deg=degree, local=local,
                                                             global_sub=global_sub,
                                                             centre_x=x, centre_y=y, frame=frame)
                            background_path = background_origin + fil + "/" + file_name.replace("SCIENCE_REDUCED",
                                                                                                "PHOT_BACKGROUND_FITTED")
                        print("Writing background to:")
                        print(background_path)
                        background.writeto(background_path, overwrite=True)

                        if trim_image:
                            left = int(x - frame)
                            right = int(x + frame)
                            bottom = int(y - frame)
                            top = int(y + frame)
                            print("TRIMMED_PATH_FIL:", trimmed_path_fil)

                            science = f.trim_file(path=science, left=left, right=right, top=top, bottom=bottom,
                                                  new_path=trimmed_path_fil + file_name.replace("norm.fits",
                                                                                                "trimmed_to_back.fits"))
                            print("Science after trim:", science)
                            background = f.trim_file(path=background, left=left, right=right, top=top, bottom=bottom,
                                                     new_path=background_path)

                    print("SCIENCE:", science)
                    print("BACKGROUND:", background)
                    subbed = f.subtract_file(file=science, sub_file=background, output=new_path)

                    plt.hist(subbed[0].data[int(y - frame):int(y + frame), int(x - frame):int(x + frame)].flatten(),
                             bins=10)
                    plt.savefig(new_path[:new_path.find("bg_sub")] + "histplot.png")
                    plt.close()

    copyfile(data_dir + "/" + origin + "/" + data_title + ".log", destination + data_title + ".log")
    u.write_log(path=destination + data_title + ".log",
                action=f'Backgrounds subtracted using 4-background_subtract.py with method {method}\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Subtract backgrounds provided by ESO Reflex from science images, and also divide all values by "
                    "exposure time.")
    parser.add_argument('--directory', help='Main data directory(probably starts with "MJD"')
    parser.add_argument('--op', help='Name of object parameter file without .yaml, eg FRB180924_1')
    parser.add_argument('--origin',
                        help='Folder within data_dir to copy target files from.',
                        type=str,
                        default="4-divided_by_exp_time")
    parser.add_argument('--destination',
                        help='Folder within data_dir to copy processed files to.',
                        type=str,
                        default="5-background_subtracted_with_python")

    # Load arguments

    args = parser.parse_args()

    main(data_dir=args.directory, data_title=args.op, origin=args.origin,
         destination=args.destination)
