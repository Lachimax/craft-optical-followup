# Code by Lachlan Marnoch, 2019

from PyCRAFT import params
from PyCRAFT import photometry
from PyCRAFT import utils
from PyCRAFT import fits_files as f

from datetime import datetime as dt

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import astropy.io.fits as fits
import astropy.table as table
import astropy.time as time
from astropy import wcs
from astropy.visualization import (ImageNormalize, SqrtStretch, ZScaleInterval)


def main(data_title,
         trial_name,
         pix_tol,
         star_class_tol,
         show_plots):
    # Append current time to trial name.
    now = time.Time.now()
    now.format = 'isot'
    trial_name = str(now) + '_' + trial_name

    # Load parameter file
    properties = params.object_params_fors2(data_title)

    # Load values written by pipeline
    outputs = params.object_output_params(obj=data_title, instrument='FORS2')
    # Load path file written by pipeline
    paths = params.object_output_paths(obj=data_title, instrument='FORS2')
    # Load sextractor column names
    sextractor_names = params.sextractor_names_psf()

    # PAths to sextractor directories.
    ind_sextractor_path = properties['data_dir'] + 'analysis/sextractor/individuals_psf/'
    # We take the trimmed image instead of the final astrometry image because we need the astrometry to line up with
    # the individual files, which have not been astrometried, as much as possible.
    coadded_sextractor_path = properties['data_dir'] + 'analysis/sextractor/combined_trimmed_psf/'

    # Set up an output directory
    output_path = properties['data_dir'] + "analysis/photometry_tests/sextractor/" + trial_name + '/'
    os.mkdir(output_path)

    # Get filter names from paths inside sextractor folder.
    filter_dirs = next(os.walk(ind_sextractor_path))[1]

    for fil in filter_dirs:

        # Dictionary of parameters to be written to file at end.
        parameters = {'trial': trial_name,
                      'time': dt.now().strftime('%Y-%m-%dT%H:%M:%S'),
                      'pix_tol': pix_tol,
                      }

        # Read path of coadded image.
        coadded_path = paths[fil[0] + '_trimmed_image']
        # Open coadded image.
        coadded_image = fits.open(coadded_path)
        # Get pixel scales from coadded image.
        ra_pix_scale, dec_pix_scale = f.get_pixel_scale(coadded_image)
        parameters['pix_scale_deg'] = dec_pix_scale
        parameters['pix_scale_arc'] = dec_pix_scale * 3600
        # Use pixel scales to convert pixel tolerance into angular tolerance.
        ra_tolerance = pix_tol * ra_pix_scale
        dec_tolerance = pix_tol * dec_pix_scale
        # TODO: Do I need to worry about the pixel scale changing --slightly-- between the coadded and individual
        #  images? Probably not, but worth thinking about.
        # Get some parameters specific to the filter.
        extinction = properties[fil[0] + '_ext_up']
        extinction_err = properties[fil[0] + '_ext_up']
        airmass_coadded = outputs[fil[0] + '_airmass_mean']
        airmass_coadded_err = outputs[fil[0] + '_airmass_err']
        n_exposures = int(outputs[fil[0] + '_n_exposures'])
        parameters['extinction'] = extinction

        # Make a separate directory inside our output directory for each filter.
        output_path_fil = output_path + fil + "/"
        utils.mkdir_check(output_path_fil)
        parameters['output'] = output_path_fil

        # Path for this filter within the sextractor directory
        sextractor_fil_path = ind_sextractor_path + fil + "/"
        # Get only the image files in the sextractor directory
        fits_files = list(filter(lambda file: file[-10:] == '_norm.fits', os.listdir(sextractor_fil_path)))
        fits_files.sort()
        # Collect all of the individual sextractor catalogue files.
        cat_files = list(filter(lambda file: file[-12:] == '_psf-fit.cat', os.listdir(sextractor_fil_path)))
        cat_files.sort()

        # Path to sextractor catalogue of coadded image
        coadded_sex_cat_path = f'{coadded_sextractor_path}{fil[0]}_coadded.fits_psf-fit.cat'
        parameters['input_coadded'] = coadded_sex_cat_path
        parameters['input_individual'] = ind_sextractor_path

        # Load table of sextracted values
        coadded = table.Table(np.genfromtxt(coadded_sex_cat_path,
                                            names=sextractor_names))
        # Calculated the magnitudes from the flux column.
        coadded['mag'], coadded['mag_err_plus'], coadded['mag_err_minus'] = photometry.magnitude_complete(
            flux=coadded['flux_psf'],
            exp_time=1.,
            airmass=airmass_coadded,
            airmass_err=airmass_coadded_err,
            ext=extinction,
            ext_err=extinction_err
        )

        # Filter out nan mags
        remove = np.isnan(coadded['mag'])
        print(sum(np.invert(remove)), 'sources in coadded after removing nan mags')
        # Filter out non-stars
        remove = remove + (coadded['class_star'] < star_class_tol)
        print(sum(np.invert(remove)), 'sources in coadded after removing objects class_star <', star_class_tol)
        # Apply filter to full table.
        coadded = coadded[np.invert(remove)]
        coadded_data = coadded_image[0].data
        coadded_wcs = wcs.WCS(header=coadded_image[0].header)

        if show_plots:
            norm = ImageNormalize(coadded_data, interval=ZScaleInterval(), stretch=SqrtStretch())
            plt.imshow(coadded_data, norm=norm, origin='lower')
            plt.scatter(coadded['x'], coadded['y'])
            plt.show()

        # Empty dictionary to contain sextracted tables.
        tables = {}

        # Load individual catalogues for upper files.
        for j in range(len(cat_files)):
            if j % 2 == 0:
                title = cat_files[j]
                fits_file = fits_files[j]
                # Load individual image
                image = fits.open(sextractor_fil_path + '/' + fits_file)
                n_x = image[0].data.shape[1]
                n_y = image[0].data.shape[0]
                airmass = image[0].header['AIRMASS']
                # wcs_info = wcs.WCS(header=image[0].header)
                print(sextractor_fil_path + title)

                # Load catalogue from an individual sextractor cat file
                cat = table.Table(np.genfromtxt(sextractor_fil_path + title,
                                                names=sextractor_names))
                # Calculate magnitudes as above.
                cat['mag'], cat['mag_err_plus'], cat['mag_err_minus'] = photometry.magnitude_complete(
                    flux=cat['flux_aper'],
                    exp_time=1.,
                    airmass=airmass,
                    ext=extinction,
                    ext_err=extinction_err
                )

                # Filter out nan mags
                remove = np.isnan(cat['mag'])
                print(sum(np.invert(remove)), 'sources after removing nan mags')
                # Filter out non-stars, but be a little more generous here than with the coadded image.
                remove = remove + (cat['class_star'] < star_class_tol - 0.1)
                print(sum(np.invert(remove)), 'sources after removing objects class_star <', star_class_tol)
                # Filter out stars which lie outside the image, somehow.
                remove = remove + (cat['x'] > n_x)
                remove = remove + (cat['x'] < 0)
                remove = remove + (cat['y'] > n_y)
                remove = remove + (cat['y'] < 0)
                # Apply filter to full table.
                cat_filtered = cat[np.invert(remove)]

                tables[title] = cat_filtered
                if show_plots:
                    norm = ImageNormalize(image[0].data, interval=ZScaleInterval(), stretch=SqrtStretch())
                    plt.imshow(image[0].data, norm=norm, origin='lower')
                    plt.scatter(cat_filtered['x'], cat_filtered['y'])
                    plt.show()

        # Match all sources to the coadded table.
        mag_table = photometry.match_coordinates_multi(prime=coadded, match_tables=tables,
                                                       ra_tolerance=ra_tolerance,
                                                       dec_tolerance=dec_tolerance,
                                                       ra_name='ra', dec_name='dec',
                                                       mag_name='mag',
                                                       x_name='x',
                                                       y_name='y')
        print('Number of full matches: ', len(mag_table))
        parameters['matches'] = len(mag_table)

        if show_plots:
            norm = ImageNormalize(coadded_data, interval=ZScaleInterval(), stretch=SqrtStretch())
            plt.imshow(coadded_data, norm=norm, origin='lower')
            plt.scatter(mag_table['x'], mag_table['y'])
            plt.show()

        # Set up columns for some statistical analysis of the magnitudes:
        # Maximum magnitude value
        mag_table['max'] = np.nan
        # Minimum magnitude value
        mag_table['min'] = np.nan
        # Maximum value without considering coadded magnitude
        mag_table['max_nc'] = np.nan
        # Minimum value without considering coadded magnitude
        mag_table['min_nc'] = np.nan
        # Difference between min and max values
        mag_table['spread'] = np.nan
        # Difference between coadded value and maximum value
        mag_table['delta_up'] = np.nan
        # Difference between coadded value and maximum value
        mag_table['delta_down'] = np.nan
        # Standard deviation of values
        mag_table['std'] = np.nan
        # Standard deviation of values without considering coadded value
        mag_table['std_no_combine'] = np.nan
        # Mean
        mag_table['mean'] = np.nan
        # Mean without considering coadded value
        mag_table['mean_no_combine'] = np.nan
        # Median
        mag_table['median'] = np.nan
        # Median without considering coadded value
        mag_table['median_no_combine'] = np.nan

        # Fill those columns we just set up.
        for j, row in mag_table.iterrows():
            maxx = max(row[4:])
            max_nc = max(row[5:])
            minn = min(row[4:])
            min_nc = min(row[5:])
            spread = abs(maxx - minn)
            delta_up = abs(row['mag_prime'] - maxx)
            delta_down = abs(row['mag_prime'] - minn)
            std = np.std(row[4:])
            std_nc = np.std(row[5:])
            mean = np.nanmean(row[4:])
            mean_nc = np.nanmean(row[5:])
            median = np.nanmedian(row[4:])
            median_nc = np.nanmedian(row[5:])
            row['max'] = maxx
            row['max_nc'] = max_nc
            row['min'] = minn
            row['min_nc'] = min_nc
            row['spread'] = spread
            row['delta_up'] = delta_up
            row['delta_down'] = delta_down
            row['std'] = std
            row['std_no_combine'] = std_nc
            row['mean'] = mean
            row['mean_no_combine'] = mean_nc
            row['median'] = median
            row['median_no_combine'] = median_nc

        print('Median standard deviation: ', np.nanmedian(mag_table['std']))
        print('Median std, no combine', np.nanmedian(mag_table['std_no_combine']))

        # Subtraction test
        indices_1 = []
        indices_2 = []
        mag_primes = []
        mag_individuals = {}

        for i in range(20):
            index_1 = np.random.randint(len(mag_table))
            index_2 = np.random.randint(len(mag_table))

            indices_1.append(index_1)
            indices_2.append(index_2)
            mag_primes.append(mag_table.iloc[index_1]['mag_prime'] - mag_table.iloc[index_2]['mag_prime'])
            for exposure in range(n_exposures):
                if str(exposure) not in mag_individuals:
                    mag_individuals[str(exposure)] = []
                mag_individuals[str(exposure)].append(
                    mag_table.iloc[index_1]['mag_' + str(exposure)] - mag_table.iloc[index_2]['mag_' + str(exposure)])

        sub_test = pd.DataFrame()
        sub_test['obj_1'] = indices_1
        sub_test['obj_2'] = indices_2
        sub_test['mag_prime'] = mag_primes
        for exposure in range(n_exposures):
            sub_test['mag_' + str(exposure)] = mag_individuals[str(exposure)]

        for i, row in sub_test.iterrows():
            maxx = max(row[2:])
            max_nc = max(row[3:])
            minn = min(row[2:])
            min_nc = min(row[3:])
            spread = abs(maxx - minn)
            std = np.std(row[2:])
            std_nc = np.std(row[2:])
            mean = np.nanmean(row[2:])
            mean_nc = np.nanmean(row[3:])
            median = np.nanmedian(row[2:])
            median_nc = np.nanmedian(row[3:])
            row['max'] = maxx
            row['min'] = minn
            row['max_nc'] = max_nc
            row['min_nc'] = min_nc
            row['spread'] = spread
            row['std'] = std
            row['std_no_combine'] = std_nc
            row['mean'] = mean
            row['mean_no_combine'] = mean_nc
            row['median'] = median
            row['median_no_combine'] = median_nc

        sub_test.to_csv(output_path_fil + 'subtract_test_' + str(trial_name) + '.csv')
        mag_table.to_csv(output_path_fil + 'matched_mag_table_' + str(trial_name) + '.csv')

        parameters['median_stddev'] = mag_table['std'].median()
        parameters['min_stddev'] = mag_table['std'].min()
        parameters['max_stddev'] = mag_table['std'].max()
        parameters['mean_delta_up'] = mag_table['delta_up'].mean()
        parameters['mean_delta_down'] = mag_table['delta_down'].mean()

        mag_table.sort_values(by=['mag_prime'], inplace=True)

        plt.plot(mag_table['mag_prime'], mag_table['max'], label='Maximum', c='red')
        plt.plot(mag_table['mag_prime'], mag_table['min'], label='Minimum', c='blue')
        plt.plot(mag_table['mag_prime'], mag_table['mag_prime'], label='Coadded', c='violet')
        plt.legend()
        plt.title("Object magnitudes across images")
        plt.ylabel("Magnitude")
        plt.xlabel("Magnitude found in coadded image")
        # plt.xlim(max_mag, min_mag)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        
        plt.savefig(output_path_fil + "mags_max_min_prime.png")
        if show_plots:
            plt.show()
        plt.close()

        plt.plot(mag_table['mag_prime'], mag_table['max_nc'], label='Maximum in individual frames', c='red')
        plt.plot(mag_table['mag_prime'], mag_table['min_nc'], label='Minimum in individual frames', c='blue')
        plt.plot(mag_table['mag_prime'], mag_table['mag_prime'], label='Coadded', c='violet')
        plt.legend()
        plt.title("Object magnitudes across images")
        plt.ylabel("Magnitude found")
        plt.xlabel("Magnitude found in coadded image")
        # plt.xlim(max_mag, min_mag)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        
        plt.savefig(output_path_fil + "mags_max_min_prime_nc.png")
        if show_plots:
            plt.show()
        plt.close()

        plt.plot(mag_table['mag_prime'], mag_table['max_nc'], label='Maximum in individual frames', c='red')
        plt.plot(mag_table['mag_prime'], mag_table['min_nc'], label='Minimum in individual frames', c='blue')
        plt.plot(mag_table['mag_prime'], mag_table['median_no_combine'], label='Median of individual frames',
                 c='green')
        plt.plot(mag_table['mag_prime'], mag_table['mag_prime'], label='Coadded', c='violet')
        plt.legend()
        plt.title("Object magnitudes across images")
        plt.ylabel("Magnitude found")
        plt.xlabel("Magnitude found in coadded image")
        # plt.xlim(max_mag, min_mag)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        
        plt.savefig(output_path_fil + "mags_max_min_median_prime_nc.png")
        if show_plots:
            plt.show()
        plt.close()

        plt.plot(mag_table['mag_prime'], mag_table['max_nc'], label='Maximum in individual frames', c='red')
        plt.plot(mag_table['mag_prime'], mag_table['min_nc'], label='Minimum in individual frames', c='blue')
        plt.plot(mag_table['mag_prime'], mag_table['mean_no_combine'], label='Mean of individual frames',
                 c='green')
        plt.plot(mag_table['mag_prime'], mag_table['mag_prime'], label='Coadded', c='violet')
        plt.legend()
        plt.title("Object magnitudes across images")
        plt.ylabel("Magnitude found")
        plt.xlabel("Magnitude found in coadded image")
        # plt.xlim(max_mag, min_mag)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        
        plt.savefig(output_path_fil + "mags_max_min_mean_prime_nc.png")
        if show_plots:
            plt.show()
        plt.close()

        plt.plot(mag_table['mag_prime'], mag_table['max_nc'], label='Maximum in individual frames', c='red')
        plt.plot(mag_table['mag_prime'], mag_table['min_nc'], label='Minimum in individual frames', c='blue')
        plt.plot(mag_table['mag_prime'], mag_table['median_no_combine'], label='Median of individual frames',
                 c='green')
        plt.legend()
        plt.title("Object magnitudes across images")
        plt.ylabel("Magnitude found")
        plt.xlabel("Magnitude found in coadded image")
        # plt.xlim(max_mag, min_mag)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        
        plt.savefig(output_path_fil + "mags_max_min_median_nc.png")
        if show_plots:
            plt.show()
        plt.close()

        plt.plot(mag_table['mag_prime'], mag_table['max_nc'], label='Maximum in individual frames', c='red')
        plt.plot(mag_table['mag_prime'], mag_table['min_nc'], label='Minimum in individual frames', c='blue')
        plt.plot(mag_table['mag_prime'], mag_table['mag_prime'], label='Coadded', c='violet')
        for exposure in range(n_exposures):
            plt.scatter(mag_table['mag_prime'], mag_table['mag_' + str(exposure)], label='Mag in frame 0', marker='.',
                        s=4)
        plt.legend()
        plt.title("Object magnitudes across images")
        plt.ylabel("Magnitude found")
        plt.xlabel("Magnitude found in coadded image")
        # plt.xlim(max_mag, min_mag)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        
        plt.savefig(output_path_fil + "mags_max_min_prime_scatter.png")
        if show_plots:
            plt.show()
        plt.close()

        plt.plot(mag_table['mag_prime'], mag_table['mag_prime'], label='Coadded', c='violet')
        plt.plot(mag_table['mag_prime'], mag_table['median_no_combine'], label='Median of individual frames',
                 c='green')
        for exposure in range(n_exposures):
            plt.scatter(mag_table['mag_prime'], mag_table['mag_' + str(exposure)], label='Mag in frame 0', marker='.',
                        s=4)
        plt.legend()
        plt.title("Object magnitudes across images")
        plt.ylabel("Magnitude found")
        plt.xlabel("Magnitude found in coadded image")
        # plt.xlim(max_mag, min_mag)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        
        plt.savefig(output_path_fil + "mags_median_prime_scatter_nolines.png")
        if show_plots:
            plt.show()
        plt.close()

        plt.plot(mag_table['mag_prime'], mag_table['mag_prime'], label='Coadded', c='violet')
        for exposure in range(n_exposures):
            plt.plot(mag_table['mag_prime'], mag_table['mag_' + str(exposure)], label='Mag in frame ' + str(exposure))
        plt.legend()
        plt.title("Object magnitudes across images")
        plt.ylabel("Magnitude found")
        plt.xlabel("Magnitude found in coadded image")
        # plt.xlim(max_mag, min_mag)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        
        plt.savefig(output_path_fil + "mags_median_prime_scatter_nolines.png")
        if show_plots:
            plt.show()
        plt.close()

        utils.write_params(output_path_fil + "parameters_" + trial_name, parameters)

        shutil.copyfile(properties['data_dir'] + 'A-no_back_subtract/6-combined_with_montage/' + data_title + ".log",
                        output_path + data_title + ".log")
        utils.write_log(path=output_path + data_title + ".log",
                        action='Photometry tested with photometry_tests_sextractor.py')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run tests on the coadded images in comparison with individuals.')

    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('--trial_name',
                        help='Name of test, and of the folder within "output" where the data will be written',
                        type=str,
                        default='photometry_test_psf')
    parser.add_argument('--pix_tol',
                        help='Search tolerance for matching objects, in FORS2 (g) pixels.',
                        default=10.,
                        type=float)
    parser.add_argument('--star_class_tol',
                        help='Minimum value of star_class to be included in appraisal.',
                        default=0.9,
                        type=float)
    parser.add_argument('-show_plots',
                        help='Show plots onscreen? (Warning: may be tedious)',
                        action='store_true')

    args = parser.parse_args()

    main(data_title=args.op,
         trial_name=args.trial_name,
         pix_tol=args.pix_tol,
         star_class_tol=args.star_class_tol,
         show_plots=args.show_plots)
