# Code by Lachlan Marnoch, 2019

import os

import astropy.time as time
import matplotlib
import numpy as np
from astropy import table
from craftutils import fits_files as ff
from craftutils import params as p
from craftutils import photometry
from craftutils.retrieve import update_std_photometry, cat_columns, photometry_catalogues
from craftutils.utils import mkdir_check, error_product
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'errorbar.capsize': 3})


def main(epoch,
         test_name,
         sex_x_col,
         sex_y_col,
         sex_ra_col,
         sex_dec_col,
         sex_flux_col,
         stars_only,
         show_plots,
         mag_range_sex_lower,
         mag_range_sex_upper,
         pix_tol,
         instrument):
    print("\nExecuting Python script pipeline_fors2/9-zeropoint.py, with:")
    print(f"\tepoch {epoch}")
    print()

    properties = p.object_params_fors2(epoch)
    outputs = p.object_output_params(obj=epoch, instrument='fors2')

    proj_paths = p.config

    output = properties['data_dir'] + '9-zeropoint/'
    mkdir_check(output)
    output_std = output + 'std/'
    mkdir_check(output_std)
    mkdir_check(output_std)

    std_path = properties['data_dir'] + 'calibration/std_star/'

    filters = outputs['filters']

    std_field_path = proj_paths['top_data_dir'] + "std_fields/"
    std_fields = list(filter(lambda path: os.path.isdir(path), os.listdir(std_field_path)))
    print('Standard fields with available catalogues:')
    print(std_fields)
    print()

    cat_names = photometry_catalogues

    for fil in filters:
        print('Doing filter', fil)

        f = fil[0]

        # Obtain zeropoints from FRB field, if data is available.

        print(f"\nDetermining science-field zeropoints for {epoch}, filter {fil}:\n")

        if f"{f}_zeropoints" in outputs:
            zeropoints = outputs[f"{f}_zeropoints"]
        else:
            zeropoints = {}
        zeropoints["science_field"] = {}

        for cat_name in cat_names:
            zeropoints["science_field"][cat_name], _ = photometry.zeropoint_science_field(epoch=epoch,
                                                                                          instrument=instrument,
                                                                                          test_name=None,
                                                                                          sex_x_col='XPSF_IMAGE',
                                                                                          sex_y_col='YPSF_IMAGE',
                                                                                          sex_ra_col='ALPHAPSF_SKY',
                                                                                          sex_dec_col='DELTAPSF_SKY',
                                                                                          sex_flux_col='FLUX_PSF',
                                                                                          stars_only=True,
                                                                                          star_class_col='CLASS_STAR',
                                                                                          star_class_tol=0.95,
                                                                                          show_plots=False,
                                                                                          mag_range_sex_lower=-100.,
                                                                                          mag_range_sex_upper=100.,
                                                                                          pix_tol=5.,
                                                                                          separate_chips=True,
                                                                                          cat_name=cat_name)

        # Obtain zeropoints from available standard fields and available data.

        print(f"\nDetermining standard-field zeropoints for {epoch}, filter {fil}\n")

        zeropoints["standard_field"] = {}

        fil_path = std_path + fil + '/'

        if os.path.isdir(fil_path):
            fields = filter(lambda d: os.path.isdir(fil_path + d), os.listdir(fil_path))

            output_path_fil_std = output_std + '/' + fil + '/'
            mkdir_check(output_path_fil_std)

            for field in fields:

                zeropoints["standard_field"][field] = {}

                ra = float(field[field.find("RA") + 2:field.find("_")])
                dec = float(field[field.find("DEC") + 3:])

                print("Looking for photometry data in field " + field + ":")
                mkdir_check(std_field_path + field)
                field_path = fil_path + field + '/'
                output_path = output_path_fil_std + field + '/'

                std_cat_path = std_field_path + field + '/'

                std_properties = p.load_params(field_path + 'params.yaml')
                use_sex_star_class = std_properties['use_sex_star_class']

                # Cycle through the three catalogues used to determine zeropoint, in order of preference.

                for cat_name in cat_names:

                    # Check for photometry on-disk in the relevant catalogue; if none present, attempt to retrieve from
                    # online archive.
                    print(f"In {cat_name}:")

                    output_path_cat = output_path + cat_name

                    cat_path = f"{std_cat_path}{cat_name}/{cat_name}.csv"

                    if not (os.path.isdir(std_cat_path + cat_name)) or os.path.isfile(cat_path):
                        print("None found on disk. Attempting retrieval from archive...")
                        if update_std_photometry(ra=ra, dec=dec, cat=cat_name) is None:
                            print("\t\tNo data found in archive.")
                            zeropoints["standard_field"][field][cat_name] = None
                            continue

                    column_names = cat_columns(cat=cat_name, f=f)
                    cat_ra_col = column_names['ra']
                    cat_dec_col = column_names['dec']
                    cat_mag_col = column_names['mag_psf']
                    if not use_sex_star_class:
                        star_class_col = column_names['class_star']
                    else:
                        star_class_col = 'CLASS_STAR'

                    cat_type = 'csv'

                    sextractor_path = field_path + 'sextractor/_psf-fit.cat'
                    image_path = field_path + '3-trimmed/standard_trimmed_img_up.fits'
                    star_class_tol = std_properties['star_class_tol']

                    now = time.Time.now()
                    now.format = 'isot'
                    test_name = str(now) + '_' + test_name

                    mkdir_check(properties['data_dir'] + '/analysis/zeropoint/')
                    mkdir_check(output_path)

                    exp_time = ff.get_exp_time(image_path)

                    print('SExtractor catalogue path:', sextractor_path)
                    print('Image path:', image_path)
                    print('Catalogue name:', cat_name)
                    print('Catalogue path:', cat_path)
                    print('Class star column:', star_class_col)
                    print('Output:', output_path_cat)
                    print('Exposure time:', exp_time)
                    print("Use sextractor class star:", use_sex_star_class)

                    zeropoints["standard_field"][field][cat_name] = photometry.determine_zeropoint_sextractor(
                        sextractor_cat_path=sextractor_path,
                        image=image_path,
                        cat_path=cat_path,
                        cat_name=cat_name,
                        output_path=output_path_cat,
                        show=show_plots,
                        cat_ra_col=cat_ra_col,
                        cat_dec_col=cat_dec_col,
                        cat_mag_col=cat_mag_col,
                        sex_ra_col=sex_ra_col,
                        sex_dec_col=sex_dec_col,
                        sex_x_col=sex_x_col,
                        sex_y_col=sex_y_col,
                        pix_tol=pix_tol,
                        flux_column=sex_flux_col,
                        mag_range_sex_upper=mag_range_sex_upper,
                        mag_range_sex_lower=mag_range_sex_lower,
                        stars_only=stars_only,
                        star_class_tol=star_class_tol,
                        star_class_col=star_class_col,
                        exp_time=exp_time,
                        y_lower=0,
                        cat_type=cat_type,
                    )

        output_dict = {f + '_zeropoints': zeropoints}
        p.add_output_values(obj=epoch, instrument='FORS2', params=output_dict)

    outputs = p.object_output_params(obj=epoch, instrument='fors2')

    output_path_final = output + "collated/"
    mkdir_check(output_path_final)

    print("Collating zeropoints...")

    for fil in filters:
        print(f"For {fil}:")
        f = fil[0]
        output_path_final_f = output_path_final + fil + "/"
        mkdir_check(output_path_final_f)
        zeropoints = outputs[f"{f}_zeropoints"]
        airmass_sci = outputs[f"{f}_airmass_mean"]
        airmass_sci_err = outputs[f"{f}_airmass_err"]
        extinction = outputs[f"{f}_extinction"]
        extinction_err = outputs[f"{f}_extinction_err"]
        zeropoint_tbl = table.Table(
            dtype=[("type", 'S15'),
                   ("field", 'S25'),
                   ("cat", 'S10'),
                   ("zeropoint", float),
                   ("zeropoint_err", float),
                   ("airmass", float),
                   ("airmass_err", float),
                   ("extinction", float),
                   ("extinction_err", float),
                   ("zeropoint_ext_corr", float),
                   ("zeropoint_ext_corr_err", float),
                   ("n_matches", float)])

        if f"provided" in zeropoints:
            print(f"\tProvided:")
            zeropoint_prov = zeropoints["provided"]
            zeropoint_tbl.add_row(["provided",
                                   "N/A",
                                   "N/A",
                                   zeropoint_prov["zeropoint"],
                                   zeropoint_prov["zeropoint_err"],
                                   0.0,
                                   0.0,
                                   extinction,
                                   extinction_err,
                                   zeropoint_prov["zeropoint"],
                                   zeropoint_prov["zeropoint_err"],
                                   0])

        print(f"\tScience field:")
        for cat_name in zeropoints["science_field"]:
            print(f"\t\t{cat_name}")
            zeropoint_sci = zeropoints["science_field"][cat_name]

            if zeropoint_sci is not None:
                zeropoint_corrected = zeropoint_sci["zeropoint"] + extinction * airmass_sci

                print("\t\t\t", zeropoint_sci["zeropoint"], extinction, airmass_sci)

                zeropoint_corrected_err = zeropoint_sci["zeropoint_err"] + error_product(value=extinction * airmass_sci,
                                                                                         measurements=[extinction,
                                                                                                       airmass_sci],
                                                                                         errors=[extinction_err,
                                                                                                 airmass_sci_err])
                zp_cat = table.Table.read(zeropoint_sci["matches_cat_path"])
                zeropoint_tbl.add_row(["science_field",
                                       epoch[:-2],
                                       cat_name,
                                       zeropoint_sci["zeropoint"],
                                       zeropoint_sci["zeropoint_err"],
                                       airmass_sci,
                                       airmass_sci_err,
                                       extinction,
                                       extinction_err,
                                       zeropoint_corrected,
                                       zeropoint_corrected_err,
                                       len(zp_cat)
                                       ])
                plt.scatter(zp_cat["mag_cat"], zp_cat["mag"], c='green')
                plt.errorbar(zp_cat["mag_cat"], zp_cat["mag"], yerr=zp_cat["mag_err"], linestyle="None", c='black')
                plt.plot(zp_cat["mag_cat"], zp_cat["mag_cat"] - zeropoint_sci["zeropoint"], c="violet")
                plt.title(f"{fil}: science-field, {cat_name}")
                plt.figtext(0, 0.01,
                            f'zeropoint - kX: {zeropoint_sci["zeropoint"]} ± {zeropoint_sci["zeropoint_err"]}\n'
                            f'zeropoint: {zeropoint_corrected} ± {zeropoint_corrected_err}\n'
                            f'num stars: {len(zp_cat)}')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig(output_path_final_f + f"zeropoint_science_{cat_name}.pdf")
                plt.show()
                plt.close()

        print("\tStandard fields:")
        for field in zeropoints["standard_field"]:
            print(f"\t\t{field}")
            for cat_name in zeropoints["standard_field"][field]:
                print(f"\t\t\t{cat_name}")
                zeropoint_std = zeropoints["standard_field"][field][cat_name]
                if zeropoint_std is not None:
                    zeropoint_corrected = zeropoint_std["zeropoint"] + extinction * zeropoint_std["airmass"]

                    print("\t\t\t\t", zeropoint_std["zeropoint"], extinction, zeropoint_std["airmass"])

                    zeropoint_corrected_err = zeropoint_std["zeropoint_err"] + error_product(
                        value=extinction * zeropoint_std["airmass"],
                        measurements=[extinction, zeropoint_std["airmass"]],
                        errors=[extinction_err, 0.0])
                    zp_cat = table.Table.read(zeropoint_std["matches_cat_path"])
                    zeropoint_tbl.add_row(["standard_field",
                                           field,
                                           cat_name,
                                           zeropoint_std["zeropoint"],
                                           zeropoint_std["zeropoint_err"],
                                           zeropoint_std["airmass"],
                                           0.0,
                                           extinction,
                                           extinction_err,
                                           zeropoint_corrected,
                                           zeropoint_corrected_err,
                                           len(zp_cat)
                                           ])
                    plt.scatter(zp_cat["mag_cat"], zp_cat["mag"], c='green')
                    plt.errorbar(zp_cat["mag_cat"], zp_cat["mag"], yerr=zp_cat["mag_err"], linestyle="None", c='black')
                    plt.plot(zp_cat["mag_cat"], zp_cat["mag_cat"] - zeropoint_std["zeropoint"], c="violet")
                    plt.title(f"{fil}: standard-field {field}, {cat_name}")
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.figtext(0, 0.01,
                                f'zeropoint - kX: {zeropoint_std["zeropoint"]} ± {zeropoint_std["zeropoint_err"]}\n'
                                f'zeropoint: {zeropoint_corrected} ± {zeropoint_corrected_err}\n'
                                f'num stars: {len(zp_cat)}')
                    plt.savefig(output_path_final_f + f"zeropoint_standard_{field}_{cat_name}.pdf")
                    plt.show()
                    plt.close()

        zeropoint_tbl["selection_index"] = zeropoint_tbl["n_matches"] / zeropoint_tbl["zeropoint_ext_corr_err"]
        best_arg = np.argmax(zeropoint_tbl["selection_index"])
        print("Best zeropoint:")
        best_zeropoint = zeropoint_tbl[best_arg]
        print(best_zeropoint)

        zeropoints = outputs[f + '_zeropoints']
        zeropoints['best'] = {"zeropoint": float(best_zeropoint['zeropoint']),
                              "zeropoint_err": float(best_zeropoint["zeropoint_err"]),
                              "airmass": float(best_zeropoint["airmass"]),
                              "airmass_err": float(best_zeropoint["airmass_err"]),
                              "type": str(best_zeropoint["type"]),
                              "catalogue": str(best_zeropoint["cat"])}
        output_dict = {f + '_zeropoints': zeropoints}
        p.add_output_values(obj=epoch, instrument='FORS2', params=output_dict)

        zeropoint_tbl.write(output_path_final_f + "zeropoints.csv", format="ascii.csv", overwrite=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Use the PSF-fitted magnitudes to extract a zeropoint.')

    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        default='FRB181112_1',
                        type=str)
    parser.add_argument('--test_name',
                        help='Name of test, and of the folder within "output" where the data will be written',
                        default='test',
                        type=str)
    parser.add_argument('--instrument',
                        help='Name of instrument',
                        default='fors2',
                        type=str)
    parser.add_argument('--sextractor_x_column',
                        help='Name of SExtractor column containing x pixel coordinate.',
                        default='XPSF_IMAGE',
                        type=str)
    parser.add_argument('--sextractor_y_column',
                        help='Name of SExtractor column containing y pixel coordinate.',
                        default='YPSF_IMAGE',
                        type=str)
    parser.add_argument('--sextractor_ra_column',
                        help='Name of SExtractor column containing RA coordinate.',
                        default='ALPHAPSF_SKY',
                        type=str)
    parser.add_argument('--sextractor_dec_column',
                        help='Name of SExtractor column containing DEC coordinate.',
                        default='DELTAPSF_SKY',
                        type=str)
    parser.add_argument('--sextractor_flux_column',
                        help='Name of SExtractor column containing flux.',
                        default='FLUX_PSF',
                        type=str)
    parser.add_argument('--catalogue_path',
                        help='Path to catalogue for comparison.',
                        type=str)
    parser.add_argument('-not_stars_only',
                        help='Only use stars for zeropoint appraisal.',
                        action='store_false')
    parser.add_argument('-show',
                        help='Show plots onscreen? (Warning: may be tedious)',
                        action='store_true')
    parser.add_argument('--sextractor_mag_range_upper',
                        help='Upper SExtractor magnitude cutoff for zeropoint determination',
                        default=100.,
                        type=float)
    parser.add_argument('--sextractor_mag_range_lower',
                        help='Lower SExtractor magnitude cutoff for zeropoint determination',
                        default=-100.,
                        type=float)
    parser.add_argument('--pixel_tolerance',
                        help='Search tolerance for matching objects, in FORS2 (g) pixels.',
                        default=4.,
                        type=float)

    args = parser.parse_args()
    main(epoch=args.op,
         test_name=args.test_name,
         sex_x_col=args.sextractor_x_column,
         sex_y_col=args.sextractor_y_column,
         sex_ra_col=args.sextractor_ra_column,
         sex_dec_col=args.sextractor_dec_column,
         sex_flux_col=args.sextractor_flux_column,
         stars_only=args.not_stars_only,
         show_plots=args.show,
         mag_range_sex_lower=args.sextractor_mag_range_lower,
         mag_range_sex_upper=args.sextractor_mag_range_upper,
         pix_tol=args.pixel_tolerance,
         instrument=args.instrument
         )
