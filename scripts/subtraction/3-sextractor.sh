#!/usr/bin/env bash
# Code by Lachlan Marnoch, 2019
# TODO: Turn this into a proper bash script.

param_file=$1
destination=$2
proj_param_file=$3
type=$4

if [[ -z ${proj_param_file} ]]; then
    proj_param_file=unicomp
fi

if [[ -z ${destination} ]]; then
    destination=1_FORS2/
fi

if [[ -z ${type} ]]; then
    type=normal
fi

# Extract parameters from param json files using jq.

proj_dir=$(jq -r .proj_dir param/project/${proj_param_file}.json)

data_dir=$(jq -r .data_dir "param/FRBs/${param_file}".json)
threshold=$(jq -r .threshold "param/FRBs/${param_file}".json)

sextractor_destination_path=${data_dir}subtraction/${destination}

if [[ ${type::5} != multi ]] ; then
    if cd "${sextractor_destination_path}" ; then
        for filter in  **/ ; do
            cd "${filter}" || exit
            cp "${proj_dir}param/sextractor/default/"* .
            # cp ${proj_dir}/param/psfex/* .
            for image in *difference.fits ; do
                # sextractor ${image} -c im.sex -CATALOG_NAME difference.cat -DETECT_THRESH 2.0
                fwhm_pix=$(jq -r .max_fwhm_pix "${sextractor_destination_path}${filter}output_values.json")
                fwhm_arcsec=$(jq -r .max_fwhm_arcsec "${sextractor_destination_path}${filter}output_values.json")
                aperture=$(jq -r .measurement_aperture "${sextractor_destination_path}${filter}output_values.json")
                echo "FWHM: ${fwhm_pix} pixels"
                echo "FWHM: ${fwhm_arcsec} arcseconds"
                echo "Aperture: ${aperture} pixels"
                sex "${image}" -c im.sex -CATALOG_NAME difference.cat -PHOT_APERTURES "${aperture}" -SEEING_FWHM "${fwhm_arcsec}" -DETECT_THRESH "${threshold}" -ANALYSIS_THRESH "${threshold}"
                cd "${proj_dir}" || exit
                cd "${sextractor_destination_path}${filter}" || exit
                # Run PSFEx to get PSF analysis
                # psfex psfex.fits

                # cd ${proj_dir}
                # Use python to extract the FWHM from the PSFEx output.
                # python3 ${proj_dir}/scripts/pipeline_fors2/9-psf.py --directory ${sextractor_destination_path} --psfex_file ${sextractor_destination_path}${filter}psfex.psf --image_file ${sextractor_destination_path}${filter}${image} --prefix ${f_0}
                # cd ${sextractor_destination_path}${filter}


                #sextractor ${image} -c psf-fit.sex -CATALOG_NAME difference_psf-fit.cat -PSF_NAME psfex.psf -PHOT_APERTURES 10 #-SEEING_FWHM ${fwhm} -DETECT_THRESH ${threshold} -ANALYSIS_THRESH ${threshold}
            done
            cd ..
        done
    fi

else
    if cd "${sextractor_destination_path}" ; then
        for path in  **/ ; do
            echo Going to "$(pwd)"/"${path}"
            cd "${path}" || exit
            for filter in **/ ; do
                cd "${filter}" || exit
                cp "${proj_dir}/param/sextractor/default/"* .
                # cp ${proj_dir}/param/psfex/* .
                pwd
                for image in *difference.fits ; do
                    # sextractor ${image} -c im.sex -CATALOG_NAME difference.cat -DETECT_THRESH 2.0
                    echo "${sextractor_destination_path}${path}${filter}output_values.json"
                    fwhm_pix=$(jq -r .max_fwhm_pix "${sextractor_destination_path}${path}${filter}output_values.json")
                    fwhm_arcsec=$(jq -r .max_fwhm_arcsec "${sextractor_destination_path}${path}${filter}output_values.json")
                    aperture=$(jq -r .measurement_aperture "${sextractor_destination_path}${path}${filter}output_values.json")
                    #echo "FWHM: ${fwhm_pix} pixels"
                    #echo "FWHM: ${fwhm_arcsec} arcseconds"
                    #echo "Aperture: ${aperture} pixels"
                    sex "${image}" -c im.sex -CATALOG_NAME difference.cat -PHOT_APERTURES "${aperture}" -DETECT_THRESH "${threshold}" -ANALYSIS_THRESH "${threshold}"
                    # cd "${proj_dir}" || exit
                    # python3 plots/draw_fitting_params.py --image ${sextractor_destination_path}${path}${filter}${image} --cat ${sextractor_destination_path}${path}${filter}difference.cat
                    # cd "${sextractor_destination_path}${path}${filter}" || exit
                    # Run PSFEx to get PSF analysis
                    # psfex psfex.fits

                    # cd ${proj_dir}
                    # Use python to extract the FWHM from the PSFEx output.
                    # python3 ${proj_dir}/scripts/pipeline_fors2/9-psf.py --directory ${sextractor_destination_path} --psfex_file ${sextractor_destination_path}${filter}psfex.psf --image_file ${sextractor_destination_path}${filter}${image} --prefix ${f_0}
                    # cd ${sextractor_destination_path}${filter}


                    #sextractor ${image} -c psf-fit.sex -CATALOG_NAME difference_psf-fit.cat -PSF_NAME psfex.psf -PHOT_APERTURES 10 #-SEEING_FWHM ${fwhm} -DETECT_THRESH ${threshold} -ANALYSIS_THRESH ${threshold}
                done
                cd ..
            done
            cd ..
        done
    fi
fi