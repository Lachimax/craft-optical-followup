#!/usr/bin/env bash

# Code by Lachlan Marnoch, 2019

# TODO: Turn this into a proper bash script.

param_file=$1
proj_param_file=$2
origin=$3
destination=$4

if [[ -z ${proj_param_file} ]]; then
    proj_param_file=unicomp
fi

if [[ -z ${origin} ]]; then
    origin=0-data/
fi

if [[ -z ${destination} ]]; then
    destination=2-sextractor/
fi

# Extract parameters from param json files using jq.

proj_dir=$(jq -r .proj_dir param/project/${proj_param_file}.json)

data_dir=$(jq -r .data_dir "param/epochs_des/${param_file}.json")
back_size=$(jq -r .sextractor_field_back_size "param/epochs_des/${param_file}.json")
threshold=$(jq -r .threshold "param/epochs_des/${param_file}.json")

sextractor_destination_path=${data_dir}${destination}

mkdir "${sextractor_destination_path}"
cp "${data_dir}${origin}"*cutout* "${sextractor_destination_path}"
if cd "${sextractor_destination_path}" ; then
    cp "${proj_dir}/param/psfex/"* .
    cp "${proj_dir}/param/sextractor/default/"* .
    pwd
    for image in $(ls *cutout.fits) ; do
        image_f=${image::1}
        sex "${image}" -c "pre-psfex.sex" -CATALOG_NAME "${image_f}_psfex.fits"
        # Run PSFEx to get PSF analysis
        psfex "${image_f}_psfex.fits"
        cd "${proj_dir}" || exit
        # Use python to extract the FWHM from the PSFEx output.
        python3 "${proj_dir}/scripts/pipeline_fors2/9-psf.py" --directory "${data_dir}" --psfex_file "${sextractor_destination_path}${image_f}_psfex.psf" --image_file "${sextractor_destination_path}${image}" --prefix "${image_f}"
        cd "${sextractor_destination_path}" || exit
        fwhm=$(jq -r ".${image_f}_fwhm_arcsec" "${data_dir}output_values.json")
        echo "FWHM: ${fwhm} arcsecs"
        echo "BACKSIZE: ${back_size}"
        echo "THRESHOLD: ${threshold}"
        sex "${image}" -c psf-fit.sex -CATALOG_NAME "${image_f}_psf-fit.cat" -PSF_NAME "${image_f}_psfex.psf" -SEEING_FWHM "${fwhm}" -BACK_SIZE "${back_size}" -DETECT_THRESH "${threshold}" -ANALYSIS_THRESH "${threshold}"
    done
fi