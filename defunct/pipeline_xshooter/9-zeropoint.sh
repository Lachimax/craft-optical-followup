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
  origin=8-astrometry/
fi

if [[ -z ${destination} ]]; then
  destination=9-zeropoint/
fi

# Extract parameters from param json files using jq.

proj_dir=$(jq -r .proj_dir param/project/${proj_param_file}.json)

data_dir=$(jq -r .data_dir "param/epochs_xshooter/${param_file}.json")
data_title=$(jq -r .data_title "param/epochs_xshooter/${param_file}.json")
do_dual_mode=$(jq -r .do_dual_mode "param/epochs_xshooter/${param_file}.json")
back_size=$(jq -r .sextractor_field_back_size "param/epochs_xshooter/${param_file}.json")
deepest_filter=$(jq -r .deepest_filter "param/epochs_xshooter/${param_file}.json")
threshold=$(jq -r .threshold "param/epochs_xshooter/${param_file}.json")
df=${deepest_filter::1}

sextractor_destination_path=${data_dir}${destination}

# TODO: Make psf a non-pipeline-specific file and rewrite sextractor scripts to point to it.

mkdir "${sextractor_destination_path}"
cp "${data_dir}${origin}"*astrometry* "${sextractor_destination_path}"
if cd "${sextractor_destination_path}"; then
  cp "${proj_dir}/param/psfex/"* .
  cp "${proj_dir}/param/sextractor/default/"* .
  for image in *astrometry.fits; do
    image_f=${image::1}
    sex "${image}" -c "pre-psfex.sex" -CATALOG_NAME "${image_f}_psfex.fits"
    # Run PSFEx to get PSF analysis
    psfex "${image_f}_psfex.fits"
    cd "${proj_dir}" || exit
    # Use python to extract the FWHM from the PSFEx output.
    python3 "${proj_dir}/pipeline_xshooter/7-psf.py" --directory "${data_dir}" --psfex_file "${sextractor_destination_path}${image_f}_psfex.psf" --image_file "${sextractor_destination_path}${image}" --prefix "${image_f}"
    cd "${sextractor_destination_path}" || exit
    fwhm=$(jq -r ".${image_f}_fwhm_arcsec" "${data_dir}output_values.json")
    echo "FWHM: ${fwhm} arcsecs"
    echo "BACKSIZE: ${back_size}"
    echo "THRESHOLD: ${threshold}"
    sex "${image}" -c psf-fit.sex -CATALOG_NAME "${image_f}_psf-fit.cat" -PSF_NAME "${image_f}_psfex.psf" -SEEING_FWHM "${fwhm}" -BACK_SIZE "${back_size}" -DETECT_THRESH "${threshold}" -ANALYSIS_THRESH "${threshold}"
    if [[ ${image_f} != "${df}" ]]; then
      if ${do_dual_mode}; then
        sex "${df}_astrometry.fits,${image}" -c psf-fit.sex -CATALOG_NAME "${image_f}_dual-mode.cat" -PSF_NAME "${image_f}_psfex.psf" -SEEING_FWHM "${fwhm}" -BACK_SIZE "${back_size}" -DETECT_THRESH "${threshold}" -ANALYSIS_THRESH "${threshold}"
        cd "${proj_dir}" || exit
        python3 /add_path.py --op "${param_file}" --key "${image_f}_cat_path" --path "${sextractor_destination_path}${image_f}_dual-mode.cat" --instrument XSHOOTER
      else
        cd "${proj_dir}" || exit
        python3 /add_path.py --op "${data_title}" --key "${image_f}_cat_path" --path "${sextractor_destination_path}${image_f}_psf-fit.cat" --instrument XSHOOTER
      fi
    else
      cd "${proj_dir}" || exit
      python3 /add_path.py --op "${data_title}" --key "${image_f}_cat_path" --path "${sextractor_destination_path}${image_f}_psf-fit.cat" --instrument XSHOOTER
    fi
    #cd "${proj_dir}" || exit
    #python3 plots/draw_fitting_params.py --image "${sextractor_destination_path}${image}" --cat "${sextractor_destination_path}${image_f}_psf-fit.cat"
    cd "${sextractor_destination_path}" || exit
  done
fi

cd "${proj_dir}" || exit
echo 'Science-field zeropoint:'
python3 "/zeropoint.py" --op "${data_title}" -write --instrument XSHOOTER -show --mag_tolerance 0.5 #-not_stars_only