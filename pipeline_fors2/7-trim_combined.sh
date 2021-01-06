#!/usr/bin/env bash
# Code by Lachlan Marnoch, 2019

param_file=$1
origin=$2
if [[ -z ${origin} ]]; then
  origin=6-combined_with_montage
fi
destination=$3
if [[ -z ${destination} ]]; then
  destination=7-trimmed_again
fi
sextractor_destination=$4

config_file="param/config.json"
if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi
param_dir=$(jq -r .param_dir "${config_file}")

data_dir=$(jq -r .data_dir "${param_dir}/epochs_fors2/${param_file}.json")
data_title=${param_file}
do_sextractor=$(jq -r .do_sextractor "${param_dir}/epochs_fors2/${param_file}.json")

mkdir "${data_dir}/${destination}/"

if ${do_sextractor}; then
  if [[ -z ${sextractor_destination} ]]; then
    sextractor_destination=${destination}/
  fi
  sextractor_destination_path=${data_dir}/analysis/sextractor/${sextractor_destination}
  mkdir "${sextractor_destination_path}"
  if ! python3 "${proj_dir}/scripts/pipeline_fors2/7-trim_combined.py" --directory "${data_dir}/${origin}/" --destination "${data_dir}/${destination}/" --object "${data_title}" --sextractor_directory "${sextractor_destination_path}" ; then
    echo "There was an error with the Python script for trimming."
    exit 1
  fi
  if cd "${sextractor_destination_path}"; then
    cp "${proj_dir}/param/psfex/"* .
    for image in *_coadded.fits; do
      sex "${image}" -c pre-psfex.sex -CATALOG_NAME "${image}_psfex.fits"
      # Run PSFEx to get PSF analysis
      psfex "${image}_psfex.fits"
      cd "${proj_dir}" || exit
      # Use python to extract the FWHM from the PSFEx output.
      python3 "${proj_dir}/scripts/pipeline_fors2/9-psf.py" --directory "${sextractor_destination_path}" --output_file "${image}_output_values" --psfex_file "${sextractor_destination_path}${image}_psfex.psf" --image_file "${sextractor_destination_path}${image}"
      cd "${sextractor_destination_path}" || exit
      fwhm=$(jq -r "._fwhm_arcsec" "${sextractor_destination_path}${image}_output_values.json")
      echo "FWHM: ${fwhm} arcsecs"
      sex "${image}" -c psf-fit.sex -CATALOG_NAME "${image}_psf-fit.cat" -PSF_NAME "${image}_psfex.psf" -SEEING_FWHM "${fwhm}"
    done
  fi
else
  python3 "${proj_dir}/scripts/pipeline_fors2/7-trim_combined.py" --directory "${data_dir}/${origin}/" --destination "${data_dir}/${destination}/" --object "${data_title}"
fi

if cd "${data_dir}/${destination}/"; then
  cp "${data_dir}/${origin}/${data_title}.log" "./${data_title}.log"
  date +%Y-%m-%dT%T >>"${data_title}.log"
  echo Coadded files trimmed with 7-trim_combined.sh >>"${data_title}.log"
fi

# Do Sextractor
#if ${do_sextractor} ; then
#    if cd ${sextractor_destination_path} ; then
#        cp ${proj_dir}/param/sextractor/default/* .
#        for sextract in $(ls sextract*.sh) ; do
#            chmod u+x ${sextract}
#            ./${sextract}
#         done
#     fi
#fi
