#!/usr/bin/env bash
# Code by Lachlan Marnoch, 2019

param_file=$1
proj_param_file=$2
origin=$3
destination=$4
sextractor_destination=$5

if [[ -z ${proj_param_file} ]]; then
  proj_param_file=unicomp
fi

proj_dir=$(jq -r .proj_dir "param/project/${proj_param_file}.json")

data_dir=$(jq -r .data_dir "param/epochs_fors2/${param_file}.json")
data_title=$(jq -r .data_title "param/epochs_fors2/${param_file}.json")
do_sextractor=$(jq -r .do_sextractor "param/epochs_fors2/${param_file}.json")

if [[ -z ${origin} ]]; then
  origin=6-combined_with_montage
fi

if [[ -z ${destination} ]]; then
  destination=7-trimmed_again
fi

mkdir "${data_dir}/${destination}/"

if ${do_sextractor}; then
  if [[ -z ${sextractor_destination} ]]; then
    sextractor_destination=${destination}/
  fi
  sextractor_destination_path=${data_dir}/analysis/sextractor/${sextractor_destination}
  mkdir "${sextractor_destination_path}"
  python3 "${proj_dir}/scripts/pipeline_fors2/7-trim_combined.py" --directory "${data_dir}/${origin}/" --destination "${data_dir}/${destination}/" --object "${data_title}" --sextractor_directory "${sextractor_destination_path}"
  if cd "${sextractor_destination_path}"; then
    cp "${proj_dir}/param/psfex/"* .
    for image in *_coadded.fits; do
      sextractor "${image}" -c pre-psfex.sex -CATALOG_NAME "${image}_psfex.fits"
      # Run PSFEx to get PSF analysis
      psfex "${image}_psfex.fits"
      cd "${proj_dir}" || exit
      # Use python to extract the FWHM from the PSFEx output.
      python3 "${proj_dir}/scripts/pipeline_fors2/9-psf.py" --directory "${sextractor_destination_path}" --output_file "${image}_output_values" --psfex_file "${sextractor_destination_path}${image}_psfex.psf" --image_file "${sextractor_destination_path}${image}"
      cd "${sextractor_destination_path}" || exit
      fwhm=$(jq -r "._fwhm_arcsec" "${sextractor_destination_path}${image}_output_values.json")
      echo "FWHM: ${fwhm} arcsecs"
      sextractor "${image}" -c psf-fit.sex -CATALOG_NAME "${image}_psf-fit.cat" -PSF_NAME "${image}_psfex.psf" -SEEING_FWHM "${fwhm}"
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
