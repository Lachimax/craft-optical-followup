#!/usr/bin/env bash
# Code by Lachlan Marnoch, 2019

param_file=$1

origin=$2
if [[ -z ${origin} ]]; then
  origin=3-trimmed_with_python/
fi

destination=$3
if [[ -z ${destination} ]]; then
  destination=4-divided_by_exp_time/
fi

echo
echo "Executing bash script pipeline_fors2/4-divide_by_exp_time.sh, with:"
echo "   epoch directory ${param_file}"
echo "   destination directory ${destination}"
echo

config_file="param/config.json"
if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi

param_dir=$(jq -r .param_dir "${config_file}")

data_dir=$(jq -r .data_dir "${param_dir}/epochs_fors2/${param_file}.json")
data_title=${param_file}
do_sextractor_individual=$(jq -r .do_sextractor_individual "${param_dir}/epochs_fors2/${param_file}.json")

mkdir -p "${data_dir}/${destination}/backgrounds_sextractor/"

if ${do_sextractor_individual}; then
  mkdir "${data_dir}/analysis/sextractor/${destination}/"
  python3 "${proj_dir}/pipeline_fors2/4-divide_by_exp_time.py" --op "${data_title}" --origin "${origin}" --destination ${destination} --sextractor_directory "${data_dir}/analysis/sextractor/${destination}"
  if cd "${data_dir}/analysis/sextractor/${destination}/"; then
    for fil in **/; do
      if cd "${fil}"; then
        sextractor_destination_path="${data_dir}/analysis/sextractor/${destination}/${fil}"
        if cp "${proj_dir}/param/psfex/"* .; then
          for image in *_norm.fits; do
            sex "${image}" -c pre-psfex.sex -CATALOG_NAME "${image::-5}_psfex.fits"
            # Run PSFEx to get PSF analysis
            psfex "${image::-5}_psfex.fits"
            cd "${proj_dir}" || exit
            # Use python to extract the FWHM from the PSFEx output.
            python3 "${proj_dir}/pipeline_fors2/9-psf.py" --directory "${sextractor_destination_path}" --output_file "${image::-5}_output_values" --psfex_file "${sextractor_destination_path}${image::-5}_psfex.psf" --image_file "${sextractor_destination_path}${image}"
            cd "${sextractor_destination_path}" || exit
            fwhm=$(jq -r "._fwhm_arcsec" "${sextractor_destination_path}${image::-5}_output_values.json")
            echo "FWHM: ${fwhm} arcsecs"
            sex "${image}" -c psf-fit.sex -CATALOG_NAME "${image::-5}_psf-fit.cat" -PSF_NAME "${image::-5}_psfex.psf" -SEEING_FWHM "${fwhm}" -CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME "${image::-5}_back.fits"
          done
        fi
        cd "${proj_dir}" || exit
        python3 "${proj_dir}/plot_fwhms.py" --path "${data_dir}/analysis/sextractor/${destination}/${fil}"
        cd "${data_dir}/analysis/sextractor/${destination}/${fil}" || exit
        mkdir -p "${data_dir}/${destination}/backgrounds_sextractor/${fil}/"
        cp ./*_back.fits "${data_dir}/${destination}/backgrounds_sextractor/${fil}/"
        cd ..
      fi
    done
  else
    exit
  fi
  cp "${data_dir}/3-trimmed_with_python/${data_title}.log" "${data_dir}/analysis/sextractor/${destination}/"
  date +%Y-%m-%dT%T >>"${data_dir}/analysis/sextractor/${destination}/${data_title}.log"
  echo Sextracted >>"${data_dir}/analysis/sextractor/${destination}/${data_title}.log"
  echo "All done."

else
  python3 "${proj_dir}/pipeline_fors2/4-divide_by_exp_time.py" --op "${data_title}" --origin "${origin}" --destination ${destination}
fi

date +%Y-%m-%dT%T >>"${data_dir}${destination}/${data_title}.log"
echo "Files divided by exposure time using 4-divide_by_exp_time.sh" >>"${data_dir}${destination}/${data_title}.log"
