#!/usr/bin/env bash

# Code by Lachlan Marnoch, 2019

#TODO: Build into pipeline with option in param file
#TODO: Make into a proper bash script

param_file=$1
dir=$2

echo
echo "Executing bash script pipeline_fors2/9-esorex_zeropoint.sh, with:"
echo "   epoch ${param_file}"
echo "   directory ${dir}"
echo

config_file="param/config.json"
if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi
param_dir=$(jq -r .param_dir "${config_file}")
eso_install_dir=$(jq -r .eso_install_dir "${config_file}")

eso_calib_dir=$(ls -d "${eso_install_dir}calib/fors-"*)
eso_bin_dir="${eso_install_dir}bin/"

data_dir=$(jq -r .data_dir "${param_dir}/epochs_fors2/${param_file}.json")
data_title=${param_file}

if [[ -z ${dir} ]]; then
  dir=${data_dir}calibration/std_star/
  echo "${dir}"
fi

mkdir "${dir}"

if python3 "${proj_dir}/pipeline_fors2/9-esorex_zeropoint_prep.py" --op "${data_title}" --directory "${dir}"; then
  cd "${dir}" || exit
  for filter in **/; do
    f_0=${filter::1}
    cd "${filter}" || exit
    shopt -s nullglob
    for pointing in RA*_DEC*/; do
      std_dir=${dir}${filter}${pointing}
      cd "${proj_dir}" || exit
      cp param/std_param_template.yaml "${std_dir}/params.yaml"
      if python3 "${proj_dir}/pipeline_fors2/9-esorex_zeropoint.py" --directory "${std_dir}" --eso_calib_dir "${eso_calib_dir}"; then
        cd "${std_dir}0-data_with_raw_calibs" || exit
        "${eso_bin_dir}esorex" fors_bias bias_up.sof
        mv master_bias.fits master_bias_up.fits
        "${eso_bin_dir}esorex" fors_img_sky_flat flats_up.sof
        mv master_sky_flat_img.fits master_sky_flat_img_up.fits
        "${eso_bin_dir}esorex" fors_zeropoint zp_up.sof
        mv standard_reduced_img.fits standard_reduced_img_up.fits

        cd ..

        mkdir 1-masters
        mkdir 2-std_reduced

        mv 0-data_with_raw_calibs/master*.fits 1-masters/
        mv 0-data_with_raw_calibs/standard_reduced_img*.fits 2-std_reduced/

        mkdir sextractor
        cp "${proj_dir}/param/sextractor/std/"* sextractor

        mkdir 3-trimmed
        cd "${proj_dir}" || exit
        python3 "${proj_dir}/pipeline_fors2/9-esorex_zeropoint_trim.py" --op "${data_title}" --input "${std_dir}2-std_reduced/standard_reduced_img_up.fits" --output "${std_dir}3-trimmed/standard_trimmed_img_up.fits" --path "${std_dir}"
        cd "${std_dir}" || exit

        cp 3-trimmed/* sextractor
        cd sextractor || exit

        # sextractor standard_reduced_img_up.fits -c fors_landolt.sex -CATALOG_NAME up.cat
        # sextractor standard_reduced_img_down.fits -c fors_landolt.sex -CATALOG_NAME down.cat

        cp "${proj_dir}param/psfex/"* .

        sex standard_trimmed_img_up.fits -c pre-psfex.sex -CATALOG_NAME standard_psfex.fits
        psfex standard_psfex.fits
        cd "${proj_dir}" || exit
        if python3 "${proj_dir}/pipeline_fors2/9-psf.py" --directory "${std_dir}" --psfex_file "${std_dir}sextractor/standard_psfex.psf" --image_file "${std_dir}sextractor/standard_trimmed_img_up.fits" --prefix ""; then
          cd "${std_dir}sextractor" || exit
          fwhm=$(jq -r "._fwhm_arcsec" "${std_dir}output_values.json")
          echo "FWHM: ${fwhm}"
          sex standard_trimmed_img_up.fits -c psf-fit.sex -CATALOG_NAME _psf-fit.cat -PSF_NAME standard_psfex.psf -SEEING_FWHM "${fwhm}"
          cd "${proj_dir}" || exit
          python3 add_path.py --op "${data_title}" --instrument fors2 --key "${f_0}_std_cat_sextractor" --path "${std_dir}sextractor/_psf-fit.cat"
        else
          echo "Python failed to extract a FWHM from PSFEx output."
        fi
      fi
      cd "${dir}" || exit
    done
  done
fi
