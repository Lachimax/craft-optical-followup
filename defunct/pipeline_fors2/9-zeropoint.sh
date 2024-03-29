#!/usr/bin/env bash

# TODO: Turn this into a proper bash script.

param_file=$1
origin=$2
if [[ -z ${origin} ]]; then
  origin=8-astrometry/
fi
destination=$3
if [[ -z ${destination} ]]; then
  destination=9-zeropoint/
fi
folder=$4
write_paths=$5
if [[ -z ${write_paths} ]]; then
  write_paths=true
fi
kron_radius=$6

config_file="param/config.json"
if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi
param_dir=$(jq -r .param_dir "${config_file}")

data_dir=$(jq -r .data_dir "${param_dir}/epochs_fors2/${param_file}.json")
data_title=${param_file}
skip_esorex=$(jq -r .skip_esorex "${param_dir}/epochs_fors2/${param_file}.json")
do_dual_mode=$(jq -r .do_dual_mode "${param_dir}/epochs_fors2/${param_file}.json")
do_sextractor=$(jq -r .do_sextractor "${param_dir}/epochs_fors2/${param_file}.json")
photometry_apertures=$(jq -r .photometry_apertures "${param_dir}/epochs_fors2/${param_file}.json")
min_radius=$(jq -r .min_radius "${param_dir}/epochs_fors2/${param_file}.json")

if [[ -z ${kron_radius} ]]; then
  kron_radius=$(jq -r .sextractor_kron_radius "${param_dir}/epochs_fors2/${param_file}.json")
fi
deepest_filter=$(jq -r .deepest_filter "${param_dir}/epochs_fors2/${param_file}.json")
threshold=$(jq -r .threshold "${param_dir}/epochs_fors2/${param_file}.json")
df=${deepest_filter::1}

if [ "${photometry_apertures}" != "[]" ]; then
  str=""
  for ap in ${photometry_apertures}; do
    if [[ -n ${ap::-1} ]]; then
      str+="${ap}"
    fi
  done
  apertures=${str}
fi

echo
echo "Executing bash script pipeline_fors2/9-zeropoint.sh, with:"
echo "   epoch ${param_file}"
echo "   origin directory ${origin}"
echo "   destination directory ${destination}"
echo "   write_paths ${write_paths}"
echo "   kron_radius ${kron_radius}"
echo

if ${do_sextractor}; then
  # Copy final processed image to SExtractor directory
  sextractor_destination_path=${data_dir}/analysis/sextractor/${destination}
  echo "SExtractor destination path: ${sextractor_destination_path}"
  mkdir -p "${sextractor_destination_path}"
  if cp "${data_dir}${origin}"*"astrometry_tweaked.fits" "${sextractor_destination_path}"; then
    suff="astrometry_tweaked.fits"
  elif cp "${data_dir}${origin}"*"astrometry.fits" "${sextractor_destination_path}"; then
    suff="astrometry.fits"
  elif cp "${data_dir}${origin}"*"coadded.fits" "${sextractor_destination_path}"; then
    suff="coadded.fits"
  else
    echo "Could not find any processed image in ${data_dir}${origin} to SExtract."
    exit
  fi

  if cd "${sextractor_destination_path}"; then
    cp "${proj_dir}/param/psfex/"* .
    cp "${proj_dir}/param/sextractor/default/"* .
    pwd
    for image in *"${suff}"; do
      cd "${sextractor_destination_path}" || exit
      image_0=${image::1}
      sex "${image}" -c pre-psfex.sex -CATALOG_NAME "${image_0}_psfex.fits"
      # Run PSFEx to get PSF analysis
      psfex "${image_0}_psfex.fits"
      cd "${proj_dir}" || exit
      # Use python to extract the FWHM from the PSFEx output.
      python3 "${proj_dir}/pipeline_fors2/9-psf.py" --directory "${data_dir}" --psfex_file "${sextractor_destination_path}${image_0}_psfex.psf" --image_file "${sextractor_destination_path}${image}" --prefix "${image_0}"
      cd "${sextractor_destination_path}" || exit
      fwhm=$(jq -r ".${image_0}_fwhm_arcsec" "${data_dir}output_values.json")
      echo "FWHM: ${fwhm} arcsecs"
      echo "MIN RADIUS:" "${min_radius}"
      echo "KRON RADIUS: ${kron_radius}"
      echo "FIXED APERTURES: ${apertures}"
      echo "THRESHOLD: ${threshold}"
      sex "${image}" -c "psf-fit.sex" -CATALOG_NAME "${image_0}_psf-fit.cat" -PSF_NAME "${image_0}_psfex.psf" -SEEING_FWHM 0 -PHOT_AUTOPARAMS "${kron_radius},${min_radius}" -DETECT_THRESH "${threshold}" -ANALYSIS_THRESH "${threshold}" -CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME "${image_0}_check.fits" -PHOT_APERTURES "${apertures}"
      # Run Sextractor again in local background mode.
      sex "${image}" -c "psf-fit.sex" -CATALOG_NAME "${image_0}_psf-fit_back_local.cat" -PSF_NAME "${image_0}_psfex.psf" -SEEING_FWHM 0 -PHOT_AUTOPARAMS "${kron_radius},${min_radius}" -DETECT_THRESH "${threshold}" -ANALYSIS_THRESH "${threshold}" -CHECKIMAGE_TYPE BACKGROUND -BACKPHOTO_TYPE LOCAL -PHOT_APERTURES "${apertures}"
      # If this is not the deepest image, we run in dual mode, using the deepest image for finding.
      if [[ ${image_0} != "${df}" ]]; then
        if ${do_dual_mode}; then
          sex "${df}_${suff},${image}" -c "psf-fit.sex" -CATALOG_NAME "${image_0}_dual-mode.cat" -PSF_NAME "${image_0}_psfex.psf" -SEEING_FWHM 0 -PHOT_AUTOPARAMS "${kron_radius},${min_radius}" -DETECT_THRESH "${threshold}" -ANALYSIS_THRESH "${threshold}" -CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME "${image_0}_check.fits" -PHOT_APERTURES "${apertures}"
          # Run Sextractor again in local background mode.
          sex "${df}_${suff},${image}" -c "psf-fit.sex" -CATALOG_NAME "${image_0}_dual-mode_back_local.cat" -PSF_NAME "${image_0}_psfex.psf" -SEEING_FWHM 0 -PHOT_AUTOPARAMS "${kron_radius},${min_radius}" -DETECT_THRESH "${threshold}" -ANALYSIS_THRESH "${threshold}" -BACKPHOTO_TYPE LOCAL -PHOT_APERTURES "${apertures}"
          cd "${proj_dir}" || exit
          if ${write_paths}; then
            python3 add_path.py --op "${data_title}" --key "${image_0}_cat_path${folder}" --path "${sextractor_destination_path}${image_0}_dual-mode.cat" --instrument FORS2
          fi
        else
          cd "${proj_dir}" || exit
          if ${write_paths}; then
            python3 add_path.py --op "${data_title}" --key "${image_0}_cat_path${folder}" --path "${sextractor_destination_path}${image_0}_psf-fit.cat" --instrument FORS2
          fi
        fi
      else
        cd "${proj_dir}" || exit
        if ${write_paths}; then
          python3 add_path.py --op "${data_title}" --key "${image_0}_cat_path${folder}" --path "${sextractor_destination_path}${image_0}_psf-fit.cat" --instrument FORS2
        fi
      fi
    done
  fi
fi

cd "${proj_dir}" || exit
echo 'Atmospheric extinction:'
python3 "pipeline_fors2/9-extinction_atmospheric.py" --op "${data_title}" -write

echo "Skip esorex ${skip_esorex}"
if ! ${skip_esorex}; then
  bash pipeline_fors2/9-esorex_zeropoint.sh "${param_file}"
fi

python3 "pipeline_fors2/9-zeropoint.py" --op "${data_title}" --instrument "fors2"
