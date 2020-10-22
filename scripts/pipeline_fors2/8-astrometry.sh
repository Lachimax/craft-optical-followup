#!/usr/bin/env bash

# Copy header from coadded image to astrometry image.

param_file=$1
proj_param_file=$2
origin=$3
destination=$4

if [[ -z ${proj_param_file} ]]; then
  proj_param_file=unicomp
fi

if [[ -z ${origin} ]]; then
  origin=7-trimmed_again/
fi

if [[ -z ${destination} ]]; then
  destination=8-astrometry/
fi

proj_dir=$(jq -r .proj_dir param/project/${proj_param_file}.json)

if ! key=$(jq -r .astrometry param/keys.json)
then
  echo "Astrometry.net key required; keys.json not found."
  exit
fi

skip_astrometry=$(jq -r .skip_astrometry "param/epochs_fors2/${param_file}.json")
data_dir=$(jq -r .data_dir "param/epochs_fors2/${param_file}.json")
data_title=$(jq -r .data_title "param/epochs_fors2/${param_file}.json")

data_dir=$(jq -r .data_dir "param/epochs_fors2/${param_file}.json")
do_sextractor=$(jq -r .do_sextractor "param/epochs_fors2/${param_file}.json")
threshold=$(jq -r .threshold "param/epochs_fors2/${param_file}.json")
deepest_filter=$(jq -r .deepest_filter "param/epochs_fors2/${param_file}.json")

object=${data_title::-2}

ra=$(jq -r .hg_ra "param/FRBs/${object}.json")
dec=$(jq -r .hg_dec "param/FRBs/${object}.json")

dir=${data_dir}${destination}

mkdir "${dir}"

cd "${dir}" || return

# Move relevant files to astrometry folder and rename for anonymity.
rm ./*coadded.fits
if ! ${skip_astrometry}; then
  rm ./*astrometry.fits
fi
cp "${data_dir}/${origin}/"*_coadded.fits .

# Submit to astrometry.net for solving.

coadded=$(ls -d *coadded.fits*)
cd "${proj_dir}" || exit
# Use astrometry.net client to tweak astrometry of images.
for image in ${coadded}; do
  if ! ${skip_astrometry}; then
    python2 "${proj_dir}scripts/astrometry-client.py" --apikey "${key}" -u "${dir}/${image}" -w --newfits "${dir}/${image::1}_astrometry.fits" --ra "${ra}" --dec "${dec}" --radius 1 --private --no_commercial
  fi
done
cd "${dir}" || exit

cp "${data_dir}${origin}${data_title}.log" "${data_dir}${destination}${data_title}.log"
date +%Y-%m-%dT%T >>"${data_dir}${destination}${data_title}.log"
echo Astrometry updated using astrometry.net, via 8-astrometry.sh >>"${data_dir}${destination}${data_title}.log"

sextractor_destination_path=${data_dir}/analysis/sextractor/${destination}/

if ${do_sextractor}; then
  # Copy final processed image to SExtractor directory
  mkdir "${sextractor_destination_path}"
  cp "${data_dir}${destination}"*"astrometry.fits" "${sextractor_destination_path}"
  if cd "${sextractor_destination_path}"; then
    cp "${proj_dir}/param/psfex/"* .
    cp "${proj_dir}/param/sextractor/default/"* .
    pwd
    for image in *astrometry.fits; do
      cd "${sextractor_destination_path}" || exit
      image_0=${image::1}
      sextractor "${image}" -c pre-psfex.sex -CATALOG_NAME "${image_0}_psfex.fits"
      # Run PSFEx to get PSF analysis
      psfex "${image_0}_psfex.fits"
      cd "${proj_dir}" || exit
      # Use python to extract the FWHM from the PSFEx output.
      python3 "${proj_dir}/scripts/pipeline_fors2/9-psf.py" --directory "${data_dir}" --psfex_file "${sextractor_destination_path}${image_0}_psfex.psf" --image_file "${sextractor_destination_path}${image}" --prefix "${image_0}"
      cd "${sextractor_destination_path}" || exit
      fwhm=$(jq -r ".${image_0}_fwhm_arcsec" "${data_dir}output_values.json")
      echo "FWHM: ${fwhm} arcsecs"
      sextractor "${image}" -c psf-fit.sex -CATALOG_NAME "${image_0}_psf-fit.cat" -PSF_NAME "${image_0}_psfex.psf" -SEEING_FWHM "${fwhm}" -DETECT_THRESH "${threshold}" -ANALYSIS_THRESH "${threshold}"
    done
  fi
fi

# Copy important header items across and do tweak if necessary.
cd "${proj_dir}" || exit
python3 "${proj_dir}/scripts/pipeline_fors2/8-astrometry.py" --op "${data_title}" --astrometry_path "${dir}/" --sextractor_path "${sextractor_destination_path}" --template "${deepest_filter}"
