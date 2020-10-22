#!/usr/bin/env bash
# Script by Lachlan Marnoch 2019

# Syntax: ./initial.sh param_file_name

param_file=$1

config_file="param/config.json"

if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi

param_dir=$(jq -r .param_dir "${config_file}")

data_dir=$(jq -r .data_dir "${param_dir}/epochs_fors2/${param_file}.json")
data_title=$(jq -r .data_title "${param_dir}/epochs_fors2/${param_file}.json")
skip_download=$(jq -r .skip_download "${param_dir}/epochs_fors2/${param_file}.json")
skip_copy=$(jq -r .skip_copy "${param_dir}/epochs_fors2/${param_file}.json")

eso_destination=$(jq -r .esoreflex_input_dir "${config_file}")

cd "${proj_dir}" || exit

cd "${data_dir}/" || exit

mkdir "${data_dir}/analysis"
mkdir "${data_dir}/analysis/sextractor"
mkdir "${data_dir}/analysis/sextractor/individuals/"
mkdir "${data_dir}/analysis/photometry_tests"
mkdir "${data_dir}/analysis/photometry_tests/sextractor"

mkdir "${data_dir}/calibration"
mkdir "${data_dir}/calibration/std_star"

if ! ${skip_download}; then

  # Download files
  echo "Enter ESO password:"
  chmod u+x download*script.sh
  for dl_script in download*.sh; do
    chmod u+x "${dl_script}"
    ./"${dl_script}"
  done

  # Rename folder
  if mv data_with_raw_calibs/ 0-data_with_raw_calibs/; then

    # Decompress any .Z files
    cd 0-data_with_raw_calibs/ || exit

  else
    mkdir 0-data_with_raw_calibs/
    mv FORS2* 0-data_with_raw_calibs/
    cd 0-data_with_raw_calibs/ || exit

  fi
  date +%Y-%m-%dT%T >>"${data_title}.log"
  echo "Files downloaded from ESO archive." >>"${data_title}.log"
  echo "Decompressing files..."
  if uncompress ./*.Z; then
    echo "Done."
  fi
fi

cd "${proj_dir}" || exit

echo "Writing FITS properties to file..."
if python3 scripts/pipeline_fors2/1-initial.py --output "${data_dir}" -op "${data_title}"; then
  echo "Done."
fi

for filter_path in "${data_dir}"calibration/std_star/*_*/; do
  for pointing_path in "${filter_path}"RA*_DEC*/; do
    if mkdir "${pointing_path}/0-data_with_raw_calibs/"; then
      mv "${pointing_path}/"*.fits "${pointing_path}/0-data_with_raw_calibs/"
    else
      echo "No standard star data provided for $(basename "${filter_path}")."
    fi
  done
done

if ! ${skip_copy}; then
  mkdir "${eso_destination}${data_title}"
  echo "Copying to ESOReflex input directory"
  cp "${data_dir}"/0-data_with_raw_calibs/* "${eso_destination}${data_title}"
  echo "Done."
fi
