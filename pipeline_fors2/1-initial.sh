#!/usr/bin/env bash
# Script by Lachlan Marnoch 2019

# Syntax: ./initial.sh param_file_name

param_file=$1

config_file="param/config.json"

if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi

echo
echo "Executing bash script pipeline_fors2/1-initial.sh, with epoch ${param_file}"
echo

param_dir=$(jq -r .param_dir "${config_file}")

data_dir=$(jq -r .data_dir "${param_dir}/epochs_fors2/${param_file}.json")
data_title=${param_file}
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

# If we're skipping the download but it hasn't actually been downloaded, that's a problem.
if ${skip_download}; then
  if ! [[ -d "${data_dir}/0-data_with_raw_calibs" ]] || [[ -d "${data_dir}/0-data_with_raw_calibs/*.fits" ]]; then
    echo "There is no raw data present, however skip_download is set to 'true'. Would you like to override this, or cancel the procedure?"
    select yn in "Override" "Exit"; do
      case ${yn} in
      Override)
        skip_download=false
        break
        ;;
      Exit) exit ;;
      esac
    done
  fi
fi

if ! ${skip_download}; then

  if ! [[ -d "0-data_with_raw_calibs/" ]]; then
    mkdir 0-data_with_raw_calibs/
  fi

  mv ./*download*.sh 0-data_with_raw_calibs/ 2>/dev/null

  cd 0-data_with_raw_calibs/ || exit

  # Download files
  echo "Enter ESO password:"
  for dl_script in download*.sh; do
    bash ./"${dl_script}"
  done

  # Rename folder
  if [[ -d data_with_raw_calibs/ ]]; then
    mv data_with_raw_calibs/ .
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
if python3 pipeline_fors2/1-initial.py -op "${data_title}"; then
  echo "Done."
  if [[ "${data_dir}" == *"new_epoch"* ]]; then
    new_data_dir=$(jq -r .data_dir "${param_dir}/epochs_fors2/${param_file}.json")
    mv "${data_dir}" "${new_data_dir}"
    data_dir=${new_data_dir}
  fi
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
