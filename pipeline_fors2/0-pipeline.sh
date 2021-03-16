#!/usr/bin/env bash
# Code by Lachlan Marnoch, 2019 - 2021

usage() {
  echo "Usage: $0 -e FRBXXXXXX_X [-d subdirectory] [-b] [-s]" 1>&2
  exit 1
}

sub_back=false
insert_test_synth=false

while getopts "e:d:bs" option; do
  case "${option}" in
  e)
    param_file=${OPTARG}
    ;;
  d)
    folder=${OPTARG}
    ;;
  b)
    sub_back=true
    ;;
  s)
    insert_test_synth=true
    ;;
  *)
    usage
    ;;
  esac
done

if [[ -z ${param_file} ]]; then
  echo "No epoch specified."
  usage
  exit
fi

if [[ -z ${folder} ]]; then
  if ${sub_back}; then
    folder="B-back_subtract/"
  else
    folder=""
  fi
fi

echo
echo "Executing $0, with:"
echo "   epoch ${param_file}"
echo "   folder ${folder}"
echo "   sub_back ${sub_back}"
echo "   insert_test_synth ${insert_test_synth}"

if ! python3 "refresh_params.py"; then
  echo "Something went wrong with reading or writing the param files."
  exit
fi

config_file="param/config.json"

if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi

param_dir=$(jq -r .param_dir "${config_file}")
top_data_dir=$(jq -r .top_data_dir "${config_file}")

# For a new epoch, we can set up the directory.
if [ "${param_file}" == "new" ]; then
  echo "You are initialising a new epoch dataset."
  echo "Which FRB is this an observation of?"
  read -r frb_name
  while ! { [[ ${frb_name} =~ ^FRB[0-9]{6}$ ]] || [[ ${frb_name} =~ ^[0-9]{6}$ ]]; }; do
    echo "Please format response as FRBXXXXXX or as XXXXXX; eg FRB180924, or 180924."
    read -r frb_name
  done
  if [[ ${frb_name} =~ ^[0-9]{6} ]]; then
    frb_name="FRB${frb_name}"
  fi

  echo "Assign this epoch a unique number:"
  read -r epoch_number
  while ! [[ ${epoch_number} =~ ^[0-9]+$ ]]; do
    echo "${epoch_number} is not a number. Please try again."
    read -r epoch_number
  done
  while [[ -f ${param_dir}epochs_fors2/${param_file}.yaml ]]; do
    echo "${frb_name}_${epoch_number} already exists. Please try again."
    read -r epoch_number
  done

  frb_dir="${top_data_dir}${frb_name}/"
  if ! [[ -d ${frb_dir} ]]; then
    echo "This seems to be the first epoch processed for this FRB. Setting up directories at ${frb_dir}:"
    mkdir "${frb_dir}"
    epoch_number=1
  fi
  if ! [[ -d "${frb_dir}FORS2" ]]; then
    mkdir "${frb_dir}FORS2"
  fi
  if ! [[ -f "${param_dir}FRBs/${frb_name}.yaml" ]]; then
    cp "${proj_dir}param/FRBs/FRB_template.yaml" "${param_dir}FRBs/${frb_name}.yaml"
    python3 pipeline_frb/0-new_frb.py --op "${frb_name}"
    echo "No FRB param file found; I have created a new one at ${param_dir}FRBs/${frb_name}.yaml, with some default values. Please check this file before proceeding."
  fi
  mkdir "${frb_dir}FORS2/new_epoch"
  shopt -s nullglob
  echo "Looking for download scripts..."
  options=("Quit" "Enter path manually")
  options+=(~/Downloads/*download*.sh)
  options+=("${top_data_dir}"/*download*.sh)
  options+=("${top_data_dir}${frb_name}"/*download*.sh)

  echo "Select an option:"
  select script_path in "${options[@]}"; do
    if ((REPLY == 1)); then
      exit
    elif ((REPLY == 2)); then
      echo "Please enter the path to the ESO download script:"
      read -r script_path
      while [[ ${script_path} != *download*.sh ]]; do
        echo "This is not a valid script file. Try again:"
        read -r script_path
      done

      while [[ ! -f ${script_path} ]]; do
        echo "Path does not exist. Try again:"
        read -r script_path
      done
      break

    elif ((REPLY > 0 && REPLY <= ${#options[@]})); then
      break

    else
      echo "Invalid option. Try another one."
    fi
  done

  echo "You have selected ${script_path}"

  param_file="${frb_name}_${epoch_number}"
  cp "${proj_dir}param/epochs_fors2/FRB_fors2_epoch_template.yaml" "${param_dir}epochs_fors2/${param_file}.yaml"
  python3 pipeline_fors2/0-new_epoch.py --op "${param_file}"
  if ! [[ -d "${frb_dir}FORS2/new_epoch/0-data_with_raw_calibs/" ]]; then
    mkdir "${frb_dir}FORS2/new_epoch/0-data_with_raw_calibs/"
  fi
  cp "${script_path}" "${frb_dir}FORS2/new_epoch/0-data_with_raw_calibs/download${param_file}script.sh"
  echo "I have created a new epoch parameter file at ${param_dir}epochs_fors2/${param_file}.yaml, with some default values. Please check this file before proceeding."
fi

if ! python3 "refresh_params.py"; then
  echo "Something went wrong with reading or writing the param files."
  exit
fi

echo "Checking for epoch parameters at ${param_dir}epochs_fors2/${param_file}.json"

# Special thanks to https://stackoverflow.com/questions/15807845/list-files-and-show-them-in-a-menu-with-bash

if ! [[ -f "${param_dir}epochs_fors2/${param_file}.json" ]]; then
  echo "Epoch parameter file not found; checking for FRB parameter file."
  frb_name=${param_file::-2}
  if [[ -f "${param_dir}FRBs/${frb_name}.json" ]]; then
    echo
    echo "Specify epoch parameter file:"
    options=("${param_dir}epochs_fors2/${frb_name}_"?".json")
    select opt in "${options[@]}" "Quit"; do
      if ((REPLY == 1 + ${#options[@]})); then
        exit

      elif ((REPLY > 0 && REPLY <= ${#options[@]})); then
        param_file=${opt: -16:11}
        echo "You have selected epoch ${param_file}."

        break

      else
        echo "Invalid option. Try another one."
      fi
    done
  else
    echo "No parameter file matching ${param_file}.json found. Exiting."
    exit
  fi
fi

data_dir=$(jq -r .data_dir "${param_dir}epochs_fors2/${param_file}.json")

run_script() {
  script=$1
  extra_message=$2
  echo ""
  echo "Run ${script}? ${extra_message}"
  select yn in "Yes" "Skip" "Exit"; do
    case ${yn} in
    Yes)
      if "${proj_dir}/pipeline_fors2/${script}.sh" "${param_file}"; then
        break
      else
        echo "Something went wrong. Try again?"
        echo "1) Yes"
        echo "2) Skip"
        echo "3) Exit"
      fi
      ;;
    Skip) break ;;
    Exit) exit ;;
    esac
  done
}

run_script_folders() {
  script=$1
  extra_message=$2
  origin=$3
  destination=$4
  other_arguments=$5
  logfile="${data_dir}${folder}${script}$(date +%Y-%m-%dT%T).log"
  echo ""
  echo "Run ${script}? ${extra_message}"
  select yn in "Yes" "Skip" "Exit"; do
    case ${yn} in
    Yes)
      if "${proj_dir}/pipeline_fors2/${script}.sh" "${param_file}" "${origin}" "${destination}" "${other_arguments}"; then # | tee "${logfile}"; then
        break
      else
        echo "Something went wrong. Try again?"
        echo "1) Yes"
        echo "2) Skip"
        echo "3) Exit"
      fi
      ;;
    Skip) break ;;
    Exit) exit ;;
    esac
  done
}

run_python() {
  script=$1
  extra_message=$2
  origin=$3
  destination=$4
  logfile="${data_dir}${folder}${script}$(date +%Y-%m-%dT%T).log"
  echo ""
  echo "Run ${script}? ${extra_message}"
  select yn in "Yes" "Skip" "Exit"; do
    case ${yn} in
    Yes)
      if python3 "${proj_dir}/pipeline_fors2/${script}.py" --op "${param_file}" --origin "${origin}" --destination "${destination}"; then
        break
      else
        echo "Something went wrong. Try again?"
        echo "1) Yes"
        echo "2) Skip"
        echo "3) Exit"
      fi
      ;;
    Skip) break ;;
    Exit) exit ;;
    esac
  done
}

if [[ ${folder} != "" ]]; then
  if [[ ${folder: -1} != "/" ]]; then
    folder="${folder}/"
  fi
  mkdir "${data_dir}${folder}"
fi

run_script_folders 1-initial ''
run_script_folders 2-sort_after_esoreflex 'Requires reducing data with ESOReflex first.'
run_script_folders 3-trim ''
run_script_folders 4-divide_by_exp_time ''

individuals="4-divided_by_exp_time/"

if ${insert_test_synth}; then
  individuals="${folder}4.1-test_synth_inserted/"
  run_python 4.1-insert_test_synth '' "4-divided_by_exp_time/" "${individuals}"
fi

if ${sub_back}; then
  run_script_folders 5-background_subtract '' "${individuals}" "${folder}5-background_subtracted_with_python/"
  run_script_folders 6-montage '(Science images)' "${folder}5-background_subtracted_with_python/science/" "${folder}6-combined_with_montage/science/"
  run_script_folders 6-montage '(Background images)' "${folder}5-background_subtracted_with_python/backgrounds/" "${folder}6-combined_with_montage/backgrounds/"
elif ${insert_test_synth}; then
  run_script_folders 6-montage '' "${individuals}" "${folder}6-combined_with_montage/science/"
else
  run_script_folders 6-montage '' "${folder}4-divided_by_exp_time/science/" "${folder}6-combined_with_montage/science/"
fi

run_script_folders 7-trim_combined '' "${folder}6-combined_with_montage/science/" "${folder}7-trimmed_again/" "${folder}"
run_script_folders 8-astrometry '' "${folder}7-trimmed_again/" "${folder}8-astrometry/"
if compgen -G "${data_dir}${folder}8-astrometry/*_astrometry.fits" >/dev/null; then
  zp_origin="${folder}8-astrometry/"
else
  zp_origin="${folder}7-trimmed_again/"
fi
run_script_folders 9-zeropoint '' "${zp_origin}" "${folder}9-zeropoint/" "${folder}"

if ! ${sub_back}; then
  run_python insert_synthetic_range_at_frb
  run_python insert_synthetic_sn_random_ia
fi
