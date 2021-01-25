#!/usr/bin/env bash
# Code by Lachlan Marnoch, 2019 - 2021

param_file=$1

if [[ -z ${param_file} ]]; then
  echo "No epoch specified."
  exit
fi

sub_back=$2

if [[ -z ${sub_back} ]]; then
  sub_back=false
fi

echo
echo "Executing pipeline_fors2/0-pipeline.sh, with epoch ${param_file} and sub_back=${sub_back}"
echo

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
  if ! [[ -d ${frb_dir} ]] ; then
    echo "This seems to be the first epoch processed for this FRB. Setting up directories at ${frb_dir}:"
    mkdir "${frb_dir}"
    epoch_number=1
  fi
  if ! [[ -d "${frb_dir}FORS2" ]] ; then
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
  echo ""
  echo "Run ${script}? ${extra_message}"
  select yn in "Yes" "Skip" "Exit"; do
    case ${yn} in
    Yes)
      if "${proj_dir}/pipeline_fors2/${script}.sh" "${param_file}" "${origin}" "${destination}"; then
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
  echo ""
  echo "Run ${script}? ${extra_message}"
  select yn in "Yes" "Skip" "Exit"; do
    case ${yn} in
    Yes)
      if python3 "${proj_dir}/pipeline_fors2/${script}.py" --op "${param_file}"; then
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

run_script 1-initial
run_script 2-sort_after_esoreflex 'Requires reducing data with ESOReflex first.'
run_script 3-trim
run_script 4-divide_by_exp_time

if ${sub_back}; then
  folder=B-back_subtract/
else
  folder=""
fi

if ! [[ -d "${data_dir}${folder}" ]]; then
  mkdir "${data_dir}${folder}"
fi

if ${sub_back}; then
  run_script_folders 5-background_subtract '' "4-divided_by_exp_time/"
  run_script_folders 6-montage '' "${folder}5-background_subtracted_with_python/" "${folder}6-combined_with_montage/"
else
  run_script_folders 6-montage '' "${folder}4-divided_by_exp_time/science/" "${folder}6-combined_with_montage/"
fi

run_script_folders 7-trim_combined '' "${folder}6-combined_with_montage/" "${folder}7-trimmed_again/"
run_script_folders 8-astrometry '' "${folder}7-trimmed_again/" "${folder}8-astrometry/"
if compgen -G "${data_dir}${folder}8-astrometry/*_astrometry.fits" > /dev/null ; then
  zp_origin="${folder}8-astrometry/"
else
  zp_origin="${folder}7-trimmed_again/"
fi
run_script_folders 9-zeropoint '' "${zp_origin}" "${folder}"

if ! ${sub_back}; then
  run_python insert_synthetic_range_at_frb
  run_python insert_synthetic_sn_random_ia
fi