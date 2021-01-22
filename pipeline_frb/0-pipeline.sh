#!/usr/bin/env bash
# Code by Lachlan Marnoch, 2021

frb_name=$1

echo
echo "Executing pipeline_fors2/0-pipeline.sh, with epoch ${frb_name}."
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

# If no FRB specified at run, ask the user for one.
if [[ -z ${frb_name} ]]; then
  echo "No FRB specified. Please enter FRB name."
  read -r frb_name
  while ! { [[ ${frb_name} =~ ^FRB[0-9]{6}$ ]] || [[ ${frb_name} =~ ^[0-9]{6}$ ]]; }; do
    echo "Please format response as FRBXXXXXX or as XXXXXX; eg FRB180924, or 180924."
    read -r frb_name
  done
  if [[ ${frb_name} =~ ^[0-9]{6} ]]; then
    frb_name="FRB${frb_name}"
  fi
fi

# Set up the FRB data directory.
frb_dir="${top_data_dir}${frb_name}/"
if ! [[ -d ${frb_dir} ]]; then
  echo "Setting up directories at ${frb_dir}:"
  mkdir "${frb_dir}"
fi

# If the param file doesn't exist, make a new one from template.
if ! [[ -f "${param_dir}FRBs/${frb_name}.yaml" ]]; then
  cp "${proj_dir}param/FRBs/FRB_template.yaml" "${param_dir}FRBs/${frb_name}.yaml"
  echo "No FRB param file found; I have created a new one at ${param_dir}FRBs/${frb_name}.yaml, with some default values. Please check this file before proceeding."
  python3 pipeline_fors2/0-new_frb.py --op "${frb_name}"
fi

run_script() {
  script=$1
  extra_message=$2
  echo ""
  echo "${extra_message}"
  select yn in "Yes" "Skip" "Exit"; do
    case ${yn} in
    Yes)
      if bash "${proj_dir}/pipeline_fors2/${script}.sh" "${frb_name}"; then
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
  echo "${extra_message}"
  select yn in "Yes" "Skip" "Exit"; do
    case ${yn} in
    Yes)
      if python3 "${proj_dir}/pipeline_frb/${script}.py" --op "${frb_name}"; then
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

run_python 1-check_catalogues "Check for catalogue data at the FRB position?"
run_python 2-look_for_host "Attempt to identify host galaxy?"
