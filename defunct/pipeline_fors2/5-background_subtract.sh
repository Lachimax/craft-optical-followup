#!/usr/bin/env bash

# Code by Lachlan Marnoch, 2019

param_file=$1
origin=$2
if [[ -z ${origin} ]]; then
  origin=4-divided_by_exp_time/
fi

destination=$3
if [[ -z ${destination} ]]; then
  destination=B-back_subtract/5-background_subtracted_with_python/
fi

all_synths=$4
if [[ -z ${all_synths} ]]; then
  all_synths=false
fi

echo
echo "Executing bash script pipeline_fors2/5-background_subtract.sh, with:"
echo "   epoch ${param_file}"
echo "   origin directory ${origin}"
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

mkdir "${data_dir}${destination}"
mkdir "${data_dir}${destination}science"

if ${all_synths}; then
  python3 "${proj_dir}/pipeline_fors2/5-background_subtract.py" --directory "${data_dir}" --op "${data_title}" --origin "${origin}" --destination "${destination}" --all_synths || exit 1
else
  python3 "${proj_dir}/pipeline_fors2/5-background_subtract.py" --directory "${data_dir}" --op "${data_title}" --origin "${origin}" --destination "${destination}" || exit 1
fi
