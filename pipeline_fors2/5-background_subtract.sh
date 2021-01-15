#!/usr/bin/env bash

# Code by Lachlan Marnoch, 2019


param_file=$1
origin=$2
if [[ -z ${origin} ]]; then
    origin=4-divided_by_exp_time/
fi

destination=$3
if [[ -z ${destination} ]]; then
    destination=5-background_subtracted_with_python/
fi

normalise=$4
if [[ -z ${normalise} ]]; then
    normalise=true
fi

echo
echo "Executing bash script pipeline_fors2/5-background_subtract.sh, with:"
echo "   epoch ${param_file}"
echo "   origin directory ${origin}"
echo "   destination directory ${destination}"
echo "   normalise ${normalise}"
echo

config_file="param/config.json"
if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi

data_dir=$(jq -r .data_dir "${proj_dir}/epochs_fors2/${param_file}.json")
data_title=${param_file}

python3 "${proj_dir}/pipeline_fors2/5-background_subtract.py" --directory "${data_dir}" --op "${data_title}" --origin "${origin}" --destination "${destination}"