#!/usr/bin/env bash
# Script by Lachlan Marnoch 2019
# For sorting the products of the FORS ESO pipeline.

param_file=$1

echo
echo "Executing pipeline_fors2/2-sort_after_esoreflex.sh, with epoch ${param_file}"
echo

config_file="param/config.json"

if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi

param_dir=$(jq -r .param_dir "${config_file}")

data_dir=$(jq -r .data_dir "${param_dir}/epochs_fors2/${param_file}.json")
data_title=${param_file}
skip_copy=$(jq -r .skip_copy "${param_dir}/epochs_fors2/${param_file}.json")

destination=${data_dir}/2-sorted
mkdir "${destination}"

cd "${proj_dir}" || exit

if ! ${skip_copy}; then
  python3 "${proj_dir}/pipeline_fors2/2-sort_after_esoreflex.py" --op "${data_title}" || exit
fi
