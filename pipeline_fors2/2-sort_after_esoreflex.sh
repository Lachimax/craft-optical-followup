#!/usr/bin/env bash
# Script by Lachlan Marnoch 2019
# For sorting the products of the FORS ESO pipeline.

param_file=$1

config_file="param/config.json"

if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi

param_dir=$(jq -r .param_dir "${config_file}")
esoreflex_input_dir=$(jq -r .esoreflex_input_dir "${config_file}")
esoreflex_output_dir=$(jq -r .esoreflex_output_dir "${config_file}")

data_dir=$(jq -r .data_dir "${param_dir}/epochs_fors2/${param_file}.json")
data_title=${param_file}
skip_copy=$(jq -r .skip_copy "${param_dir}/epochs_fors2/${param_file}.json")

object=$(jq -r .obs_name "${data_dir}/output_values.json")

destination=${data_dir}/1-reduced_with_esoreflex
mkdir "${destination}"

cd "${destination}" || exit
origin_global=${destination}
destination=${data_dir}/2-sorted
mkdir "${destination}"

cd "${proj_dir}" || exit

python3 "${proj_dir}/pipeline_fors2/2-sort_after_esoreflex.py" --op "${data_title}" --directory "${data_dir}/2-sorted/" || exit

pwd

cp "${origin_global}/${data_title}.log" "${destination}/"
date +%Y-%m-%dT%T >>"${destination}/${data_title}.log"
echo Files sorted using sort_after_esoreflex.sh >>"${destination}/${data_title}.log"
echo "All done."
