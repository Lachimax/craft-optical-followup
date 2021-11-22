#!/usr/bin/env bash

param_file=$1
config_file="param/config.json"

echo
echo "Executing bash script pipeline_fors2/3-trim.sh, with epoch ${param_file}"
echo

if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi

param_dir=$(jq -r .param_dir "${config_file}")
proj_dir=$(jq -r .proj_dir "${config_file}")

data_dir=$(jq -r .data_dir "${param_dir}/epochs_fors2/${param_file}.json")
data_title=${param_file}

python3 "${proj_dir}/pipeline_fors2/3-trim.py" --origin "${data_dir}/2-sorted/" --destination "${data_dir}/3-trimmed_with_python/" -op "${data_title}"