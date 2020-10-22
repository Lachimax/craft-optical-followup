#!/usr/bin/env bash

param_file=$1
config_file="param/config.json"

if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi

param_dir=$(jq -r .param_dir "${config_file}")
proj_dir=$(jq -r .proj_dir "${config_file}")

data_dir=$(jq -r .data_dir "${param_dir}/epochs_fors2/${param_file}.json")
data_title=$(jq -r .data_title "${param_dir}/epochs_fors2/${param_file}.json")
do_sextractor=$(jq -r .do_sextractor "${param_dir}/epochs_fors2/${param_file}.json")

#if ${do_sextractor} ; then
#    python3 ${proj_dir}/scripts/pipeline_fors2/3-trim.py --origin ${data_dir}/2-sorted/ --destination ${data_dir}/3-trimmed_with_python/ -op ${data_title} --sextractor_directory ${data_dir}/analysis/sextractor/individuals_trimmed/
#
#    if cd ${data_dir}/analysis/sextractor/individuals_trimmed/ ; then
#        pwd
#        for fil in $(ls -d **/) ; do
#            if cd ${fil} ; then
#                if cp ${proj_dir}/param/sextractor/default/* . ; then
#                    for sextract in $(ls sextract*.sh) ; do
#                        chmod u+x ${sextract}
#                        ./${sextract}
#                    done
#                fi
#                cd ..
#            fi
#        done
#        cp ${data_dir}/3-trimmed_with_python/${data_title}.log ${data_dir}/analysis/sextractor/individuals_trimmed/
#        date +%Y-%m-%dT%T >> ${data_dir}/analysis/sextractor/individuals_trimmed/${data_title}.log
#        echo Sextracted >> ${data_dir}/analysis/sextractor/individuals_trimmed//${data_title}.log
#        echo "All done."
#    else echo 'Error'
#    fi
#else
#    python3 ${proj_dir}/scripts/pipeline_fors2/3-trim.py --origin ${data_dir}/2-sorted/ --destination ${data_dir}/3-trimmed_with_python/ -op ${data_title}
#fi

python3 "${proj_dir}/scripts/pipeline_fors2/3-trim.py" --origin "${data_dir}/2-sorted/" --destination "${data_dir}/3-trimmed_with_python/" -op "${data_title}"