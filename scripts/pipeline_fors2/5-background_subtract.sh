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

config_file="param/config.json"
if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi

param_dir=$(jq -r .param_dir "${config_file}")

data_dir=$(jq -r .data_dir "${proj_dir}/epochs_fors2/${param_file}.json")
data_title=$(jq -r .data_title "${proj_dir}/epochs_fors2/${param_file}.json")

#if ${do_sextractor} ; then
#    python3 ${proj_dir}/scripts/pipeline_fors2/5-background_subtract.py --directory ${data_dir} -op ${data_title} --sextractor_directory ${data_dir}/analysis/sextractor/individuals_back_subtracted/
#
#    mkdir ${data_dir}/analysis/sextractor/individuals_back_subtracted/
#    if cd ${data_dir}/analysis/sextractor/individuals_back_subtracted/ ; then
#        for fil in $(ls -d **/) ; do
#            if cd ${fil} ; then
#                if cp ${proj_dir}/param/sextractor/default/* . ; then
#                    for sextract in $(ls sextract*.sh) ; do
#                        chmod u+x ${sextract}
#                        ./${sextract}
#                    done
#                    mkdir ${data_dir}/analysis/sextractor/ind_selected/${fil}
#                    cp aperture_${aperture_diam}/* ${data_dir}/analysis/sextractor/individuals_back_subtracted/${fil}
#                fi
#                cd ..
#            fi
#        done
#    else exit
#    fi
#    cp ${data_dir}/3-trimmed_with_python/${data_title}.log ${data_dir}/analysis/sextractor/individuals_back_subtracted/
#    date +%Y-%m-%dT%T >> ${data_dir}/analysis/sextractor/individuals_back_subtracted/${data_title}.log
#    echo Sextracted >> ${data_dir}/analysis/sextractor/individuals_back_subtracted/${data_title}.log
#    echo "All done."
#
#else
#    python3 ${proj_dir}/scripts/pipeline_fors2/5-background_subtract.py --directory ${data_dir} -op ${data_title}
#fi

python3 "${proj_dir}/scripts/pipeline_fors2/5-background_subtract.py" --directory "${data_dir}" --op "${data_title}" --origin "${origin}" --destination "${destination}"