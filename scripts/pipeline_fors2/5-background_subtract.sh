#!/usr/bin/env bash

# Code by Lachlan Marnoch, 2019

param_file=$1
proj_param_file=$2
origin=$3
destination=$4
normalise=$5

if [[ -z ${proj_param_file} ]]; then
    proj_param_file=unicomp
fi
if [[ -z ${origin} ]]; then
    origin=4-divided_by_exp_time/
fi
if [[ -z ${destination} ]]; then
    destination=5-background_subtracted_with_python/
fi
if [[ -z ${normalise} ]]; then
    normalise=true
fi

proj_dir=$(jq -r .proj_dir "param/project/${proj_param_file}.json")

data_dir=$(jq -r .data_dir "param/epochs_fors2/${param_file}.json")
data_title=$(jq -r .data_title "param/epochs_fors2/${param_file}.json")

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