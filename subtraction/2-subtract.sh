#!/usr/bin/env bash
# Code by Lachlan Marnoch, 2019
param_file=$1
destination=$2
proj_param_file=$3
type=$4
epoch=$5
instrument=$6
template_instrument=$7
force=$8

if [[ -z ${proj_param_file} ]]; then
    proj_param_file=unicomp
fi

if [[ -z ${epoch} ]]; then
    epoch=1
fi

if [[ -z ${instrument} ]]; then
    instrument=FORS2
fi

if [[ -z ${type} ]]; then
    type=normal
fi

data_dir=$(jq -r .data_dir "param/FRBs/${param_file}.json")
proj_dir=$(jq -r .proj_dir "param/project/${proj_param_file}.json")
if [[ -z ${destination} ]]; then
    destination=${epoch}_${instrument}_${type}/
fi

if [[ -z ${force} ]]; then
    force=false
fi

if [[ ${type::5} == multi ]] ; then
    cd ${data_dir}subtraction/${destination} || exit
    for test in **/ ; do
        cd ${proj_dir}
        python3 scripts/subtraction/2-subtract.py --field "${param_file}" --destination "${destination}${test}/" --epoch "${epoch}" --instrument "${instrument}" --instrument_template "${template_instrument}" --type "${type}"
    done
else
    if ${force} ; then
        echo "python3 scripts/subtraction/2-subtract.py --field ${param_file} --destination ${destination} --epoch ${epoch} --instrument ${instrument} --instrument_template ${template_instrument} --type ${type} -force_subtract_better_seeing"
        python3 scripts/subtraction/2-subtract.py --field "${param_file}" --destination ${destination} --epoch ${epoch} --instrument ${instrument} --instrument_template "${template_instrument}" --type ${type} -force_subtract_better_seeing
    else
        echo "python3 scripts/subtraction/2-subtract.py --field ${param_file} --destination ${destination} --epoch ${epoch} --instrument ${instrument} --instrument_template ${template_instrument} --type ${type}"
        python3 scripts/subtraction/2-subtract.py --field "${param_file}" --destination ${destination} --epoch ${epoch} --instrument ${instrument} --instrument_template "${template_instrument}" --type ${type}
    fi
fi