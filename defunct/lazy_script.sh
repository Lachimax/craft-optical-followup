#!/usr/bin/env bash
# Code by Lachlan Marnoch, 2019
param_file=$1

if [[ -z ${param_file} ]]; then
    echo "No object specified."
    exit
fi
burst=${param_file::-2}

case=$2
if [[ -z ${case} ]]; then
    case=1
fi

n=$3
if [[ -z ${n} ]]; then
    n=100
fi

sn_type=$4
if [[ -z ${sn_type} ]]; then
    sn_type=iil
fi

proj_param_file=$5
if [[ -z ${proj_param_file} ]]; then
    proj_param_file=unicomp
fi

epoch=$6
if [[ -z ${epoch} ]]; then
    epoch=1
fi

instrument=$7
if [[ -z ${instrument} ]]; then
  if [[ ${epoch} == 0 ]]; then
    instrument=DES
  else
    instrument=FORS2
  fi
fi

instrument_template=$8
if [[ -z ${instrument_template} ]]; then
    instrument_template=FORS2
fi



proj_dir=$(jq -r .proj_dir param/project/${proj_param_file}.json)
data_dir=$(jq -r .data_dir param/epochs_${instrument,,}/${param_file}.json)

run_python () {
    script=$1
    python3 "${proj_dir}/pipeline_fors2/${script}.py" --op "${param_file}"

}

echo ${case}
if [[ ${case} == 1 ]] ; then
    run_python insert_synthetic_range_at_frb
    ${proj_dir}/subtraction/0-subtraction.sh ${burst} multi_frb_range ${proj_param_file} ${epoch} ${instrument} ${instrument_template} true
elif [[ ${case} == 2 ]] ; then
    python3 "${proj_dir}/pipeline_fors2/insert_synthetic_sn_random_ia.py" --op "${param_file}" --n ${n} --instrument ${instrument}
    ${proj_dir}/subtraction/0-subtraction.sh ${burst} multi_sn_random_ia ${proj_param_file} ${epoch} ${instrument} ${instrument_template} true
elif [[ ${case} == 3 ]] ; then
    python3 "${proj_dir}/pipeline_fors2/insert_synthetic_sn_random.py" --op "${param_file}" --sn_type ${sn_type} --n ${n}
    ${proj_dir}/subtraction/0-subtraction.sh ${burst} multi_sn_random_${sn_type} ${proj_param_file} ${epoch} ${instrument} ${instrument_template} true
elif [[ ${case} == 4 ]] ; then
    python3 "${proj_dir}/pipeline_fors2/insert_synthetic_sn_random_ia_fix_position.py" --op "${param_file}" --n ${n} --instrument ${instrument}
    ${proj_dir}/no_subtraction_synth_search.sh ${burst} multi_sn_random_ia ${proj_param_file} ${epoch} ${instrument}
elif [[ ${case} == 5 ]] ; then
    python3 "${proj_dir}/pipeline_fors2/insert_synthetic_sn_random_fix_position.py" --op "${param_file}" --sn_type ${sn_type} --n ${n} --instrument ${instrument}
    ${proj_dir}/no_subtraction_synth_search.sh ${burst} multi_sn_random_${sn_type} ${proj_param_file} ${epoch} ${instrument}
elif [[ ${case} == 6 ]] ; then
    python3 "${proj_dir}/pipeline_fors2/insert_synthetic_sn_random_ia.py" --op "${param_file}" --n ${n} --instrument ${instrument}
    ${proj_dir}/no_subtraction_synth_search.sh ${burst} multi_sn_random_ia ${proj_param_file} ${epoch} ${instrument}
elif [[ ${case} == 7 ]] ; then
    python3 "${proj_dir}/pipeline_fors2/insert_synthetic_sn_random.py" --op "${param_file}" --sn_type ${sn_type} --n ${n} --instrument ${instrument}
    ${proj_dir}/no_subtraction_synth_search.sh ${burst} multi_sn_random_${sn_type} ${proj_param_file} ${epoch} ${instrument} ${instrument_template} true
fi
