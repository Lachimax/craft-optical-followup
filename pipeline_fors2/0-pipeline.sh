#!/usr/bin/env bash

param_file=$1

if [[ -z ${param_file} ]]; then
    echo "No object specified."
    exit
fi

sub_back=$2

if [[ -z ${sub_back} ]]; then
    sub_back=false
fi

if ! python3 "refresh_params.py"; then
  echo "Something went wrong with reading or writing the param files."
  exit
fi

config_file="param/config.json"

if ! proj_dir=$(jq -r .proj_dir ${config_file}); then
  echo "Configuration file not found."
  exit
fi

param_dir=$(jq -r .param_dir "${config_file}")

echo "Loading epoch parameters from ${param_dir}/epochs_fors2/${param_file}.json"

if ! data_dir=$(jq -r .data_dir "${param_dir}/epochs_fors2/${param_file}.json"); then
  echo "Epoch parameter file not found."
  exit
fi

export PYTHONPATH=PYTHONPATH:${proj_dir}

run_script () {
    script=$1
    extra_message=$2
    echo ""
    echo "Run ${script}? ${extra_message}"
    select yn in "Yes" "Skip" "Exit"; do
      case ${yn} in
          Yes )
              if "${proj_dir}scripts/pipeline_fors2/${script}.sh" "${param_file}"; then
                  break;
              else
                  echo "Something went wrong. Try again?"
                  echo "1) Yes"
                  echo "2) Skip"
                  echo "3) Exit"
              fi;;
          Skip ) break;;
          Exit ) exit;;
      esac
    done
}

run_script_folders () {
    script=$1
    extra_message=$2
    origin=$3
    destination=$4
    echo ""
    echo "Run ${script}? ${extra_message}"
    select yn in "Yes" "Skip" "Exit"; do
      case ${yn} in
          Yes )
              if "${proj_dir}scripts/pipeline_fors2/${script}.sh" "${param_file}" "${origin}" "${destination}"; then
                  break;
              else
                  echo "Something went wrong. Try again?"
                  echo "1) Yes"
                  echo "2) Skip"
                  echo "3) Exit"
              fi;;
          Skip ) break;;
          Exit ) exit;;
      esac
    done
}

run_python () {
    script=$1
    extra_message=$2
    echo ""
    echo "Run ${script}? ${extra_message}"
    select yn in "Yes" "Skip" "Exit"; do
    case ${yn} in
        Yes )
            if python3 "${proj_dir}scripts/pipeline_fors2/${script}.py" --op "${param_file}"; then
                break;
            else
                echo "Something went wrong. Try again?"
                echo "1) Yes"
                echo "2) Skip"
                echo "3) Exit"
            fi;;
        Skip ) break;;
        Exit ) exit;;
    esac
done
}

run_script 1-initial
run_script 2-sort_after_esoreflex 'Requires reducing data with ESOReflex first.'
run_script 3-trim
run_script 4-divide_by_exp_time

if ${sub_back} ; then
    folder=B-back_subtract/
else
    folder=""
fi

mkdir "${data_dir}${folder}"

if ${sub_back} ; then
    cp -r "${data_dir}4-divided_by_exp_time" "${data_dir}${folder}4-divided_by_exp_time"
    run_script_folders 5-background_subtract '' ${folder}4-divided_by_exp_time/
    run_script_folders 6-montage '' ${folder}5-background_subtracted_with_python/ ${folder}6-combined_with_montage/
else
    run_script_folders 6-montage '' ${folder}4-divided_by_exp_time/science/ ${folder}6-combined_with_montage/
fi

run_script_folders 7-trim_combined '' ${folder}6-combined_with_montage/ ${folder}7-trimmed_again/
run_script_folders 8-astrometry '' ${folder}7-trimmed_again/ ${folder}8-astrometry/
run_script_folders 9-zeropoint '' ${folder}8-astrometry/ ${folder}

if ! ${sub_back} ; then
    run_python insert_synthetic_range_at_frb
    run_python insert_synthetic_sn_random_ia
fi