#!/usr/bin/env bash
# Code by Lachlan Marnoch, 2019
param_file=$1

if [[ -z ${param_file} ]]; then
  echo "No object specified."
  exit
fi

proj_param_file=$2

if [[ -z ${proj_param_file} ]]; then
  proj_param_file=unicomp
fi

if ! python3 scripts/params.py -op "${param_file}" -pp "${proj_param_file}"; then
  echo "Something went wrong with reading or writing the param files."
fi

proj_dir=$(jq -r .proj_dir "param/project/${proj_param_file}.json")

export PYTHONPATH=PYTHONPATH:${proj_dir}

run_bash() {
  script=$1
  extra_message=$2
  echo ""
  echo "Run ${script}? ${extra_message}"
  select yn in "Yes" "Skip" "Exit"; do
    case ${yn} in
    Yes)
      if "${proj_dir}scripts/pipeline_xshooter/${script}.sh" "${param_file}" "${proj_param_file}"; then
        break
      else
        echo "Something went wrong. Try again?"
        echo "1) Yes"
        echo "2) Skip"
        echo "3) Exit"
      fi
      ;;
    Skip) break ;;
    Exit) exit ;;
    esac
  done
}

run_python() {
  script=$1
  extra_message=$2
  echo ""
  echo "Run ${script}? ${extra_message}"
  select yn in "Yes" "Skip" "Exit"; do
    case ${yn} in
    Yes)
      if python3 "${proj_dir}scripts/pipeline_xshooter/${script}.py" --op "${param_file}"; then
        break
      else
        echo "Something went wrong. Try again?"
        echo "1) Yes"
        echo "2) Skip"
        echo "3) Exit"
      fi
      ;;
    Skip) break ;;
    Exit) exit ;;
    esac
  done
}

run_bash 1-initial
run_python 2-reduce
run_python 3-defringe
run_python 4-trim
run_python 5-divide_by_exp_time
run_bash 6-montage
run_bash 7-sextractor
run_python 8-astrometry
run_bash 9-zeropoint
