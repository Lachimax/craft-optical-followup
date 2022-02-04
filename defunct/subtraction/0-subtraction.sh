#!/usr/bin/env bash
# Code by Lachlan Marnoch, 2019
param_file=$1
if [[ -z ${param_file} ]]; then
    echo "No object specified."
    exit
fi

type=$2
multi=false
if [[ -z ${type} ]]; then
    type=normal
fi

if [[ ${type} != normal ]] ; then
    if [[ ${type::5} == multi ]] ; then
        multi=true
    fi
fi

proj_param_file=$3
if [[ -z ${proj_param_file} ]]; then
    proj_param_file=unicomp
fi

epoch=$4
if [[ -z ${epoch} ]]; then
    epoch=1
fi

instrument=$5
if [[ -z ${instrument} ]]; then
  if [[ ${epoch} == 0 ]]; then
    instrument=DES
  else
    instrument=FORS2
  fi
fi

template_instrument=$6
if [[ -z ${template_instrument} ]]; then
    template_instrument=FORS2
fi
template_epoch=$(jq -r .template_epoch_${template_instrument,,} "param/FRBs/${param_file}.json")

automate=$7
if [[ -z ${automate} ]]; then
    automate=false
fi

destination=$8
manual=$9
if [[ -z ${manual} ]]; then
    manual=false
fi
echo Manual
echo ${manual}

#if ! python3 /refresh_params.py -op "${param_file}" -pp "${proj_param_file}"; then
#    echo "Something went wrong with reading or writing the param files."
#fi



if [[ -z ${destination} ]]; then
    destination=${instrument}_${epoch}-${template_instrument}_${template_epoch}_${type}/
fi
if [[ ${destination::-1} != / ]] ; then
    destination=${destination}/
fi

proj_dir=$(jq -r .proj_dir "param/project/${proj_param_file}.json")

export PYTHONPATH=PYTHONPATH:${proj_dir}

run_bash () {
    script=$1
    extra_argument=$2
    extra_message=$3
    echo ""
    echo "Run ${script}? ${extra_message}"
    select yn in "Yes" "Skip" "Exit"; do
    case ${yn} in
        Yes )
            if "${proj_dir}/subtraction/${script}.sh" "${param_file}" "${destination}" "${proj_param_file}" "${type}" "${epoch}" "${instrument}" "${template_instrument}" "${extra_argument}" ; then
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

run_bash_lie () {
    script=$1
    extra_message=$2
    echo ""
    echo "Run ${script} in deceit mode? ${extra_message}"
    select yn in "Yes" "Skip" "Exit"; do
    case ${yn} in
        Yes )
            if "${proj_dir}/subtraction/${script}.sh" "${param_file}" "${instrument}_${epoch}-${template_instrument}_${template_epoch}_normal/" "${proj_param_file}" "normal" "${epoch}" "${instrument}" ; then
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
            if python3 "${proj_dir}/subtraction/${script}.py" --field "${param_file}" --subtraction_path "${destination}" --epoch "${epoch}" --instrument "${instrument}" --instrument_template "${template_instrument}"; then
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


if ${automate}; then
    if ${multi} ; then
        ${proj_dir}/subtraction/1-align.sh "${param_file}" "${instrument}_${epoch}-${template_instrument}_${template_epoch}_normal/" "${proj_param_file}" "normal" "${epoch}" "${instrument}" "${template_instrument}"
    fi
    ${proj_dir}/subtraction/1-align.sh "${param_file}" "${destination}" "${proj_param_file}" "${type}" "${epoch}" "${instrument}" "${template_instrument}"
    ${proj_dir}/subtraction/2-subtract.sh "${param_file}" "${destination}" "${proj_param_file}" "${type}" "${epoch}" "${instrument}" "${template_instrument}"
    ${proj_dir}/subtraction/3-sextractor.sh "${param_file}" "${destination}" "${proj_param_file}" "${type}" "${epoch}" "${instrument}" "${template_instrument}"
    if [[ ${type} != normal ]] ; then
        if ${multi} ; then
            python3 "${proj_dir}/subtraction/4-recover_synthetics_multi.py" --field "${param_file}" --subtraction_path "${destination}" --epoch "${epoch}" --instrument "${instrument}" --instrument_template "${template_instrument}"
        else
            python3 "${proj_dir}/subtraction/4-recover_synthetics.py" --field "${param_file}" --subtraction_path "${destination}" --epoch "${epoch}" --instrument "${instrument}"
        fi
    else
      python3 "${proj_dir}/subtraction/4-find_transient.py" --field "${param_file}" --subtraction_path "${destination}" --epoch "${epoch}" --instrument "${instrument}"
    fi

else
    if ${multi} ; then
        run_bash_lie 1-align
    fi
    run_bash 1-align
    run_bash 2-subtract
    run_bash 3-sextractor
    if [[ ${type} != normal ]] ; then
        if ${multi} ; then
            run_python 4-recover_synthetics_multi
        else
            run_python 4-recover_synthetics
        fi
    else
      run_python 4-find_transient
    fi
fi
