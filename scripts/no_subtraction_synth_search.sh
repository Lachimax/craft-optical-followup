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
    instrument=XSHOOTER
fi

automate=$6
if [[ -z ${automate} ]]; then
    automate=false
fi

destination=$7

#if ! python3 scripts/refresh_params.py -op "${param_file}" -pp "${proj_param_file}"; then
#    echo "Something went wrong with reading or writing the param files."
#fi



if [[ -z ${destination} ]]; then
    destination=${instrument}_${epoch}_nosub_${type}/
fi
if [[ ${destination::-1} != / ]] ; then
    destination=${destination}/
fi

data_dir=$(jq -r .data_dir "param/FRBs/${param_file}.json")
proj_dir=$(jq -r .proj_dir "param/project/${proj_param_file}.json")
threshold=$(jq -r .threshold "param/FRBs/${param_file}".json)

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
            if "${proj_dir}scripts/subtraction/${script}.sh" "${param_file}" "${destination}" "${proj_param_file}" "${type}" "${epoch}" "${instrument}" "XSHOOTER" "${extra_argument}" ; then
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
            if python3 "${proj_dir}scripts/subtraction/${script}.py" --field "${param_file}" --subtraction_path "${destination}" --epoch "${epoch}" --instrument "${instrument}" --instrument_template "XSHOOTER"; then
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

# Prep

sub_dir="${data_dir}subtraction/${destination}"

echo "python3 scripts/subtraction/1-prep_many.py --field ${param_file} --destination ${destination} --epoch ${epoch} --instrument ${instrument} --type ${type} --instrument_template XSHOOTER"
python3 scripts/subtraction/1-prep_many.py --field "${param_file}" --destination ${destination} --epoch ${epoch} --instrument ${instrument} --type ${type} --instrument_template "XSHOOTER"

# SExtractor

if cd "${sub_dir}" ; then
    for path in  **/ ; do
        echo Going to "$(pwd)"/"${path}"
        cd "${path}" || exit
        for filter in **/ ; do
            cd "${filter}" || exit
            cp "${proj_dir}/param/sextractor/default/"* .
            # cp ${proj_dir}/param/psfex/* .
            pwd
            for image in *comparison.fits ; do
                cp ${image} ${image}_difference.fits
                cp ${image} ${image}_comparison_aligned.fits
            done
            cd ..
        done
        cd ..
    done
fi

cd ${proj_dir}

if cd "${sub_dir}" ; then
        for path in  **/ ; do
            echo Going to "$(pwd)"/"${path}"
            cd "${path}" || exit
            for filter in **/ ; do
                cd "${filter}" || exit
                cp "${proj_dir}/param/sextractor/default/"* .
                # cp ${proj_dir}/param/psfex/* .
                pwd
                for image in *difference.fits ; do
                    # sextractor ${image} -c im.sex -CATALOG_NAME difference.cat -DETECT_THRESH 2.0
                    echo "${sub_dir}${path}${filter}output_values.json"
                    fwhm_pix=$(jq -r .max_fwhm_pix "${sub_dir}${path}${filter}output_values.json")
                    fwhm_arcsec=$(jq -r .max_fwhm_arcsec "${sub_dir}${path}${filter}output_values.json")
                    aperture=$(jq -r .measurement_aperture "${sub_dir}${path}${filter}output_values.json")
                    #echo "FWHM: ${fwhm_pix} pixels"
                    #echo "FWHM: ${fwhm_arcsec} arcseconds"
                    #echo "Aperture: ${aperture} pixels"
                    sextractor "${image}" -c im.sex -CATALOG_NAME difference.cat -PHOT_APERTURES "${aperture}" -DETECT_THRESH "${threshold}" -ANALYSIS_THRESH "${threshold}" -BACK_SIZE 5
                    # cd "${proj_dir}" || exit
                    # python3 plots/draw_fitting_params.py --image ${sextractor_destination_path}${path}${filter}${image} --cat ${sextractor_destination_path}${path}${filter}difference.cat
                    # cd "${sextractor_destination_path}${path}${filter}" || exit
                    # Run PSFEx to get PSF analysis
                    # psfex psfex.fits

                    # cd ${proj_dir}
                    # Use python to extract the FWHM from the PSFEx output.
                    # python3 ${proj_dir}/scripts/pipeline_fors2/9-psf.py --directory ${sextractor_destination_path} --psfex_file ${sextractor_destination_path}${filter}psfex.psf --image_file ${sextractor_destination_path}${filter}${image} --prefix ${f_0}
                    # cd ${sextractor_destination_path}${filter}


                    #sextractor ${image} -c psf-fit.sex -CATALOG_NAME difference_psf-fit.cat -PSF_NAME psfex.psf -PHOT_APERTURES 10 #-SEEING_FWHM ${fwhm} -DETECT_THRESH ${threshold} -ANALYSIS_THRESH ${threshold}
                done
                cd ..
            done
            cd ..
        done
    fi

cd ${proj_dir}
if ${multi} ; then
        python3 "${proj_dir}scripts/subtraction/4-recover_synthetics_multi.py" --field "${param_file}" --subtraction_path "${destination}" --epoch "${epoch}" --instrument "${instrument}" --instrument_template "XSHOOTER"
    else
        python3 "${proj_dir}scripts/subtraction/4-recover_synthetics.py" --field "${param_file}" --subtraction_path "${destination}" --epoch "${epoch}" --instrument "${instrument}" --instrument_template "XSHOOTER"
fi
