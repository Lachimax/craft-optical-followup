#!/usr/bin/env bash
# Code by Lachlan Marnoch, 2019
param_file=$1
destination=$2
proj_param_file=$3
type=$4
epoch=$5
instrument=$6
template_instrument=$7
manual=$8

if [[ -z ${proj_param_file} ]]; then
    proj_param_file=unicomp
fi

if [[ -z ${epoch} ]]; then
    epoch=1
fi

if [[ -z ${instrument} ]]; then
    instrument=FORS2
fi

if [[ -z ${template_instrument} ]]; then
    template_instrument=FORS2
fi
template_epoch=$(jq -r .template_epoch_${template_instrument,,} "param/FRBs/${param_file}.json")

if [[ -z ${manual} ]]; then
    manual=false
fi

if [[ -z ${type} ]]; then
    type=normal
fi

data_dir=$(jq -r .data_dir "param/FRBs/${param_file}.json")


proj_dir=$(jq -r .proj_dir "param/project/${proj_param_file}.json")

if [[ -z ${destination} ]]; then
    destination=${epoch}_${instrument}_${type}/
fi

sub_dir="${data_dir}subtraction/${destination}"
mkdir ${sub_dir}

if [[ ${type::5} != multi ]] ; then

        echo "python3 scripts/subtraction/1-prep.py --field ${param_file} --destination ${destination} --epoch ${epoch} --instrument ${instrument} --type ${type}"
        python3 scripts/subtraction/1-prep.py --field "${param_file}" --destination ${destination} --epoch ${epoch} --instrument ${instrument} --instrument_template ${template_instrument} --type ${type}

        cd ${sub_dir} || exit
        for filter in **/; do
            cd ${sub_dir}
            mkdir ${sub_dir}${filter}/sextractor/
            sextractor_destination_path=${sub_dir}${filter}/sextractor/alignment/
            mkdir "${sextractor_destination_path}"
            cp "${sub_dir}${filter}"*"comparison.fits" "${sextractor_destination_path}"
            cp "${sub_dir}${filter}"*"template.fits" "${sextractor_destination_path}"
            if ! ${manual} ; then
                if cd "${sextractor_destination_path}"; then
                    cp "${proj_dir}/param/psfex/"* .
                    cp "${proj_dir}/param/sextractor/default/"* .
                    pwd
                    sextractor *comparison.fits -c pre-psfex.sex -CATALOG_NAME "comparison_psfex.fits"
                    sextractor *template.fits -c pre-psfex.sex -CATALOG_NAME "template_psfex.fits"

                    psfex "comparison_psfex.fits"
                    psfex "template_psfex.fits"

                    fwhm_comparison=$(jq -r ".${filter::0}_fwhm_arcsec" "${sub_dir}${filter}"*comparison_output_values.json)
                    fwhm_template=$(jq -r ".${filter::0}_fwhm_arcsec" "${sub_dir}${filter}"*template_output_values.json)

                    sextractor *comparison.fits -c psf-fit.sex -CATALOG_NAME "comparison.cat" -PSF_NAME "comparison_psfex.psf" -SEEING_FWHM "${fwhm_comparison}"
                    sextractor *template.fits -c psf-fit.sex -CATALOG_NAME "template.cat" -PSF_NAME "template_psfex.psf" -SEEING_FWHM "${fwhm_template}"

                #      cd "${sextractor_destination_path}" || exit

                #      sex "${image}" -c pre-psfex.sex -CATALOG_NAME "${image_0}_psfex.fits"
                #      # Run PSFEx to get PSF analysis
                #      psfex "${image_0}_psfex.fits"
                #      cd "${proj_dir}" || exit
                #      # Use python to extract the FWHM from the PSFEx output.
                #      python3 "${proj_dir}/pipeline_fors2/9-psf.py" --directory "${sub_dir}" --psfex_file "${sextractor_destination_path}${image_0}_psfex.psf" --image_file "${sextractor_destination_path}${image}" --prefix "${image_0}"
                #      cd "${sextractor_destination_path}" || exit
                #      fwhm=$(jq -r ".${image_0}_fwhm_arcsec" "${sub_dir}output_values.json")
                #      echo "FWHM: ${fwhm} arcsecs"
                #      sex "${image}" -c psf-fit.sex -CATALOG_NAME "${image_0}_psf-fit.cat" -PSF_NAME "${image_0}_psfex.psf" -SEEING_FWHM "${fwhm}" -DETECT_THRESH "${threshold}" -ANALYSIS_THRESH "${threshold}"
                #   done
                fi
            fi
        cd ${proj_dir}
        echo Manual
        echo ${manual}
        if ${manual} ; then
            python3 scripts/subtraction/1-align.py --destination ${sub_dir}${filter} --field ${param_file} --epoch ${epoch} --instrument_template ${template_instrument} --filter ${filter::-1} -manual
        else
            python3 scripts/subtraction/1-align.py --destination ${sub_dir}${filter} --field ${param_file} --epoch ${epoch} --instrument_template ${template_instrument}
        fi
    done

else
    echo "python3 scripts/subtraction/1-prep_many.py --field ${param_file} --destination ${destination} --epoch ${epoch} --instrument ${instrument} --type ${type}"
    python3 scripts/subtraction/1-prep_many.py --field "${param_file}" --destination ${destination} --epoch ${epoch} --instrument ${instrument} --type ${type}

    template_dir="${data_dir}subtraction/${instrument}_${epoch}-${template_instrument}_${template_epoch}_normal/"
    cd ${sub_dir} || exit
    pwd
    for test in **/ ; do
        cd ${test} || exit
        test_dir=${sub_dir}${test}/
        for filter in **/; do
            template_yaml="${template_dir}${filter}${param_file}_${template_epoch}_template_output_values.yaml"
            outputs_yaml="${template_dir}${filter}output_values.yaml"

            cd ${proj_dir}
            if [[ ${template_instrument} == 'FORS2' ]] ; then
                # For FORS2 and X-shooter, we can skip reprojection and use the already-reprojected template, because the pixel scales are (almost) identical for FORS2 and smaller for X-shooter
                template="${template_dir}${filter}${param_file}_${template_epoch}_template_aligned.fits"
                cp ${template} ${test_dir}${filter}/
                echo ${template_yaml}
                cp ${template_yaml} ${test_dir}${filter}/
                python3 scripts/subtraction/1-align.py --destination ${test_dir}${filter}/ --field ${param_file} --epoch ${epoch} --offsets_yaml "${outputs_yaml}" -skip_reproject
            else
                # Not so for the others, whose footprints do not match FORS2
                template="${template_dir}${filter}${param_file}_${template_epoch}_template.fits"
                cp ${template} ${test_dir}${filter}/
                echo ${template_yaml}
                cp ${template_yaml} ${test_dir}${filter}/
                python3 scripts/subtraction/1-align.py --destination ${test_dir}${filter}/ --field ${param_file} --epoch ${epoch} --offsets_yaml "${outputs_yaml}"
            fi
            cd ${sub_dir}
        done
    done
fi

#if [[ ${type::5} == multi ]] ; then
#    echo "python3 scripts/subtraction/1-subtract_many.py --field ${param_file} --destination ${destination} --epoch ${epoch} --instrument ${instrument} --type ${type}"
#    python3 scripts/subtraction/1-subtract_many.py --field "${param_file}" --destination ${destination} --epoch ${epoch} --instrument ${instrument} --type ${type}
#else
#echo "python3 scripts/subtraction/1-subtract.py --field ${param_file} --destination ${destination} --epoch ${epoch} --instrument ${instrument} --type ${type}"
#python3 scripts/subtraction/1-subtract.py --field "${param_file}" --destination ${destination} --epoch ${epoch} --instrument ${instrument} --type ${type}
#fi