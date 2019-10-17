#!/usr/bin/env bash
# Script by Lachlan Marnoch 2019
# For sorting the products of the FORS ESO pipeline.

param_file=$1
proj_param_file=$2

if [[ -z ${proj_param_file} ]]; then
    proj_param_file=unicomp
fi

proj_dir=$(jq -r .proj_dir param/project/${proj_param_file}.json)
esoreflex_input_dir=$(jq -r .esoreflex_input_dir param/project/${proj_param_file}.json)
esoreflex_output_dir=$(jq -r .esoreflex_output_dir param/project/${proj_param_file}.json)

data_dir=$(jq -r .data_dir "param/epochs_fors2/${param_file}.json")
data_title=$(jq -r .data_title "param/epochs_fors2/${param_file}.json")
skip_copy=$(jq -r .skip_copy "param/epochs_fors2/${param_file}.json")
object=$(jq -r .object "param/epochs_fors2/${param_file}.json")

destination=${data_dir}/1-reduced_with_esoreflex
mkdir "${destination}"

if cd "${esoreflex_output_dir}" ; then

    # Copy the folder containing fits files with matching object names back from the esoreflex directory to the data directory.
    if ! ${skip_copy} ; then
        if ! [[ -z $(find . -name "*${object}*" -print -quit) ]] ; then
            for name in **/ ; do
                cd "${name}" || exit
                if [[ -n $(find . -name "*${object}*" -print -quit) ]] ; then
                    echo "Copying ${name} from ESOReflex directory to ${destination}"
                    cp -r . "${destination}"
                    cp "${esoreflex_input_dir}${data_title}/${data_title}.log" "${destination}/"
                    echo "${name%?}" >> "${destination}/${data_title}.log"
                    echo "Data reduced with ESOReflex." >> "${destination}/${data_title}.log"
                    break
                else
                    cd ..
                fi
            done
        else
            echo "Reduced data for this object not found."
            exit
        fi
    fi


    cd "${destination}" || exit
    origin_global=${destination}
    destination=${data_dir}/2-sorted
    mkdir "${destination}"

    obj_tbl_suffix="_OBJECT_TABLE_SCI_IMG.fits"
    background_suffix="_PHOT_BACKGROUND_SCI_IMG.fits"
    science_suffix="_SCIENCE_REDUCED_IMG.fits"
    source_suffix="_SOURCES_SCI_IMG.fits"

    # Go through folders and sort files by type.

    mkdir "${destination}/obj_tbls";
    mkdir "${destination}/backgrounds";
    mkdir "${destination}/science";
    mkdir "${destination}/sources";

    for name in **/
    do
        echo "${name}"
        name_no_slash=${name%?};

        # Copy object tables
        target=${destination}/obj_tbls/${object}-${name_no_slash}${obj_tbl_suffix}
        echo "Copying: " ./"${name}"*${obj_tbl_suffix}" to ${target}"
        cp ./"${name}"*${obj_tbl_suffix} "${target}"

        # Copy background images
        target=${destination}/backgrounds/${object}-${name_no_slash}${background_suffix}
        echo "Copying: " ./"${name}"*${background_suffix}" to ${target}"
        cp ./"${name}"*${background_suffix} "${target}"

        # Copy science images
        target=${destination}/science/${object}-${name_no_slash}${science_suffix}
        echo "Copying: " ./"${name}"*${science_suffix}" to ${target}"
        cp ./"${name}"*${science_suffix} "${target}"

        # Copy source images
        target=${destination}/sources/${object}${name_no_slash}${source_suffix}
        echo "Copying: " ./"${name}"*${source_suffix}" to ${target}"
        cp ./"${name}"*${source_suffix} "${target}"

    done

    cd "${proj_dir}" || exit

    python3 "${proj_dir}/scripts/pipeline_fors2/2-sort_after_esoreflex.py" --directory "${data_dir}/2-sorted/"

    pwd

    cp "${origin_global}/${data_title}.log" "${destination}/"
    date +%Y-%m-%dT%T >> "${destination}/${data_title}.log"
    echo Files sorted using sort_after_esoreflex.sh >> "${destination}/${data_title}.log"
    echo "All done."

fi
