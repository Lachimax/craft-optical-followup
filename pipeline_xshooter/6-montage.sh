#!/bin/bash
# Bruce Berriman, February, 2016
# Adapted by Lachlan Marnoch, 2019

param_file=$1
proj_param_file=$2
origin=$3
destination=$4

if [[ -z ${proj_param_file} ]]; then
  proj_param_file=unicomp
fi

# TODO: Rewrite so that the files don't need to be copied to the Montage directory before combining, instead reading them from previous path.

pwd

proj_dir=$(jq -r .proj_dir param/project/${proj_param_file}.json)

data_dir=$(jq -r .data_dir param/epochs_xshooter/${param_file}.json)
data_title=$(jq -r .data_title param/epochs_xshooter/${param_file}.json)
object=$(jq -r .object param/epochs_xshooter/${param_file}.json)

if [[ -z ${origin} ]]; then
  origin=5-divided_by_exp_time
fi

if [[ -z ${destination} ]]; then
  destination=6-montage
fi

cd "${data_dir}" || exit
cd "${origin}" || exit
filters=$(ls -d **/)
cd "${data_dir}" || exit

echo "Copy science data to Montage folder..."

pwd

mkdir ${destination}/

if cp -r ${origin}/* ${destination}/; then

  cd "${data_dir}/${destination}" || exit

  for fil in ${filters}; do

    fil_0=${fil::1}

    echo "Create directories to hold processed images"
    mkdir "${fil_0}_projdir" "${fil_0}_diffdir" "${fil_0}_corrdir" | tee -a "${fil_0}_montage.log"

    echo "Create metadata tables of the input images"
    mImgtbl "${fil}" "${fil_0}.tbl" | tee -a "${fil_0}_montage.log"

    echo "Create FITS headers describing the footprint of the mosaic"
    mMakeHdr "${fil_0}.tbl" "${fil_0}_template.hdr" | tee -a "${fil_0}_montage.log"

    cd "${proj_dir}" || exit
    # Inject header changes.
    python3 "${proj_dir}/scripts/pipeline_xshooter/6-montage.py" --directory "${data_dir}" -op "${data_title}" --destination "${data_dir}/${destination}" --filter "${fil}" --object "${object}"
    cd "${data_dir}/${destination}" || exit

    echo "Reproject the input images"
    mProjExec -p "${fil}" "${fil_0}.tbl" "${fil_0}_template.hdr" "${fil_0}_projdir" "${fil_0}_stats.tbl" | tee -a "${fil_0}_montage.log"

    echo "Create a metadata table of the reprojected images"
    mImgtbl "${fil_0}_projdir/" "${fil_0}.tbl" | tee -a "${fil_0}_montage.log"

    echo "Analyze the overlaps between images"
    mOverlaps "${fil_0}.tbl" "${fil_0}_diffs.tbl" | tee -a "${fil_0}_montage.log"
    mDiffExec -p "${fil_0}_projdir/" "${fil_0}_diffs.tbl" "${fil_0}_template.hdr" "${fil_0}_diffdir" | tee -a "${fil_0}_montage.log"
    mFitExec "${fil_0}_diffs.tbl" "${fil_0}_fits.tbl" "${fil_0}_diffdir" | tee -a "${fil_0}_montage.log"

    echo "Perform background modeling and compute corrections for each image"
    mBgModel "${fil_0}.tbl" "${fil_0}_fits.tbl" "${fil_0}_corrections.tbl" | tee -a "${fil_0}_montage.log"

    echo "Apply corrections to each image"
    mBgExec -p "${fil_0}_projdir/" "${fil_0}.tbl" "${fil_0}_corrections.tbl" "${fil_0}_corrdir" | tee -a "${fil_0}_montage.log"

    echo "Coadd the images to create mosaics with background corrections"
    mAdd -p "${fil_0}_corrdir/" -a median "${fil_0}.tbl" "${fil_0}_template.hdr" "${fil_0}_coadded.fits" | tee -a "${fil_0}_montage.log"

  done

  date +%Y-%m-%dT%T >>"${data_title}.log"
  echo Combined with 6-montage.sh >>"${data_title}.log"
fi
