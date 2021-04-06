# Code by Lachlan Marnoch, 2021

import urllib
from datetime import date
from typing import Union
import requests
import os
import time

from craftutils import params as p
from craftutils import utils as u

keys = p.keys()
fors2_filters_retrievable = ["I_BESS", "R_SPEC", "b_HIGH", "v_HIGH"]
photometry_catalogues = ['DES', 'SDSS', 'SkyMapper']


def cat_columns(cat, f: str = None):
    cat = cat.lower()
    if f is not None:
        f = f[0]
    else:
        f = ""

    if cat == 'des':
        f = f.upper()
        return {'mag_psf': f"WAVG_MAG_PSF_{f}",
                'mag_psf_err': f"WAVG_MAGERR_PSF_{f}",
                'ra': f"RA",
                'dec': f"DEC",
                'class_star': f"CLASS_STAR_{f}"}
    elif cat == 'sdss':
        f = f.lower()
        return {'mag_psf': f"psfMag_{f}",
                'mag_psf_err': f"WAVG_MAGERR_PSF_{f}",
                'ra': f"ra",
                'dec': f"dec",
                'class_star': f"probPSF_{f}"}
    elif cat == 'skymapper':
        f = f.lower()
        return {'mag_psf': f"{f}_psf",
                'mag_psf_err': f"WAVG_MAGERR_PSF_{f}",
                'ra': f"raj2000",
                'dec': f"dej2000",
                'class_star': f"class_star_SkyMapper"}
    else:
        raise ValueError(f"Catalogue {cat} not recognised.")


def update_std_photometry_all(ra: float, dec: float):
    for cat_name in photometry_catalogues:
        update_std_photometry(ra=ra, dec=dec, cat=cat_name)


def update_std_photometry(ra: float, dec: float, cat: str):
    cat = cat.lower()
    if cat == 'des':
        return update_std_des_photometry(ra=ra, dec=dec)
    elif cat == 'sdss':
        return update_std_sdss_photometry(ra=ra, dec=dec)
    elif cat == 'skymapper':
        return update_std_skymapper_photometry(ra=ra, dec=dec)
    else:
        raise ValueError("Catalogue name not recognised.")


def update_frb_photometry(frb: str, cat: str):
    cat = cat.lower()
    if cat == 'des':
        return update_frb_des_photometry(frb=frb)
    elif cat == 'sdss':
        return update_frb_sdss_photometry(frb=frb)
    elif cat == 'skymapper':
        return update_frb_skymapper_photometry(frb=frb)
    else:
        raise ValueError("Catalogue name not recognised.")


def retrieve_fors2_calib(fil: str = 'I_BESS', date_from: str = '2017-01-01', date_to: str = None):
    """
    Retrieves the full set of photometry parameters from the FORS2 quality control archive
    (http://archive.eso.org/bin/qc1_cgi?action=qc1_browse_table&table=fors2_photometry), from date_from to date_to.
    :param fil: The filter for which the data is to be retrieved. Must be "I_BESS", "R_SPEC", "b_HIGH" or "v_HIGH".
    :param date_from: The date from which to begin.
    :param date_to: The date on which to end. If None, defaults to current date.
    :return: The table of parameters, as a string.
    """
    if fil not in fors2_filters_retrievable:
        raise ValueError(f"{fil} not recognised; fil must be one of {fors2_filters_retrievable}")
    if date_to is None:
        date_to = str(date.today())
    # Construct the data expected by the FORS2 QC1 archive to send as a request.
    request = {
        "field_mjd_obs": "mjd_obs",
        "field_civil_date": "civil_date",
        "field_zeropoint": "zeropoint",
        "field_zeropoint_err": "zeropoint_err",
        "field_colour_term": "colour_term",
        "field_colour_term_err": "colour_term_err",
        "field_colour": "colour",
        "field_extinction": "extinction",
        "field_extinction_err": "extinction_err",
        "field_num_ext": "num_ext",
        "field_num_fields": "num_fields",
        "field_num_nights": "num_nights",
        "field_date_range": "date_range",
        "field_stable": "stable",
        "field_filter_name": "filter_name",
        "filter_filter_name": fil,
        "field_det_chip1_id": "det_chip1_id",
        "filter_det_chip1_id": "CCID20-14-5-3",
        "field_det_chip_num": "det_chip_num",
        "filter_det_chip_num": "1",
        "from": date_from,
        "to": date_to,
        "action": "qc1_browse_get",
        "table": "fors2_photometry",
        "output": "ascii",
    }
    request = urllib.parse.urlencode(request)
    request = bytes(request, 'utf-8')
    page = urllib.request.urlopen("http://archive.eso.org/qc1/qc1_cgi", request)
    return str(page.read().replace(b'!', b''), 'utf-8')


def save_fors2_calib(output: str, fil: str = 'I_BESS', date_from: str = '2017-01-01', date_to: str = None):
    """
    Retrieves the full set of photometry parameters from the FORS2 quality control archive
    (http://archive.eso.org/bin/qc1_cgi?action=qc1_browse_table&table=fors2_photometry), from date_from to date_to,
    formats them conveniently for numpy to read, and writes them to disk at the location given by output.
    :param output: The location on disk to which to write the file.
    :param fil: The filter for which the data is to be retrieved. Must be "I_BESS", "R_SPEC", "b_HIGH" or "v_HIGH".
    :param date_from: The date from which to begin.
    :param date_to: The date on which to end.
    :return: The table of parameters, as a string.
    """
    print(f"Updating ESO QC1 parameters for {fil} to {output}")
    string = retrieve_fors2_calib(fil=fil, date_from=date_from, date_to=date_to)
    i = j = string.find('\n') + 1
    while string[j] == '-':
        j += 1
    string = string[:i] + string[j + 1:]
    with open(output, "w") as file:
        file.write(string)
    return string


def update_fors2_calib():
    """
    Runs save_fors2_calib() for all four retrievable FORS2 filters.
    """
    print("Retrieving calibration parameters from FORS2 QC1 archive...")
    for fil in fors2_filters_retrievable:
        if fil == 'R_SPEC':
            fil = 'R_SPECIAL'
        fil_params = p.filter_params(f=fil, instrument="FORS2")
        updated = date.fromisoformat(fil_params["calib_last_updated"])
        print(f"{fil} calibration last updated on", str(updated))
        if updated == date.today():
            print("Filter calibrations already updated today; skipping.")
        else:
            path = p.config['top_data_dir'] + "photometry_calib/" + fil + '.txt'
            if fil == 'R_SPECIAL':
                fil = 'R_SPEC'
            save_fors2_calib(output=path, fil=fil)
            p.ingest_filter_properties(path=path, instrument='FORS2', update=True)


def retrieve_irsa_xml(ra: float, dec: float):
    """
    Retrieves the extinction parameters for a given sky position from the IRSA Dust Tool
    (https://irsa.ipac.caltech.edu/applications/DUST/)
    :param ra: Right Ascension of the desired field, in degrees.
    :param dec: Declination of the desired field, in degrees.
    :return: XML-formatted string.
    """
    url = f"https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr={ra}+{dec}+equ+j2000"
    print("\nRetrieving IRSA Dust Tool XML from", url)
    irsa_xml = urllib.request.urlopen(url)
    irsa_xml = irsa_xml.read()
    return str(irsa_xml, 'utf-8')


def retrieve_irsa_extinction(ra: float, dec: float):
    """
    Retrieves the extinction per bandpass table, and other relevant parameters, for a given sky position from the
    IRSA Dust Tool (https://irsa.ipac.caltech.edu/applications/DUST/).
    :param ra: Right Ascension of the desired field, in degrees.
    :param dec: Declination of the desired field, in degrees.
    :return: Tuple: dictionary of retrieved values, table-formatted string.
    """
    irsa_xml = retrieve_irsa_xml(ra=ra, dec=dec)

    to_retrieve = {"refPixelValueSandF": "dust_ebv"}
    retrieved = {}
    for tag in to_retrieve:
        val_str = u.extract_xml_param(tag="refPixelValueSandF", xml_str=irsa_xml)
        retrieved[to_retrieve[tag]], _ = u.unit_str_to_float(val_str)
    i = irsa_xml.find("extinction.tbl")
    j = i + 14
    substr = irsa_xml[i:i + 8]
    while substr != "https://":
        i -= 1
        substr = irsa_xml[i:i + 8]
    table_url = irsa_xml[i:j]
    print("Retrieving bandpass extinction table from", table_url)
    extinction = urllib.request.urlopen(table_url)
    ext_str = extinction.read()
    ext_str = str(ext_str, 'utf-8')
    return retrieved, ext_str


def save_irsa_extinction(ra: float, dec: float, output: str):
    """
    Retrieves the extinction per bandpass table for a given sky position from the IRSA Dust Tool
    (https://irsa.ipac.caltech.edu/applications/DUST/) and writes it to disk.
    :param ra: Right Ascension of the desired field, in degrees.
    :param dec: Declination of the desired field, in degrees.
    :param output: The location on disk to which to write the file.
    :return: Tuple: dictionary of retrieved values, table-formatted string.
    """
    values, ext_str = retrieve_irsa_extinction(ra=ra, dec=dec)
    ext_str = ext_str.replace("microns", "um")
    with open(output, "w") as file:
        file.write(ext_str)
    return values, ext_str


def update_frb_irsa_extinction(frb: str):
    """
    Retrieves the extinction per bandpass table, and other relevant parameters, for a given sky position from the
    IRSA Dust Tool (https://irsa.ipac.caltech.edu/applications/DUST/) and writes it to disk.
    :param frb: FRB name, FRBXXXXXX. Must match title of param file.
    :return: Tuple: dictionary of retrieved values, table-formatted string.
    """
    params = p.object_params_frb(obj=frb)
    outputs = p.frb_output_params(obj=frb)
    data_dir = params['data_dir']
    if 'dust_ebv' not in outputs and not os.path.isfile(data_dir + "galactic_extinction.txt"):
        values, ext_str = save_irsa_extinction(ra=params['burst_ra'], dec=params['burst_dec'],
                                               output=data_dir + "galactic_extinction.txt")
        p.add_output_values_frb(obj=frb, params=values)
        return values, ext_str
    else:
        print("IRSA Dust Tool data already retrieved.")


sdss_filters = ["u", "g", "r", "i", "z"]


def retrieve_sdss_photometry(ra: float, dec: float):
    """
    Retrieve SDSS photometry for a given field, in a 0.2 x 0.2 degree box centred on the passed coordinates
    coordinates. (Note - the width of the box is in RA degrees, not corrected for spherical distortion)
    :param ra: Right Ascension of the centre of the desired field, in degrees.
    :param dec: Declination of the centre of the desired field, in degrees.
    :return: Retrieved photometry table, as a pandas dataframe, if successful; if not, None.
    """
    try:
        from SciServer import Authentication, CasJobs
    except ImportError:
        print("It seems that SciScript/SciServer is not installed, or not accessible to this environment. "
              "\nIf you wish to automatically download SDSS data, please install "
              "\nSciScript (https://github.com/sciserver/SciScript-Python); "
              "\notherwise, retrieve the data manually from "
              "\nhttp://skyserver.sdss.org/dr16/en/tools/search/sql.aspx")
        return None

    print(f"Querying SDSS DR16 archive for field centring on RA={ra}, DEC={dec}")
    user = keys['sciserver_user']
    password = keys["sciserver_pwd"]
    Authentication.login(UserName=user, Password=password)
    # Construct an SQL query to send to SciServer
    query = "SELECT objid,ra,dec"
    for f in sdss_filters:
        query += f",psfMag_{f},psfMagErr_{f},fiberMag_{f},fiberMagErr_{f},fiber2Mag_{f},fiber2MagErr_{f},petroMag_{f},petroMagErr_{f} "
    query += "FROM PhotoObj "
    query += f"WHERE ra BETWEEN {ra - 0.1} AND {ra + 0.1} "
    query += f"AND dec BETWEEN {dec - 0.1} AND {dec + 0.1} "
    print(f"Retrieving photometry from SDSS DR16 via SciServer for field at {ra}, {dec}...")
    df = CasJobs.executeQuery(sql=query, context='DR16')
    if len(df.index) == 0:
        df = None
    return df


def save_sdss_photometry(ra: float, dec: float, output: str):
    """
    Retrieves and writes to disk the SDSS photometry for a given field, in a 0.2 x 0.2 degree box
    centred on the field coordinates. (Note - the width of the box is in RA degrees, not corrected for spherical
    distortion)
    :param ra: Right Ascension of the centre of the desired field, in degrees.
    :param dec: Declination of the centre of the desired field, in degrees.
    :param output: The location on disk to which to write the file.
    :return: Retrieved photometry table, as a pandas dataframe, if successful; if not, None.
    """
    df = retrieve_sdss_photometry(ra=ra, dec=dec)
    if df is not None:
        u.mkdir_check_nested(path=output)
        print("Saving SDSS photometry to" + output)
        df.to_csv(output)
    else:
        print("No data retrieved from SDSS.")
    return df


def update_std_sdss_photometry(ra: float, dec: float, force: bool = False):
    """
    Retrieves and writes to disk the SDSS photometry for a standard-star calibration field, in a 0.2 x 0.2 degree box
    centred on the field coordinates. (Note - the width of the box is in RA degrees, not corrected for spherical
    distortion)
    :param ra: Right Ascension of the centre of the desired field, in degrees.
    :param dec: Declination of the centre of the desired field, in degrees.
    :return: Retrieved photometry table, as a pandas dataframe, if successful; if not, None.
    """
    data_dir = p.config['top_data_dir']
    field_path = f"{data_dir}/std_fields/RA{ra}_DEC{dec}/"
    outputs = p.load_params(field_path + "output_values")
    if outputs is None or "in_sdss" not in outputs or force:
        path = field_path + "SDSS/SDSS.csv"
        response = save_sdss_photometry(ra=ra, dec=dec, output=path)
        params = {}
        if response is not None:
            params["in_sdss"] = True
        else:
            params["in_sdss"] = False
        p.add_params(file=field_path + "output_values", params=params)
        return response
    elif outputs["in_sdss"] is True:
        print("There is already SDSS data present for this field.")
        return True
    else:
        print("This field is not present in SDSS.")


def update_frb_sdss_photometry(frb: str, force: bool = False):
    """
    Retrieve SDSS photometry for the field of an FRB (with a valid param file in param_dir), in a 0.2 x 0.2 degree box
    centred on the FRB coordinates, and
    (Note - the width of the box is in RA degrees, not corrected for spherical distortion)
    :param frb: FRB name, FRBXXXXXX. Must match title of param file.
    :return: Retrieved photometry table, as a pandas dataframe, if successful; if not, None.
    """
    params = p.object_params_frb(frb)
    path = params['data_dir'] + "SDSS/SDSS.csv"
    outputs = p.frb_output_params(obj=frb)
    if outputs is None or "in_sdss" not in outputs or force:
        response = save_sdss_photometry(ra=params['burst_ra'], dec=params['burst_dec'], output=path)
        params = {}
        if response is not None:
            params["in_sdss"] = True
        else:
            params["in_sdss"] = False
        p.add_output_values_frb(obj=frb, params=params)
        return response
    elif outputs["in_sdss"] is True:
        print("There is already SDSS data present for this field.")
        return True
    else:
        print("This field is not present in SDSS.")


# Dark Energy Survey database functions adapted code by T. Andrew Manning, from
# https://github.com/des-labs/desaccess-docs/blob/master/_static/DESaccess_API_example.ipynb

des_url = 'https://des.ncsa.illinois.edu'
des_api_url = des_url + "/desaccess/api"
des_files_url = des_url + "/files-desaccess"
des_filters = ['g', 'r', 'i', 'z', 'Y']


def login_des():
    """
    Obtains an auth token using the username and password credentials for a given database.
    """
    # Login to obtain an auth token
    r = requests.post(
        f'{des_api_url}/login',
        data={
            'username': keys['des_user'],
            'password': keys['des_pwd'],
            'database': 'desdr'
        }
    )
    # Store the JWT auth token
    keys['des_auth_token'] = r.json()['token']
    return keys['des_auth_token']


def check_auth_token_des():
    """
    Raises a KeyError if login_des() needs to be run.
    """
    if 'des_auth_token' not in keys:
        raise KeyError("Use login_des() to log in before submitting requests.")


def check_success_des(response: requests.Response):
    """
    Checks whether a job has been submitted successfully to DESAccess API, and refreshes the auth token.
    :param response: The response to a requests.put() query to the DESAccess API
    :return: str: The Response passed to response, but in JSON format.
    """
    response = response.json()

    if response['status'] == 'ok':
        job_id = response['jobid']
        print('Job "{}" submitted.'.format(job_id))
        # Refresh auth token
        keys['des_auth_token'] = response['new_token']
    else:
        print('Error submitting job: '.format(response['message']))
    return response


def submit_query_job_des(query: str):
    """
    Submits a query job and returns the complete server response, which includes the job ID.
    :param query: An SQL query to submit to the DES DR3 catalogue.
    :return: A dictionary constructed from the JSON response to the initial query.
    """

    check_auth_token_des()

    # Specify API request parameters
    data = {
        'username': keys['des_user'],
        'db': 'desdr',
        'filename': 'DES.csv',
        'query': query
    }

    # Submit job
    r = requests.put(
        f'{des_api_url}/job/query',
        data=data,
        headers={'Authorization': f'Bearer {keys["des_auth_token"]}'}
    )
    response = check_success_des(response=r)

    return response


def get_job_status_des(job_id: str):
    """
    Returns the current status of the job identified by the unique job_id, and updates the auth_token.
    :param job_id: The unique job id generated by DES Access upon submission of the job, to be found as response['jobid']
    :return: A dictionary constructed from the JSON response to the status query.
    """

    check_auth_token_des()

    r = requests.post(
        f'{des_api_url}/job/status',
        data={'job-id': job_id},
        headers={'Authorization': f'Bearer {keys["des_auth_token"]}'}
    )
    response = r.json()
    # Refresh auth token
    keys['des_auth_token'] = response['new_token']
    # print(json.dumps(response, indent=2))
    return response


def job_status_poll_des(job_id: str):
    """
    Waits for a DESAccess job to complete, and returns the final response.
    :param job_id: The unique job id generated by DES Access upon submission of the job, to be found as response['jobid']
    :return: A dictionary constructed from the JSON response to the final query.
    """
    print(f'Polling status of job "{job_id}"...', end='')
    job_status = ''
    response = None
    while job_status != 'ok':
        # Fetch the current job status
        response = get_job_status_des(job_id)
        # Quit polling if there is an error getting a status update
        if response['status'] != 'ok':
            break
        job_status = response['jobs'][0]['job_status']
        if job_status == 'success' or job_status == 'failure':
            print(f'\nJob completed with status: {job_status}')
            break
        else:
            # Display another dot to indicate that polling is still active
            print('.', end='', sep='', flush=True)
        time.sleep(3)
    return response


def download_job_files_des(url: str, output: str):
    """
    Retrieves and writes the files hosted at the given DESAccess job URL.
    :param url: URL to access; should be formatted as https://des.ncsa.illinois.edu/files-desaccess/{username}/query/{job_id}/'
    :param output: The local directory to write the retrieved files to.
    :return: A dictionary constructed from the JSON response to the requests.get() query.
    """
    print("Checking or making directory", output)
    os.makedirs(output, exist_ok=True)
    r = requests.get(f'{url}/json')
    for item in r.json():
        if item['type'] == 'directory':
            suburl = f'{url}/{item["name"]}'
            subdir = f'{output}/{item["name"]}'
            download_job_files_des(suburl, subdir)
        elif item['type'] == 'file':
            data = requests.get(f'{url}/{item["name"]}')
            with open(f'{output}/{item["name"]}', "wb") as file:
                file.write(data.content)

    response = r.json()
    return response


def retrieve_query_csv_des(job_id: str):
    """
    Retrieves the .csv file produced from an SQL query previously submitted to DESaccess, if any exists.
    :param job_id: The unique job id generated by DES Access upon submission of the job.
    :return: Retrieved photometry table, as a Bytes object, if successful; None if not.
    """
    url = f'{des_files_url}/{keys["des_user"]}/query/{job_id}/'
    print("Retrieving DES photometry from", url)
    r = requests.get(f'{url}/json')
    for item in r.json():
        if item['name'][-4:] == '.csv':
            data = requests.get(f'{url}/{item["name"]}')
            return data.content
    return None


def retrieve_des_photometry(ra: float, dec: float):
    """
    Retrieve DES photometry for a given field, in a 0.2 x 0.2 degree box centred on the passed coordinates
    coordinates. (Note - the width of the box is in RA degrees, not corrected for spherical distortion)
    :param ra: Right Ascension of the centre of the desired field, in degrees.
    :param dec: Declination of the centre of the desired field, in degrees.
    :return: Retrieved photometry table, as a Bytes object, if successful; None if not.
    """
    print(f"Querying DES DR2 archive for field centring on RA={ra}, DEC={dec}")
    login_des()
    query = f"SELECT * " \
            f"FROM DR2_MAIN " \
            f"WHERE " \
            f"RA BETWEEN {ra - 0.1} and {ra + 0.1} and " \
            f"DEC BETWEEN {dec - 0.1} and {dec + 0.1} and " \
            f"ROWNUM < 10000 "
    print('Submitting query job...')
    response = submit_query_job_des(query)
    # Store the unique job ID for the new job
    job_id = response['jobid']
    print(f'New job submitted: {job_id}')
    response = job_status_poll_des(job_id)
    if response['status'] == 'ok':
        job_id = response['jobs'][0]['job_id']
        return retrieve_query_csv_des(job_id=job_id)


def save_des_photometry(ra: float, dec: float, output: str):
    """
    Retrieves and writes to disk the DES photometry for a given field, in a 0.2 x 0.2 degree box
    centred on the field coordinates. (Note - the width of the box is in RA degrees, not corrected for spherical
    distortion)
    :param ra: Right Ascension of the centre of the desired field, in degrees.
    :param dec: Declination of the centre of the desired field, in degrees.
    :param output: The location on disk to which to write the file.
    :return: Retrieved photometry table, as a Bytes object, if successful; None if not.
    """
    data = retrieve_des_photometry(ra=ra, dec=dec)
    if data is not None:
        u.mkdir_check_nested(path=output)
        print("Saving DES photometry to" + output)
        with open(output, "wb") as file:
            file.write(data)
    else:
        print('No data retrieved from DES.')
    return data


def update_std_des_photometry(ra: float, dec: float, force: bool = False):
    """
    Attempts to retrieve and write to disk the DES photometry for a standard-star calibration field, in a 0.2 x 0.2
    degree box centred on the field coordinates (Note - the width of the box is in RA degrees, not corrected for
    spherical distortion).
    Updates the "in_des" value (in output_values.yaml in the field path) to True if the data is successfully retrieved,
    and to False if not.
    If the data already exists in the std_fields directory, or if the field is outside the DES footprint
    (as ascertained by a previous query and stored in output_values.yaml), no attempt is made (unless
    force is True).
    :param ra: Right Ascension of the centre of the desired field, in degrees.
    :param dec: Declination of the centre of the desired field, in degrees.
    :param force: If True, ignores both the existence of the data in the std_fields directory and whether the field is
    in the DES footprint.
    :return: If the query has run successfully, the retrieved photometry table as a Bytes object;
        If the query has not run because the data is already present, True;
        If the query has not run because "in_des" is False, None
    """
    data_dir = p.config['top_data_dir']
    field_path = f"{data_dir}/std_fields/RA{ra}_DEC{dec}/"
    outputs = p.load_params(field_path + "output_values")
    if outputs is None or "in_des" not in outputs or force:
        path = field_path + "DES/DES.csv"
        response = save_des_photometry(ra=ra, dec=dec, output=path)
        params = {}
        if response is not None:
            params["in_des"] = True
        else:
            params["in_des"] = False
        p.add_params(file=field_path + "output_values", params=params)
        return response
    elif outputs["in_des"] is True:
        print("There is already DES data present for this field.")
        return True
    else:
        print("This field is not present in DES.")


def update_frb_des_photometry(frb: str, force: bool = False):
    """
    Retrieve DES photometry for the field of an FRB (with a valid param file in param_dir), in a 0.2 x 0.2 degree box
    centred on the FRB coordinates, and download it to the appropriate data directory.
    (Note - the width of the box is in RA degrees, not corrected for spherical distortion)
    Updates the "in_des" value (in output_values.yaml in the FRB data_dir path) to True if the data is successfully retrieved,
    and to False if not.
    If the data already exists in the FRB data_dir directory, or if the field is outside the DES footprint
    (as ascertained by a previous query and stored in output_values.yaml), no attempt is made (unless
    force is True).
    :param force: If True, ignores both the existence of the data in the data_dir directory and whether the field is
    in the DES footprint.
    :param frb: FRB name, FRBXXXXXX. Must match title of param file.
    :return: True if successful, False if not.
    """
    params = p.object_params_frb(frb)
    path = params['data_dir'] + "DES/DES.csv"
    outputs = p.frb_output_params(obj=frb)
    if outputs is None or "in_des" not in outputs or force:
        response = save_des_photometry(ra=params['burst_ra'], dec=params['burst_dec'], output=path)
        params = {}
        if response is not None:
            params["in_des"] = True
        else:
            params["in_des"] = False
        p.add_output_values_frb(obj=frb, params=params)
        return response
    elif outputs["in_des"] is True:
        print("There is already DES data present for this field.")
        return True
    else:  # outputs["in_des"] is False
        print("This field is not present in DES.")
        return None


def submit_cutout_job_des(ra: float, dec: float):
    """
    Submits a cutout job  and returns the complete server response, which includes the job ID.
    :param ra: The Right Ascension of the cutout centre, in degrees.
    :param dec: The Declination of the cutout centre, in degrees.
    :return:
    """

    positions = f'''
    RA,DEC,XSIZE,YSIZE,COLORS_FITS,MAKE_FITS\n
    {ra},{dec},9.0,9.0,grizy,true
    '''

    data = {'db': 'desdr',
            'release': 'dr2',
            'positions': positions}

    check_auth_token_des()

    # Submit job
    r = requests.put(
        f'{des_api_url}/job/cutout',
        data=data,
        headers={'Authorization': f'Bearer {keys["des_auth_token"]}'})
    response = check_success_des(r)

    return response


def save_des_cutout(ra: float, dec: float, output: str):
    """

    :param ra:
    :param dec:
    :return:
    """
    print(f"Requesting cutout images from DES DR2 archive for field centring on RA={ra}, DEC={dec}")
    login_des()
    u.mkdir_check_nested(output)
    response = submit_cutout_job_des(ra=ra, dec=dec)
    if response['status'] == 'ok':
        # Store the unique job ID for the new job
        job_id = response['jobid']
        print(f'New job submitted: {job_id}')
        response = job_status_poll_des(job_id)
        job_id = response['jobs'][0]['job_id']
        url = f'{des_files_url}/{keys["des_user"]}/cutout/{job_id}'
        r = requests.get(f'{url}/json')
        for item in r.json():
            if item['type'] == "directory":
                sub_url = f'{url}/{item["name"]}/'
                r_2 = requests.get(f'{sub_url}/json').json()
                sub_sub_url = f'{sub_url}/{r_2[0]["name"]}'
                r_3 = requests.get(f'{sub_sub_url}/json').json()
                for f in r_3:
                    if f["name"][-5:] == ".fits":
                        data = requests.get(f'{sub_sub_url}/{f["name"]}')
                        new_title = f["name"][-6] + "_cutout.fits"
                        with open(f'{output}/{new_title}', "wb") as file:
                            file.write(data.content)
        return True
    else:
        print(f"No cutouts could be retrieved for field {ra}, {dec}")
        return False


def update_frb_des_cutout(frb: str, force: bool = False):
    """

    :param frb:
    :return:
    """
    params = p.object_params_frb(obj=frb)
    outputs = p.frb_output_params(obj=frb)
    if "in_des" not in outputs:
        update_frb_des_photometry(frb=frb)
        outputs = p.frb_output_params(obj=frb)
    if force or outputs["in_des"]:
        path = params['data_dir'] + "DES/0-data/"
        files = os.listdir(path)
        condition = False
        for f in des_filters:
            if f"{f.lower()}_cutout.fits" not in files:
                condition = True
        if force or condition:
            return save_des_cutout(ra=params['burst_ra'], dec=params['burst_dec'], output=path)
        else:
            print("DES cutout already downloaded.")
    else:
        print("No DES cutout available for this position.")


def retrieve_skymapper_photometry(ra: float, dec: float):
    print(f"\nQuerying SkyMapper DR3 archive for field centring on RA={ra}, DEC={dec}")
    url = f"http://skymapper.anu.edu.au/sm-cone/aus/query?RA={ra}&DEC={dec}&SR=0.2&RESPONSEFORMAT=CSV"
    try:
        response = requests.get(url).content
    except requests.exceptions.SSLError:
        print('An SSL error occurred when retrieving SkyMapper data. Skipping.')
        return None
    if response.count(b"\n") <= 1:
        return None
    else:
        return response


def save_skymapper_photometry(ra: float, dec: float, output: str):
    response = retrieve_skymapper_photometry(ra=ra, dec=dec)
    if response is not None:
        u.mkdir_check_nested(path=output)
        print("Saving SkyMapper photometry to" + output)
        with open(output, "wb") as file:
            file.write(response)
    else:
        print('No data retrieved from SkyMapper.')
    return response


def update_std_skymapper_photometry(ra: float, dec: float, force: bool = False):
    data_dir = p.config['top_data_dir']
    field_path = f"{data_dir}/std_fields/RA{ra}_DEC{dec}/"
    outputs = p.load_params(field_path + "output_values")
    if outputs is None or "in_skymapper" not in outputs or force:
        path = field_path + "SkyMapper/SkyMapper.csv"
        response = save_skymapper_photometry(ra=ra, dec=dec, output=path)
        params = {}
        if response is not None:
            params["in_skymapper"] = True
        else:
            params["in_skymapper"] = False
        p.add_params(file=field_path + "output_values", params=params)
        return response
    elif outputs["in_skymapper"] is True:
        print("There is already SkyMapper data present for this field.")
        return True
    else:
        print("This field is not present in SkyMapper.")


def update_frb_skymapper_photometry(frb: str, force: bool = False):
    """
    Retrieve DES photometry for the field of an FRB (with a valid param file in param_dir), in a 0.2 deg radius cone
    centred on the FRB coordinates, and download it to the appropriate data directory.
    (Note - the width of the box is in RA degrees, not corrected for spherical distortion)
    :param frb: FRB name, FRBXXXXXX. Must match title of param file.
    :return: True if successful, False if not.
    """
    params = p.object_params_frb(frb)
    path = params['data_dir'] + "SkyMapper/SkyMapper.csv"
    outputs = p.frb_output_params(obj=frb)
    if outputs is None or "in_skymapper" not in outputs or force:
        response = save_skymapper_photometry(ra=params['burst_ra'], dec=params['burst_dec'], output=path)
        params = {}
        if response is not None:
            params["in_skymapper"] = True
        else:
            params["in_skymapper"] = False
        p.add_output_values_frb(obj=frb, params=params)
        return response
    elif outputs["in_skymapper"] is True:
        print("There is already SkyMapper data present for this field.")
        return True
    else:
        print("This field is not present in SkyMapper.")


def retrieve_skymapper_cutout(ra: float, dec: float):
    url = "http://api.skymapper.nci.org.au/aus/siap/dr3/query?"
