# Code by Lachlan Marnoch, 2021
import urllib
import os
import time
from datetime import date, datetime
from json.decoder import JSONDecodeError
from typing import Union, Iterable

import cgi
import requests
import re

import astropy.units as units
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable
from astropy.time import Time
from astroquery import log

# log.setLevel("TRACE")

try:
    import astroquery.gemini as gemini
except ModuleNotFoundError:
    print("This version of astroquery does not support Gemini. Gemini functions will not be available.")
try:
    import astroquery.ipac.irsa.irsa_dust as irsa_dust
except ModuleNotFoundError:
    import astroquery.irsa_dust as irsa_dust
try:
    import astroquery.ipac.irsa as irsa
except ModuleNotFoundError:
    import astroquery.irsa as irsa
try:
    from pyvo import dal
except ModuleNotFoundError:
    print("Pyvo not installed. Functions using pyvo will not be available.")

import craftutils.params as p
import craftutils.utils as u


# import craftutils.observation.instrument as inst


def cat_columns(cat, f: str = None):
    cat = cat.lower()
    if f == "rank":
        f = {
            "delve": "r",
            "des": "r",
            "sdss": "r",
            "skymapper": "r",
            "panstarrs1": "r",
            "source-extractor": "r",
            "gaia": "g"
        }[cat]
    if f is not None:
        f = f[0]
    else:
        f = ""

    if cat == '2mass':
        f = f.lower()
        return {
            'mag_psf': f"{f}_m",  # 2MASS doesn't have psf-fit magnitudes, so we make do with the regular magnitudes
            'mag_psf_err': f"{f}_cmsig",
            'ra': 'ra',
            'dec': 'dec',
        }
    if cat == 'ztf':
        f = f.lower()
        return {
            'mag_psf': f"{f}_m",  # 2MASS doesn't have psf-fit magnitudes, so we make do with the regular magnitudes
            'mag_psf_err': f"{f}_cmsig",
            'ra': 'ra',
            'dec': 'dec',
        }
    if cat == 'delve':
        f = f.lower()
        return {
            'mag_auto': f"wavg_mag_auto_{f}",
            'mag_auto_err': f"wavg_magerr_auto_{f}",
            'mag_psf': f"wavg_mag_psf_{f}",
            'mag_psf_err': f"wavg_magerr_psf_{f}",
            'ra': f"ra",
            'dec': f"dec",
            'class_star': f"class_star_{f}"}
    elif cat == 'des':
        f = f.upper()
        return {
            'mag_auto': f"MAG_AUTO_{f}",
            'mag_auto_err': f"MAGERR_AUTO_{f}",
            'mag_psf': f"WAVG_MAG_PSF_{f}",
            'mag_psf_err': f"WAVG_MAGERR_PSF_{f}",
            'ra': f"RA",
            'dec': f"DEC",
            'class_star': f"CLASS_STAR_{f}"}
    elif cat == 'gaia':
        return {
            'mag_auto': f"phot_{f}_mean_mag",
            'ra': f"ra",
            'dec': f"dec",
        }
    elif cat == 'panstarrs1':
        f = f.lower()
        return {
            'mag_auto': f"{f}KronMag",
            'mag_auto_err': f"{f}KronMagErr",
            'mag_psf': f"{f}PSFMag",
            'mag_psf_err': f"{f}PSFMagErr",
            'ra': f"raStack",
            'dec': f"decStack",
            'class_star': f"psfLikelihood"}
    elif cat == 'source-extractor':
        return {
            'mag_auto': "MAG_AUTO",
            'mag_auto_err': "MAGERR_AUTO",
            'mag_psf': "MAG_PSF",
            'mag_psf_err': "MAGERR_PSF",
            'ra': "RA",
            'dec': "DEC",
            'class_star': "CLASS_STAR"
        }
    elif cat == 'sdss':
        f = f.lower()
        return {
            'mag_psf': f"psfMag_{f}",
            'mag_psf_err': f"psfMagErr_{f}",
            'ra': f"ra",
            'dec': f"dec",
            'class_star': f"probPSF_{f}"}
    elif cat == 'skymapper':
        f = f.lower()
        return {
            'mag_psf': f"{f}_psf",
            'mag_psf_err': f"e_{f}_psf",
            'ra': f"raj2000",
            'dec': f"dej2000",
            'class_star': f"class_star_SkyMapper"}
    else:
        raise ValueError(f"Catalogue {cat} not recognised.")


cat_instruments = {
    "des": "decam",
    "delve": "decam",
    "panstarrs1": "panstarrs1",
    "sdss": "sdss",
    "skymapper": "skymapper"
}


def svo_filter_id(facility_name: str, instrument_name: str, filter_name: str) -> str:
    return f"{facility_name}/{instrument_name}.{filter_name}"


def retrieve_svo_filter(facility_name: str, instrument_name: str, filter_name: str):
    filter_id = svo_filter_id(
        facility_name=facility_name,
        instrument_name=instrument_name,
        filter_name=filter_name
    )
    print(f"Attempting to retrieve SVO filter data for {filter_id}...")
    url = f"http://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php?ID={filter_id}"
    try:
        response = requests.get(url).content
    except requests.exceptions.SSLError:
        print('An SSL error occurred when retrieving SVO data. Skipping.')
        return "ERROR"
    if b"ERROR" in response:
        return "ERROR"
    if response.count(b"\n") <= 1:
        return None
    else:
        return response


def save_svo_filter(facility_name: str, instrument_name: str, filter_name: str, output: str):
    """

    :param facility_name:
    :param instrument_name:
    :param filter_name:
    :param output:
    :return:
    """
    response = retrieve_svo_filter(
        facility_name=facility_name,
        instrument_name=instrument_name,
        filter_name=filter_name
    )
    u.debug_print(1, "retrieve.save_svo_filter(): response ==", response)
    if response == "ERROR":
        return response
    elif response is not None:
        u.mkdir_check_nested(path=output, remove_last=True)
        print("Saving SVO filter data to" + output)
        with open(output, "wb") as file:
            file.write(response)
    else:
        print('No data retrieved from SVO.')
    return response


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
    elif cat == 'panstarrs1':
        return update_std_mast_photometry(ra=ra, dec=dec, cat="panstarrs1")
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
    elif cat in mast_catalogues:
        return update_frb_mast_photometry(frb=frb, cat=cat)
    else:
        raise ValueError("Catalogue name not recognised.")


def save_catalogue(ra: float, dec: float, output: str, cat: str, radius: units.Quantity = 0.3 * units.deg):
    cat = cat.lower()
    if cat not in photometry_catalogues:
        raise KeyError(f"catalogue {cat} not recognised.")

    func = photometry_catalogues[cat]
    if func is save_mast_photometry:
        return func(ra=ra, dec=dec, output=output, cat=cat, radius=radius)
    else:
        return func(ra=ra, dec=dec, output=output, radius=radius)


# ESO retrieval code based on the script at
# http://archive.eso.org/programmatic/scripts/eso_authenticated_download_raw_and_calibs.py
# authored by A.Micol, Archive Science Group, ESO

eso_tap_url = "http://archive.eso.org/tap_obs"


def login_eso():
    if "eso_auth_token" not in keys:
        print("Attempting login to ESO archive.")
        token_url = "https://www.eso.org/sso/oidc/token"
        r = requests.get(token_url,
                         params={"response_type": "id_token token", "grant_type": "password",
                                 "client_id": "clientid",
                                 "username": keys["eso_user"], "password": keys["eso_pwd"]})
        try:
            token_response = r.json()
            token = token_response['id_token'] + '=='
            keys["eso_auth_token"] = token
            print("Login successful.")
        except JSONDecodeError:
            print("Login failed; either credentials are invalid, or there was a server-side error; skipping ESO tasks.")
            keys["eso_auth_token"] = None
    return keys["eso_auth_token"]


def save_eso_asset(file_url: str, output: str, filename: str = None):
    print("Downloading asset from:")
    print(file_url)
    headers = None
    token = login_eso()
    if token is not None:
        headers = {"Authorization": "Bearer " + keys["eso_auth_token"]}
        response = requests.get(file_url, headers=headers)
    else:
        # Trying to download anonymously
        response = requests.get(file_url, stream=True, headers=headers)

    if filename is None:
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition is not None:
            value, params = cgi.parse_header(content_disposition)
            filename = params["filename"]

        if filename is None:
            # last chance: get anything after the last '/'
            filename = file_url[file_url.rindex('/') + 1:]

    if response.status_code == 200:
        path = os.path.join(output, filename)
        print(f"Writing asset to {path}...")
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=50000):
                f.write(chunk)
        print("Done")
    else:
        response = None

    return response


def eso_calselector_info(description: str):
    """Parse the main calSelector description, and fetch: category, complete, certified, mode, and messages."""

    category = ""
    complete = ""
    certified = ""
    mode = ""
    messages = ""

    m = re.search('category="([^"]+)"', description)
    if m:
        category = m.group(1)
    m = re.search('complete="([^"]+)"', description)
    if m:
        complete = m.group(1).lower()
    m = re.search('certified="([^"]+)"', description)
    if m:
        certified = m.group(1).lower()
    m = re.search('mode="([^"]+)"', description)
    if m:
        mode = m.group(1).lower()
    m = re.search('messages="([^"]+)"', description)
    if m:
        messages = m.group(1)

    return category, complete, certified, mode, messages


def print_eso_calselector_info(description: str, mode_requested: str):
    """Print the most relevant params contained in the main calselector description."""

    category, complete, certified, mode_executed, messages = eso_calselector_info(description)

    alert = ""
    if complete != "true":
        alert = "ALERT: incomplete calibration cascade"

    mode_warning = ""
    if mode_executed != mode_requested:
        mode_warning = "WARNING: requested mode (%s) could not be executed" % (mode_requested)

    certified_warning = ""
    if certified != "true":
        certified_warning = "WARNING: certified=\"%s\"" % (certified)

    print("    calibration info:")
    print("    ------------------------------------")
    print(f"    science category={category}")
    print(f"    cascade complete={complete}")
    print(f"    cascade messages={messages}")
    print(f"    cascade certified={certified}")
    print(f"    cascade executed mode={mode_executed}")
    print(f"    full description: {description}")

    return alert, mode_warning, certified_warning


def save_eso_raw_data_and_calibs(
        output: str, program_id: str, date_obs: Union[str, Time],
        instrument: str, mode: str,
        obj: str = None,
        coord_tol: units.Quantity = 1.0 * units.arcmin
):
    u.mkdir_check(output)
    instrument = instrument.lower()
    login_eso()
    print(f"Querying the ESO TAP service at {eso_tap_url}")
    query = query_eso_raw(
        program_id=program_id, date_obs=date_obs, obj=obj, instrument=instrument, mode=mode, coord_tol=coord_tol
    )
    raw_frames = get_eso_raw_frame_list(query=query)
    calib_urls = get_eso_calib_associations_all(raw_frames=raw_frames)
    urls = list(raw_frames['url']) + calib_urls
    if not urls:
        print("No data was found in the raw ESO archive for the given parameters.")
    for url in urls:
        save_eso_asset(file_url=url, output=output)
    return urls


def count_epochs(dates: Iterable):
    epochs = []
    for d in dates:
        u.debug_print(2, d)
        d = Time(d)
        date_min = d - 0.25
        date_max = d + 0.25
        date_str = d.strftime('%Y-%m-%d')
        if date_str not in epochs and date_min.strftime('%Y-%m-%d') not in epochs and date_max.strftime(
                '%Y-%m-%d') not in epochs:
            epochs.append(date_str)
    return epochs


def get_eso_raw_frame_list(query: str):
    login_eso()
    tap_obs = dal.tap.TAPService(eso_tap_url)
    again = True
    raw_frames = Table()
    while again:
        # try:
        raw_frames = tap_obs.search(query=query)
        raw_frames = raw_frames.to_table()
        again = False
        # except dal.exceptions.DALQueryError:
        #     again = u.select_yn("The request timed out. Try again?")
    raw_frames['url'] = list(map(lambda r: f"https://dataportal.eso.org/dataportal_new/file/{r}", raw_frames['dp_id']))
    return raw_frames


def query_eso_raw(
        select: str = "dp_id,date_obs",
        program_id: str = None,
        date_obs: Union[str, Time] = None,
        obj: Union[str, SkyCoord] = None,
        coord_tol: units.Quantity = 1.0 * units.arcmin,
        instrument: str = "fors2",
        mode: str = "imaging"):
    instrument = instrument.lower()
    mode = mode.lower()
    if mode not in ["imaging", "spectroscopy"]:
        raise ValueError("Mode must be 'imaging' or 'spectroscopy'")
    mode_str = ""
    if instrument in ["fors2", "hawki"]:
        if mode == "imaging":
            mode_str = "dp_tech='IMAGE'"
        elif mode == "spectroscopy":
            mode_str = "dp_tech='SPECTRUM'"
    if instrument == "xshooter":
        if mode == "imaging":
            mode_str = "dp_tech='IMAGE'"
        elif mode == "spectroscopy":
            mode_str = "dp_tech like 'ECHELLE%'"

    if type(date_obs) is str:
        date_obs = Time(date_obs)
    query = f"""SELECT {select}
FROM dbo.raw
WHERE dp_cat='SCIENCE'
AND instrument='{instrument}'
AND {mode_str}
"""
    if program_id is not None:
        query += f"AND prog_id='{program_id}'"
    if date_obs is not None:
        query += f"AND date_obs>='{(date_obs - 0.5).to_datetime().date()}'\n" \
                 f"AND date_obs<='{(date_obs + 1).to_datetime().date()}'\n"
    if obj is not None:
        if isinstance(obj, str):
            query += f"AND target='{obj}'\n"
        elif isinstance(obj, SkyCoord):
            ra_min = obj.directional_offset_by(
                position_angle=90.0,
                separation=-coord_tol,
            ).ra.to(units.deg).value
            ra_max = obj.directional_offset_by(
                position_angle=90.0,
                separation=coord_tol,
            ).ra.to(units.deg).value
            dec_min = obj.directional_offset_by(
                position_angle=0.0,
                separation=-coord_tol,
            ).dec.to(units.deg).value
            dec_max = obj.directional_offset_by(
                position_angle=0.0,
                separation=coord_tol,
            ).dec.to(units.deg).value
            query += f"""AND ra>{ra_min}
AND ra<{ra_max}
AND dec>{dec_min}
AND dec<{dec_max}
"""
        else:
            raise TypeError(f"obj must be str or SkyCoord, not {type(obj)}")
    u.debug_print(1, "Query for ESO Archive:\n")
    u.debug_print(1, query)
    return query


def get_eso_calib_associations_all(raw_frames: Table, mode_requested: str = "raw2raw"):
    calib_urls = []
    for frame in raw_frames:
        calib_urls_this = get_eso_associations(raw_frame=frame['dp_id'], mode_requested=mode_requested)
        for url in calib_urls_this:
            if url not in calib_urls:
                calib_urls.append(url)
    return calib_urls


def get_eso_associations(raw_frame: str, mode_requested: str = "raw2raw"):
    print(f"Searching for associated calibration frames for {raw_frame}...")
    # Get list of calibration files associated with the raw frame.
    calselector_url = f"http://archive.eso.org/calselector/v1/associations?dp_id={raw_frame}&mode={mode_requested}&responseformat=votable"
    datalink = dal.adhoc.DatalinkResults.from_result_url(calselector_url)
    # this_description = next(datalink.bysemantics('#this')).description
    # Print cascade information and main description
    # alert, mode_warning, certified_warning = print_eso_calselector_info(this_description, mode_requested)
    # create and use a mask to get only the #calibration entries:
    calibrators = datalink['semantics'] == '#calibration'
    calib_urls = datalink.to_table()[calibrators]['access_url']
    print("Done.")
    return calib_urls


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
    print("Retrieving calibration parameters from FORS2 QC1 archive...")
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
            p.ingest_eso_filter_properties(path=path, instrument='FORS2', update=True)


def retrieve_irsa_xml(ra: float, dec: float):
    """
    Retrieves the extinction parameters for a given sky position from the IRSA Dust Tool
    (https://irsa.ipac.caltech.edu/applications/DUST/)
    :param ra: Right Ascension of the desired field, in degrees.
    :param dec: Declination of the desired field, in degrees.
    :return: XML-formatted string.
    """
    u.debug_print(1, "Retrieving IRSA data for position:", ra, dec)
    url = f"https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr={ra}+{dec}+equ+j2000"
    print("\nRetrieving IRSA Dust Tool XML from", url)
    irsa_xml = urllib.request.urlopen(url)
    irsa_xml = irsa_xml.read()
    return str(irsa_xml, 'utf-8')


def retrieve_irsa_details(ra: float = None, dec: float = None, coord: SkyCoord = None):
    if coord is None:
        if ra is None or dec is None:
            raise ValueError("Either ra & dec or coord must be provided.")
        coord = SkyCoord(ra * units.deg, dec * units.deg)

    return irsa_dust.IrsaDust.get_query_table(coord)


def retrieve_irsa_extinction(ra: float = None, dec: float = None, coord: SkyCoord = None):
    """
    Retrieves the extinction per bandpass table, and other relevant parameters, for a given sky position from the
    IRSA Dust Tool (https://irsa.ipac.caltech.edu/applications/DUST/).
    :param ra: Right Ascension of the desired field, in degrees.
    :param dec: Declination of the desired field, in degrees.
    :return: Tuple: dictionary of retrieved values, table-formatted string.
    """
    if coord is None:
        if ra is None or dec is None:
            raise ValueError("Either ra & dec or coord must be provided.")
        coord = SkyCoord(ra * units.deg, dec * units.deg)

    print(f"Retrieving IRSA extinction table for {coord}")
    table = None
    attempts = 0
    while table is None and attempts < 100:
        try:
            table = irsa_dust.IrsaDust.get_extinction_table(coord)
        except urllib.error.HTTPError:
            attempts += 1
            print(f"Could not retrieve table due to HTML error. Trying again after clearing cache ({attempts=}/100).")
            cache_path = os.path.join(p.home_path, ".astropy", "cache", "astroquery")
            u.rmtree_check(cache_path)

    if table is None:
        table = irsa_dust.IrsaDust.get_extinction_table(coord)

    return table


def save_irsa_extinction(output: str, ra: float = None, dec: float = None, coord: SkyCoord = None,
                         fmt: str = "ascii.ecsv"):
    """
    Retrieves the extinction per bandpass table for a given sky position from the IRSA Dust Tool
    (https://irsa.ipac.caltech.edu/applications/DUST/) and writes it to disk.
    :param ra: Right Ascension of the desired field, in degrees.
    :param dec: Declination of the desired field, in degrees.
    :param output: The location on disk to which to write the file.
    :return: Tuple: dictionary of retrieved values, table-formatted string.
    """
    table = retrieve_irsa_extinction(ra=ra, dec=dec, coord=coord)
    table.write(output, format=fmt)
    return table


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
    if 'dust_ebv' not in outputs and not os.path.isfile(os.path.join(data_dir, "galactic_extinction.txt")):
        values, ext_str = save_irsa_extinction(ra=params['burst_ra'], dec=params['burst_dec'],
                                               output=data_dir + "galactic_extinction.txt")
        p.add_output_values_frb(obj=frb, params=values)
        return values, ext_str
    else:
        print("IRSA Dust Tool data already retrieved.")


def retrieve_irsa_photometry(
        catalogue: str,
        ra: float,
        dec: float,
        radius: units.Quantity = 0.2 * units.deg,
):
    print(f"Querying IRSA archive for {catalogue} sources centred on RA={ra}, DEC={dec}.")
    table = irsa.Irsa.query_region(
        SkyCoord(
            ra,
            dec,
            unit=(units.deg, units.deg)
        ),
        catalog=catalogue, radius=radius
    )

    return table


def save_irsa_photometry(
        catalogue: str,
        ra: float,
        dec: float,
        output: str,
        radius: units.Quantity = 0.2 * units.deg):
    table = retrieve_irsa_photometry(
        catalogue=catalogue,
        ra=ra,
        dec=dec,
        radius=radius
    )
    if len(table) > 0:
        u.mkdir_check_nested(path=output)
        print(f"Saving {catalogue} catalogue to {output}")
        table.write(output, format="ascii.csv")
        return str(table)
    else:
        print(f"No data retrieved from {catalogue}")
        return None


def save_2mass_photometry(ra: float, dec: float, output: str, radius: units.Quantity = 0.2 * units.deg):
    return save_irsa_photometry(
        catalogue="fp_psc",
        ra=ra,
        dec=dec,
        output=output,
        radius=radius
    )


def save_ztf_photometry(ra: float, dec: float, output: str, radius: units.Quantity = 0.2 * units.deg):
    """
    NOTE: This does not currently work, due to a problem on the astroquery end.
    :param ra:
    :param dec:
    :param output:
    :param radius:
    :return:
    """
    return save_irsa_photometry(
        catalogue="ztf_objects_dr10",
        ra=ra,
        dec=dec,
        output=output,
        radius=radius
    )


sdss_filters = ["u", "g", "r", "i", "z"]


def retrieve_sdss_photometry(ra: float, dec: float, radius: units.Quantity = 0.2 * units.deg):
    """
    Retrieve SDSS photometry for a given field, in a 0.2 x 0.2 degree box centred on the passed coordinates
    coordinates. (Note - the width of the box is in RA degrees, not corrected for spherical distortion)
    :param ra: Right Ascension of the centre of the desired field, in degrees.
    :param dec: Declination of the centre of the desired field, in degrees.
    :return: Retrieved photometry table, as a pandas dataframe, if successful; if not, None.
    """
    radius = u.dequantify(radius, unit=units.deg)
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


def save_sdss_photometry(ra: float, dec: float, output: str, radius: units.Quantity = 0.2 * units.deg):
    """
    Retrieves and writes to disk the SDSS photometry for a given field, in a 0.2 x 0.2 degree box
    centred on the field coordinates. (Note - the width of the box is in RA degrees, not corrected for spherical
    distortion)
    :param ra: Right Ascension of the centre of the desired field, in degrees.
    :param dec: Declination of the centre of the desired field, in degrees.
    :param output: The location on disk to which to write the file.
    :return: Retrieved photometry table, as a pandas dataframe, if successful; if not, None.
    """
    df = retrieve_sdss_photometry(ra=ra, dec=dec, radius=radius)
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


def retrieve_delve_photometry(ra: float, dec: float, radius: units.Quantity = 0.2 * units.deg):
    print(f"\nQuerying DELVE DR2 archive for field centring on RA={ra}, DEC={dec}")
    radius = u.dequantify(radius, unit=units.deg)
    url = f"http://datalab.noirlab.edu/tap/sync?REQUEST=doQuery&lang=ADQL&FORMAT=csv&QUERY=SELECT%20q3c_dist" \
          f"%28ra%2Cdec%2C%20247.725%2C-0.972%29%2A3600%20AS%20dist%2C%20%2A%20FROM%20delve_dr2.objects%20WHERE%20%27t" \
          f"%27%20%3D%20Q3C_RADIAL_QUERY%28ra%2C%20dec%2C{ra}%2C{dec}%2C{radius}%29%20"
    try:
        response = requests.get(url).content
    except requests.exceptions.SSLError:
        print('An SSL error occurred when retrieving DELVE data. Skipping.')
        return "ERROR"
    except requests.exceptions.ConnectionError:
        print('A connection error occurred when retrieving DELVE data. Skipping.')
        return "ERROR"
    if b"ERROR" in response:
        return "ERROR"
    if response.count(b"\n") <= 1:
        return None
    else:
        return response


def save_delve_photometry(ra: float, dec: float, output: str, radius: units.Quantity = 0.2 * units.deg):
    response = retrieve_delve_photometry(ra=ra, dec=dec, radius=radius)
    if response == "ERROR":
        return response
    elif response is not None:
        u.mkdir_check_nested(path=output)
        print("Saving DELVE photometry to" + output)
        with open(output, "wb") as file:
            file.write(response)
    else:
        print('No data retrieved from DELVE.')
    return response


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
    try:
        js = r.json()
        if js['status'] == 'error':
            raise PermissionError(js['message'])
        keys['des_auth_token'] = js['token']
    except JSONDecodeError:
        print("Login failed; either credentials are invalid, or there was a server-side error; skipping DES tasks.")
        return 'ERROR'
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


def retrieve_des_photometry(ra: float, dec: float, radius: units.Quantity = 0.2 * units.deg):
    """
    Retrieve DES photometry for a given field, in a 2*radius squared degree box centred on the passed coordinates
    coordinates. (Note - the width of the box is in RA degrees, not corrected for spherical distortion)
    :param ra: Right Ascension of the centre of the desired field, in degrees.
    :param dec: Declination of the centre of the desired field, in degrees.
    :return: Retrieved photometry table, as a Bytes object, if successful; None if not.
    """
    print(f"Querying DES DR2 archive for field centring on RA={ra}, DEC={dec}")
    error = login_des()
    if error == "ERROR":
        return error

    radius = u.dequantify(radius, unit=units.deg)
    query = f"SELECT * " \
            f"FROM DR2_MAGNITUDE " \
            f"WHERE " \
            f"RA BETWEEN {ra - radius} and {ra + radius} and " \
            f"DEC BETWEEN {dec - radius} and {dec + radius} and " \
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


def save_des_photometry(ra: float, dec: float, output: str, radius: units.Quantity = 0.2 * units.deg):
    """
    Retrieves and writes to disk the DES photometry for a given field, in a 0.2 x 0.2 degree box
    centred on the field coordinates. (Note - the width of the box is in RA degrees, not corrected for spherical
    distortion)
    :param ra: Right Ascension of the centre of the desired field, in degrees.
    :param dec: Declination of the centre of the desired field, in degrees.
    :param output: The location on disk to which to write the file.
    :return: Retrieved photometry table, as a Bytes object, if successful; None if not.
    """
    data = retrieve_des_photometry(ra=ra, dec=dec, radius=radius)
    if data is not None and data != "ERROR":
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
        if response != "ERROR":
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
        if response != "ERROR":
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
    error = ""
    if "in_des" not in outputs:
        error = update_frb_des_photometry(frb=frb)
        outputs = p.frb_output_params(obj=frb)
    if error != "ERROR" and (force or outputs["in_des"]):
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


def retrieve_skymapper_photometry(ra: float, dec: float, radius: units.Quantity = 0.2 * units.deg):
    print(f"\nQuerying SkyMapper DR3 archive for field centring on RA={ra}, DEC={dec}")
    radius = u.dequantify(radius, unit=units.deg)
    url = f"http://skymapper.anu.edu.au/sm-cone/aus/query?RA={ra}&DEC={dec}&SR={radius}&RESPONSEFORMAT=CSV"
    try:
        response = requests.get(url).content
    except requests.exceptions.SSLError:
        print('An SSL error occurred when retrieving SkyMapper data. Skipping.')
        return "ERROR"
    if b"ERROR" in response:
        return "ERROR"
    if response.count(b"\n") <= 1:
        return None
    else:
        return response


def save_skymapper_photometry(ra: float, dec: float, output: str, radius: units.Quantity = 0.2 * units.deg):
    response = retrieve_skymapper_photometry(ra=ra, dec=dec, radius=radius)
    if response == "ERROR":
        return response
    elif response is not None:
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
    path = field_path + "SkyMapper/SkyMapper.csv"
    if outputs is None or "in_skymapper" not in outputs or force:
        response = save_skymapper_photometry(ra=ra, dec=dec, output=path)
        params = {}
        if response != "ERROR":
            if response is not None:
                params["in_skymapper"] = True
            else:
                params["in_skymapper"] = False
            p.add_params(file=field_path + "output_values", params=params)
            return response
        else:
            return None
    elif outputs["in_skymapper"] is True:
        if os.path.isfile(path):
            print("There is already SkyMapper data present for this field.")
            return True
        else:
            raise FileNotFoundError(f"Catalogue expected at {path}, but not found; something has gone wrong.")
    else:
        print("This field is not present in SkyMapper.")


def update_frb_skymapper_photometry(frb: str, force: bool = False):
    """
    Retrieve SkyMapper photometry for the field of an FRB (with a valid param file in param_dir), in a 0.2 deg radius cone
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
        if response != "ERROR":
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


mast_url = "https://catalogs.mast.stsci.edu/api/v0.1/"
catalogue_filters = {
    "panstarrs1": ["g", "r", "i", "z", "y"],
    "gaia": []}
catalogue_columns = {
    "panstarrs1": [
        "objID",
        "qualityFlag",
        "raStack", "decStack",
        "raStackErr", "decStackErr",
        "{:s}PSFMag", "{:s}PSFMagErr",
        "{:s}ApMag", "{:s}ApMagErr",
        "{:s}ApRadius",
        "{:s}KronMag", "{:s}KronMagErr",
        "{:s}psfLikelihood"],
    "gaia": [
        "astrometric_primary_flag",
        "ra", "ra_error", "dec", "dec_error",
        "b",
        "duplicated_source", "hip", "l", "matched_observations",
        "parallax", "parallax_error",
        "pmdec", "pmdec_error", "pmra", "pmra_error",
        "phot_g_mean_flux", "phot_g_mean_flux_error", "phot_g_mean_mag", "phot_g_n_obs",
        "phot_variable_flag",
        "random_index", "ref_epoch", "solution_id", "source_id", "tycho2_id"
    ]}


def construct_columns(cat="panstarrs1"):
    cat = cat.lower()
    columns = catalogue_columns[cat]
    filters = catalogue_filters[cat]
    columns_build = []
    for column in columns:
        if "{:s}" not in column:
            columns_build.append(column)
        else:
            for f in filters:
                columns_build.append(column.format(f))

    return columns_build


def retrieve_mast_photometry(ra: float, dec: float, cat: str = "panstarrs1", release="dr2", table="stack",
                             radius: units.Quantity = 0.1 * units.deg):
    if cat.lower() == "panstarrs1":
        cat_str = "panstarrs"
    else:
        cat_str = cat.lower()

    radius = u.dequantify(radius, unit=units.deg)
    print(f"\nQuerying {cat} {release} archive for field centring on RA={ra}, DEC={dec}, with radius {radius}")
    cat = cat.lower()
    url = f"{mast_url}{cat_str}/{release}/{table}.csv"
    print(url)
    request = {'ra': ra, 'dec': dec, 'radius': radius, 'columns': construct_columns(cat=cat)}
    response = requests.get(url, params=request)
    text = response.text
    if text == '':
        return None
    else:
        return text


def save_mast_photometry(ra: float, dec: float, output: str, cat: str = "panstarrs1",
                         radius: units.Quantity = 0.1 * units.deg):
    response = retrieve_mast_photometry(ra=ra, dec=dec, cat=cat, radius=radius)
    if response == "ERROR":
        return response
    elif isinstance(response, str) and "404 Not Found" in response:
        return "ERROR"
    elif response is not None:
        u.mkdir_check_nested(path=output)
        print(f"Saving {cat} photometry to {output}")
        with open(output, "w") as file:
            file.write(response)
    else:
        print(f'No data retrieved from {cat}.')
    return response


def update_std_mast_photometry(ra: float, dec: float, cat: str = "panstarrs1", force: bool = False):
    cat = cat.lower()
    data_dir = p.config['top_data_dir']
    field_path = f"{data_dir}/std_fields/RA{ra}_DEC{dec}/"
    u.mkdir_check_nested(field_path)
    outputs = p.load_params(field_path + "output_values")
    path = f"{field_path}{cat.upper()}/{cat.upper()}.csv"
    if outputs is None or f"in_{cat}" not in outputs or force:
        response = save_mast_photometry(ra=ra, dec=dec, output=path, cat=cat)
        params = {}
        if response != "ERROR":
            if response is not None:
                params[f"in_{cat}"] = True
            else:
                params[f"in_{cat}"] = False
            p.add_params(file=field_path + "output_values", params=params)
            return response
        else:
            return None
    elif outputs[f"in_{cat}"] is True:
        if os.path.isfile(path):
            print(f"There is already {cat} data present for this field.")
            return True
        else:
            raise FileNotFoundError(f"Catalogue expected at {path}, but not found; something has gone wrong.")
    else:
        print(f"This field is not present in {cat}.")


# TODO: A lot of basically repeated code here. Might be good to consolidate it somehow.

def update_frb_mast_photometry(frb: str, cat: str = "panstarrs1", force: bool = False):
    """
    Retrieve MAST photometry for the field of an FRB (with a valid param file in param_dir), in a 0.2 deg radius cone
    centred on the FRB coordinates, and download it to the appropriate data directory.
    (Note - the width of the box is in RA degrees, not corrected for spherical distortion)
    :param frb: FRB name, FRBXXXXXX. Must match title of param file.
    :return: True if successful, False if not.
    """
    params = p.object_params_frb(frb)
    path = f"{params['data_dir']}{cat.upper()}/{cat.upper()}.csv"
    outputs = p.frb_output_params(obj=frb)
    if outputs is None or f"in_{cat}" not in outputs or force:
        response = save_mast_photometry(ra=params['burst_ra'], dec=params['burst_dec'], output=path, cat=cat)
        params = {}
        if response != "ERROR":
            if response is not None:
                params[f"in_{cat}"] = True
            else:
                params[f"in_{cat}"] = False
            p.add_output_values_frb(obj=frb, params=params)
        return response
    elif outputs[f"in_{cat}"] is True:
        print(f"There is already {cat} data present for this field.")
        return True
    else:
        print(f"This field is not present in {cat}.")


def retrieve_gaia(ra: float, dec: float, radius: units.Quantity = 0.5 * units.deg):
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1
    print(f"\nQuerying Gaia DR2 archive for field centring on RA={ra}, DEC={dec}")
    coord = SkyCoord(ra=ra, dec=dec, unit=(units.degree, units.degree), frame='icrs')
    j = Gaia.cone_search_async(coordinate=coord, radius=radius)
    r = j.get_results()
    return r


def save_gaia(ra: float, dec: float, output: str, radius: units.Quantity = 0.5 * units.deg):
    table = retrieve_gaia(ra=ra, dec=dec, radius=radius)
    if len(table) > 0:
        u.mkdir_check_nested(path=output)
        print(f"Saving GAIA catalogue to {output}")
        table.write(output, format="ascii.csv", overwrite=True)
        return str(table)
    else:
        print("No data retrieved from Gaia DR2")
        return None


def update_frb_gaia(frb: str, force: bool = False):
    params = p.object_params_frb(frb)
    path = f"{params['data_dir']}Gaia/Gaia.csv"
    outputs = p.frb_output_params(obj=frb)
    if outputs is None or f"in_gaia" not in outputs or force:
        response = save_gaia(ra=params['burst_ra'], dec=params['burst_dec'], output=path)
        params = {}
        if response is not None:
            params[f"in_gaia"] = True
        else:
            params[f"in_gaia"] = False
        p.add_output_values_frb(obj=frb, params=params)
        return response
    elif outputs[f"in_gaia"] is True:
        print(f"There is already Gaia data present for this field.")
        return True
    else:
        print(f"This field is not present in Gaia.")


def load_catalogue(cat_name: str, cat: str):
    cat = u.path_or_table(cat, fmt="ascii.csv", load_qtable=True)
    cat_column_units = column_units[cat_name]
    cat_filters = filters[cat_name]
    for col_name in cat_column_units:
        if "{:s}" in col_name:
            for fil in cat_filters:
                cat[col_name.format(fil)] = cat[col_name.format(fil)] * cat_column_units[col_name]
        else:
            cat[col_name] = cat[col_name] * cat_column_units[col_name]
    return cat


def login_gemini():
    gemini.Observations.login(keys["gemini_user"], keys["gemini_pwd"])


def save_gemini_calibs(output: str, obs_date: Time, instrument: str = 'GSAOI', fil: str = "Kshort", overwrite=False):
    flats = {}
    date_early = obs_date.copy()
    date_late = obs_date.copy()
    while len(flats) == 0:
        program_id = f"GS-CAL{date_early.strftime('%Y%m%d')}"
        print(f"Searching for {fil} domeflats in {program_id}...")
        flats = gemini.Observations.query_criteria(
            instrument=instrument,
            observation_class='dayCal',
            program_id=program_id,
        )
        print(f"Found {len(flats)} flats total.")
        flats = flats[flats["filter_name"] == fil]
        print(f"Found {len(flats)} flats for filter {fil}.")
        if len(flats) == 0:
            program_id = f"GS-CAL{date_late.strftime('%Y%m%d')}"
            print(f"Searching for {fil} domeflats in {program_id}...")
            flats = gemini.Observations.query_criteria(
                instrument=instrument,
                observation_class='dayCal',
                program_id=f"GS-CAL{date_late.strftime('%y%m%d')}",
            )
            flats = flats[flats["filter_name"] == fil]
            print(f"Found {len(flats)} flats total.")
            flats = flats[flats["filter_name"] == fil]
            print(f"Found {len(flats)} flats for filter {fil}.")
        date_early -= 1
        date_late += 1

    save_gemini_files(flats, output=output, overwrite=overwrite)

    print(f"Found flats for filter {fil}:")
    print(flats)

    standards = {}

    date_early = obs_date.copy()
    date_late = obs_date.copy()
    while len(standards) == 0 and (obs_date - date_early) < 365:
        program_id = f"GS-CAL{date_early.strftime('%Y%m%d')}"
        print(f"Searching for {fil} standards in {program_id}...")
        standards = gemini.Observations.query_criteria(
            instrument=instrument,
            observation_class="partnerCal",
            program_id=f"GS-CAL{date_early.strftime('%y%m%d')}",
        )
        print(f"Found {len(standards)} standards total.")
        standards = standards[standards["filter_name"] == fil]
        print(f"Found {len(standards)} standards for filter {fil}.")
        if len(standards) == 0:
            program_id = f"GS-CAL{date_late.strftime('%Y%m%d')}"
            print(f"Searching for {fil} standards in {program_id}...")
            standards = gemini.Observations.query_criteria(
                instrument=instrument,
                observation_class="partnerCal",
                program_id=f"GS-CAL{date_late.strftime('%y%m%d')}",
            )
            standards = standards[standards["filter_name"] == fil]
            print(f"Found {len(standards)} standards total.")
            standards = standards[standards["filter_name"] == fil]
            print(f"Found {len(standards)} standards for filter {fil}.")
        date_early -= 1
        date_late += 1

    standards = gemini.Observations.query_criteria(
        instrument=instrument,
        observation_class="partnerCal",
        program_id=f"GS-CAL{date_early.strftime('%y%m%d')}",
    )

    print("Found standards:")
    print(standards)

    standards = standards[standards["filter_name"] == fil]

    save_gemini_files(standards, output=output, overwrite=overwrite)


def save_gemini_epoch(output: str, program_id: str, coord: SkyCoord,
                      instrument: str = 'GSAOI', overwrite: bool = False):
    science_files = gemini.Observations.query_criteria(
        instrument=instrument,
        program_id=program_id,
        observation_class='science',
        coordinates=coord,
    )

    save_gemini_files(science_files, output=output, overwrite=overwrite)
    return science_files


def save_gemini_files(file_list: Table, output: str, overwrite: bool = False):
    login_gemini()
    print(f"overwrite == {overwrite}")
    for row in file_list:
        name = row["filename"].replace(".bz2", "")
        if not os.path.isfile(os.path.join(output, name)) or overwrite:
            gemini.Observations.get_file(name, download_dir=output)
        else:
            print(f"Skipping {name}, file already exists.")


filters = {
    "gaia": ["g", "bp", "rp"]}

column_units = {
    "gaia":  # See https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
        {
            "ra": units.deg,
            "ra_error": units.milliarcsecond,
            "dec": units.deg,
            "dec_error": units.milliarcsecond,
            "parallax": units.milliarcsecond,
            "parallax_error": units.milliarcsecond,
            "pmra": units.milliarcsecond / units.year,
            "pmra_error": units.milliarcsecond / units.year,
            "pmdec": units.milliarcsecond / units.year,
            "pmdec_error": units.milliarcsecond / units.year,
            "astrometric_excess_noise": units.milliarcsecond,
            "astrometric_weight_al": units.milliarcsecond ** 2,
            "astrometric_pseudo_colour": units.micrometer ** -1,
            "astrometric_sigma5d_max": units.milliarcsecond,
            "phot_{:s}_mean_flux": units.electron / units.second,
            "phot_{:s}_mean_flux_error": units.electron / units.second,
            "phot_{:s}_mean_mag": units.mag,
            "bp_rp": units.mag,
            "bp_g": units.mag,
            "g_rp": units.mag,
            "radial_velocity": units.kilometer / units.second,
            "radial_velocity_error": units.kilometer / units.second,
            "rv_template_teff": units.Kelvin,
            "rv_template_fe_h": units.dex,
            "l": units.deg,
            "b": units.deg,
            "ecl_lon": units.deg,
            "ecl_lat": units.deg,
            "teff_val": units.Kelvin,
            "teff_percentile_lower": units.Kelvin,
            "teff_percentile_upper": units.Kelvin,
            "a_g_val": units.mag,
            "a_g_percentile_lower": units.mag,
            "a_g_percentile_upper": units.mag,
            "e_bp_min_rp_val": units.mag,
            "e_bp_min_rp_percentile_lower": units.mag,
            "e_bp_min_rp_percentile_upper": units.mag,
            "radius_val": units.solRad,
            "radius_percentile_lower": units.solRad,
            "radius_percentile_upper": units.solRad,
            "lum_val": units.solLum,
            "lum_percentile_lower": units.solLum,
            "lum_percentile_upper": units.solLum,
        }

}

keys = p.keys()
fors2_filters_retrievable = ["I_BESS", "R_SPEC", "b_HIGH", "v_HIGH"]
mast_catalogues = ['panstarrs1']
photometry_catalogues = {
    '2mass': save_2mass_photometry,
    'delve': save_delve_photometry,
    'des': save_des_photometry,
    'sdss': save_sdss_photometry,
    'skymapper': save_skymapper_photometry,
    'panstarrs1': save_mast_photometry,
    'gaia': save_gaia}
