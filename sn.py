import sncosmo
import numpy as np
from matplotlib import pyplot as plt
from typing import Union
from scipy import stats
from astropy import table
from astropy.io import fits
from astropy import wcs

import PyCRAFT.photometry as ph
import PyCRAFT.utils as u
from PyCRAFT import stats as st
import PyCRAFT.params as p
import PyCRAFT.fits_files as ff

sn_types = ['ia', 'ib', 'ic', 'iil', 'iip', 'iin']


def delta_m15(mags, days):
    day_peak = days[np.nanargmin(mags)]
    day_15 = day_peak + 15.
    arg_day_15, _ = u.find_nearest(days, day_15)
    mag_peak = np.nanmin(mags)
    mag_15 = float(mags[arg_day_15])
    return mag_15 - mag_peak


def x1_to_delta_m15(x1):
    return 1.09 - 0.161 * x1 + 0.013 * x1 ** 2 - 0.00130 * x1 ** 3


def find_model_times(source: str, show=False, fil: str = 'desg', z: float = 1):
    # This function always intiates a new model - a bit wasteful, but I couldn't see a way around it.
    days = np.arange(-135, 136)

    model = sncosmo.Model(source=source)
    model.set(z=z)
    # If B-band is outside the rest wavelength range of our model, it'll throw an error; we return None if that happens.
    try:
        mags = model.bandmag(fil, 'ab', days)
    except ValueError:
        return None, None

    # If the magnitudes our model is giving out are flat, something is clearly wrong, so we return None
    # (this occurs for some s11 models at redshifts > 0.2)
    if np.sum(mags == mags[0]) == len(mags):
        return None, None

    mag_peak = np.nanmin(mags)
    if np.isnan(mag_peak):
        return None, None
    t_peak = days[np.nanargmin(mags)]
    t_first = np.ceil(model.mintime())
    print('Peak:', t_peak, 'days, mag', mag_peak)
    print('First day:', t_first, 'days')
    if show:
        plt.plot(days, mags)
        plt.xlabel('Time (days)')
        plt.ylabel('Magnitude')
        plt.plot([t_peak, t_peak], [np.nanmin(mags), np.nanmax(mags)], c='blue', label='peak')
        plt.plot([t_first, t_first], [np.nanmin(mags), np.nanmax(mags)], c='red', label='start')
        plt.gca().invert_yaxis()
        plt.show()

    return t_first, t_peak


def find_model_peak(source: str, show=False, fil: str = 'bessellb', z: float = 1, t_start=0):
    days = np.arange(t_start, 50)
    model = sncosmo.Model(source=source)
    model.set(z=z)
    mags = model.bandmag(fil, 'ab', days)
    t_peak = days[np.nanargmin(mags)]
    if show:
        plt.plot(days, mags)
        plt.plot([t_peak, t_peak], [np.nanmin(mags), np.nanmax(mags)], c='blue')
        plt.gca().invert_yaxis()
        plt.show()
    return t_peak


def check_model(source: Union[sncosmo.Model, str]):
    if type(source) is str:
        return sncosmo.Model(source=source), False
    elif type(source) is sncosmo.Model:
        return source, True


def light_curves_salt2(z: float, filters: list, peak: float = None, days: Union[tuple, list, np.ndarray] = (0, 85),
                       show: bool = True, rise_time: float = None, ebv_mw: float = 0, ebv_host: float = 0.,
                       x1: float = None, output_path: str = None, output_title: str = None, day_markers: list = None,
                       fil_peak='bessellb', r_v: float = 2.3):
    """ Produces light curves, in the provided filters, for a Type Ia supernova using the SALT2 models as implemented 
    in sncosmo. 
    :param z: Redshift of source.
    :param filters: Filters to obtain light curves in. Must be in the sncosmo Registry.
    :param peak: Peak (ie lowest) absolute magnitude, in the filter given by fil_peak, to calibrate curves to.
    :param days: Either an array of times (in days) to calculate the light curves over, or a tuple describing the range
        of days, ie (first_day, last_day)
    :param show: Show plot onscreen?
    :param rise_time: Rest-frame time, from beginning of SN to peak magnitude, in days.
    :param ebv_mw: Reddening parameter E(B-V), using S&F11 law, for the Milky Way along the SN's line-of-sight.
    :param ebv_host: Reddening parameter E(B-V), using S&F11 law, for the host galaxy.
    :param x1: SALT2 light curve stretch parameter. See SALT2 documentation for further information.
    :param output_path: Path to which to save output. If None, does not save.
    :param output_title: Title to give output plot and table.
    :param day_markers: List of times (in days) to mark on plot, eg observation dates.
    :param fil_peak: Filter in which to set the peak absolute magnitude; usually reported in B or V.
    :return: mag_table, model
        mag_table: an astropy.table.Table with the times, in days, and the magnitudes in each filter.
        model: the entire sncosmo model instance.
    """
    if output_path is not None and output_path[-1] != '/':
        output_path += '/'
    u.mkdir_check(output_path)
    # Find time at which model begins and time of peak.
    t_first, t_peak = find_model_times(source='salt2-extended', z=z, fil=fil_peak, show=False)
    # If both are None, the model is invalid over these wavelengths
    if t_first is t_peak is None:
        return None, None
    # Set up model in sncosmo.
    dust_mw = sncosmo.F99Dust()
    dust_host = sncosmo.CCM89Dust()
    model = sncosmo.Model(source='salt2-extended', effects=[dust_mw, dust_host],
                          effect_names=['mw', 'host'],
                          effect_frames=['obs', 'rest'])
    model.set(x1=x1)
    # If a rise time is not given, allow the model to determine this itself.
    if rise_time is None:
        if t_peak <= 0:
            model.set(z=z, t0=-t_first, mwebv=ebv_mw, hostebv=ebv_host, hostr_v=r_v)
        else:
            model.set(z=z, t0=0, mwebv=ebv_mw, hostebv=ebv_host, hostr_v=r_v)
    elif rise_time == -1:
        model.set(z=z, t0=t_first, mwebv=ebv_mw, hostebv=ebv_host, hostr_v=r_v)
    else:
        # Correct rise time to observer frame
        rise_time_obs = rise_time * (1 + z)
        model.set(z=z, t0=rise_time_obs - t_peak, mwebv=ebv_mw, hostebv=ebv_host, hostr_v=r_v)
    print(model)
    # Set peak absolute magnitude of model.
    if peak is not None:
        model.set_source_peakabsmag(peak, fil_peak, 'ab')
    if type(days) is tuple:
        # Set up array of times.
        days = np.arange(days[0], days[1] + 1, 0.1)

    mags_filters = table.Table()
    mags_filters['days'] = days
    maxes = []
    t_peaks = []
    peaks = []
    for f in filters:
        # Get light curve.
        mags = model.bandmag(f, 'ab', days)
        # If the light curve is mostly flat, the model has probably broken down.
        if np.sum(mags == mags[0]) < 0.9 * len(mags):
            # If this is False, the entire light curve must be nan, and we'll get nothing useful out of it.
            if not np.isnan(np.nanmax(mags)):
                # Collect peak (lowest) magnitudes, peak times, and maximum (faintest) magnitudes for each filter.
                maxes.append(np.nanmax(mags[mags != np.inf]))
                peaks.append(np.nanmin(mags[mags != np.inf]))
                t_peaks.append(days[np.nanargmin(mags)])
                # Write light curve to table.
                mags_filters[f] = mags
                if output_path is not None or show:
                    # Plot curve.
                    plt.plot(days, mags, label=f)

    # If we have no maxima, the model has broken down.
    if len(maxes) > 0:
        # Collect this for plotting purposes.
        max_mag = np.nanmax(maxes)
        min_mag = np.nanmin(peaks)
    else:
        return None, None

    # If an output_path directory is not given and 'show' is not True, there's no point doing the plot.
    if output_path is not None or show:
        for i, t_peak in enumerate(t_peaks):
            # Plot blue lines marking the peak of each filter light curve.
            plt.plot([t_peak, t_peak], [max_mag + 1, min_mag - 1], c='blue')

        if day_markers is not None:
            for other_day in day_markers:
                # Plot red lines marking the observation dates.
                plt.plot([other_day, other_day], [max_mag + 1, min_mag - 1], c='red')
        plt.xlabel('Time (days)')
        plt.ylabel('Magnitude')
        plt.ylim(max_mag + 1, min_mag - 1)
        plt.legend()
        if output_path is not None:
            # Save figure.
            plt.savefig(output_path + output_title + '.png')
            # Save table to csv.
            mags_filters.write(output_path + output_title + '.csv', format='ascii.csv')
        if show:
            # Show figure onscreen.
            plt.show()
        plt.close()

    return mags_filters, model


def light_curves(source: str, z: float, filters: list, peak: float = None,
                 days: Union[tuple, list, np.ndarray] = (0, 85), show=True, rise_time: float = None, ebv_mw=0.,
                 ebv_host=0., output_path: str = None, output_title: str = None, day_markers: list = None,
                 fil_peak='bessellb'):
    """ Produces light curves, in the provided filters, for a Type Ia supernova using the SALT2 models as implemented 
        in sncosmo. 
        :param source: 
        :param z: Redshift of source.
        :param filters: Filters to obtain light curves in. Must be in the sncosmo Registry.
        :param peak: Peak (ie lowest) absolute magnitude, in the filter given by fil_peak, to calibrate curves to.
        :param days: Either an array of times (in days) to calculate the light curves over, or a tuple describing the range
            of days, ie (first_day, last_day)
        :param show: Show plot onscreen?
        :param rise_time: Rest-frame time, from beginning of SN to peak magnitude, in days.
        :param ebv_mw: Reddening parameter E(B-V), using S&F11 law, for the Milky Way along the SN's line-of-sight.
        :param ebv_host: Reddening parameter E(B-V), using S&F11 law, for the host galaxy.
        :param output_path: Path to which to save output. If None, does not save.
        :param output_title: Title to give output plot and table.
        :param day_markers: List of times (in days) to mark on plot, eg observation dates.
        :param fil_peak: Filter in which to set the peak absolute magnitude; usually reported in B or V.
        :return: mag_table, model
            mag_table: an astropy.table.Table with the times, in days, and the magnitudes in each filter.
            model: the entire sncosmo model instance.
        """
    if output_path is not None and output_path[-1] != '/':
        output_path += '/'
    u.mkdir_check(output_path)

    print(source.upper())
    # Find time at which model begins and time of peak.
    t_first, t_peak = find_model_times(source=source, z=z, fil=fil_peak, show=False)
    # If both are None, the model is invalid over these wavelengths
    if t_first is t_peak is None:
        return None, None, None
    # Set up model in sncosmo.
    dust = sncosmo.F99Dust()
    model = sncosmo.Model(source=source, effects=[dust, dust],
                          effect_names=['mw', 'host'],
                          effect_frames=['obs', 'rest'])
    # If a rise time is not given, allow the model to determine this itself.
    if rise_time is None:
        if t_peak <= 0:
            model.set(z=z, t0=-t_first, mwebv=ebv_mw, hostebv=ebv_host)
        else:
            model.set(z=z, t0=0, mwebv=ebv_mw, hostebv=ebv_host)
    elif rise_time == -1:
        model.set(z=z, t0=t_first, mwebv=ebv_mw, hostebv=ebv_host)
    else:
        # Correct rise time to observer frame
        rise_time_obs = rise_time * (1 + z)
        model.set(z=z, t0=rise_time_obs - t_peak, mwebv=ebv_mw, hostebv=ebv_host)
    print(model)
    # Set peak absolute magnitude of model.
    if peak is not None:
        model.set_source_peakabsmag(peak, fil_peak, 'ab')
    if type(days) is tuple:
        # Set up array of times.
        days = np.arange(days[0], days[1] + 1, 0.1)

    mags_filters = table.Table()
    mags_filters['days'] = days
    maxes = []
    t_peaks = []
    peaks = []
    for f in filters:
        # Get light curve.
        mags = model.bandmag(f, 'ab', days)
        # If the light curve is mostly flat, the model has probably broken down.
        if np.sum(mags == mags[0]) < 0.9 * len(mags):
            # If this is False, the entire light curve must be nan, and we'll get nothing useful out of it.
            if not np.isnan(np.nanmax(mags)):
                # Collect peak (lowest) magnitudes, peak times, and maximum (faintest) magnitudes for each filter.
                maxes.append(np.nanmax(mags))
                peaks.append(np.nanmin(mags))
                t_peaks.append(days[np.nanargmin(mags)])
                # Write light curve to table.
                mags_filters[f] = mags
                if output_path is not None or show:
                    # Plot curve.
                    plt.plot(days, mags, label=f)

    # If we have no maxima, the model has broken down.
    if len(maxes) > 0:
        # Collect this for plotting purposes.
        max_mag = np.nanmax(maxes)
        min_mag = np.nanmin(peaks)
    else:
        return None, None, None

    # If an output_path directory is not given and 'show' is not True, there's no point doing the plot.
    if output_path is not None or show:
        for i, t_peak in enumerate(t_peaks):
            # Plot blue lines marking the peak of each filter light curve.
            plt.plot([t_peak, t_peak], [max_mag + 1, min_mag - 1], c='blue')

        if day_markers is not None:
            for other_day in day_markers:
                # Plot red lines marking the observation dates.
                plt.plot([other_day, other_day], [max_mag + 1, min_mag - 1], c='red')
        plt.title(source)
        plt.xlabel('Time (days)')
        plt.ylabel('Magnitude')
        plt.ylim(max_mag + 1, min_mag - 1)
        plt.legend()
        if output_path is not None:
            # Save figure.
            plt.savefig(output_path + output_title + '.png')
            # Save table to csv.
            mags_filters.write(output_path + output_title + '.csv', format='ascii.csv')
        if show:
            # Show figure onscreen.
            plt.show()
        plt.close()

    return days, mags_filters, model


def light_curve_panels(days, sources: list, sources_offset: list, sources_zero: list, rise_time: float, filters: list,
                       peak: float,
                       z: float = 0, markers: list = None):
    temp_mags = None

    models = {}

    for source in sources:
        model = sncosmo.Model(source=source)
        model.set(z=z, t0=rise_time)
        model.set_source_peakabsmag(peak, 'bessellb', 'ab')
        print(model)
        models[source] = model

    for i, source in enumerate(sources_offset):
        offset = 0
        if source[:4] == 's11-':
            offset = -8
        elif source[:6] == 'snana-':
            offset = -20
        elif source[:7] == 'nugent-':
            offset = 26
        model = sncosmo.Model(source=source)
        model.set(z=z, t0=rise_time - offset)
        model.set_source_peakabsmag(peak, 'bessellb', 'ab')
        print(model)
        models[source] = model

    for i, source in enumerate(sources_zero):
        model = sncosmo.Model(source=source)
        model.set(z=z, t0=0)
        model.set_source_peakabsmag(peak, 'bessellb', 'ab')
        print(model)
        models[source] = model

    fig = plt.figure()

    first = True
    for i, f in enumerate(filters):
        band_plot = fig.add_subplot(2, 2, i + 1)
        for model_name in models:
            model = models[model_name]
            print(model_name)
            mags = model.bandmag(f, 'ab', days)
            band_plot.plot(days, mags, label=model_name)
            handles, labels = band_plot.get_legend_handles_labels()
            band_plot.scatter(days[np.nanargmin(mags)], np.nanmin(mags))

            if first:
                temp_mags = mags
                first = False

        if markers is not None:
            for marker in markers:
                band_plot.plot([marker, marker], [np.nanmax(temp_mags), np.nanmin(temp_mags)], c='red')
        band_plot.set_ylim(np.nanmax(temp_mags), np.nanmin(temp_mags) - 1)
        band_plot.set_title(f)

    fig.legend(handles, labels, loc='upper center')

    return fig


def random_position(image: fits.HDUList, hg_ra: float, hg_dec: float, limit: int = 10, show: bool = False):
    image, path = ff.path_or_hdu(image)

    header = image[0].header
    data = image[0].data
    # Find host galaxy pixel
    wcs_info = wcs.WCS(header)
    hg_x, hg_y = wcs_info.all_world2pix(hg_ra, hg_dec, 0)
    hg_x = int(np.round(hg_x))
    hg_y = int(np.round(hg_y))

    image_copy = data.copy()[hg_y - limit:hg_y + limit, hg_x - limit:hg_x + limit]

    noise = np.median(image_copy)
    image_copy = image_copy - noise

    image_flatten = image_copy.flatten(1)

    i = st.value_from_pdf(np.arange(image_flatten.shape[0]), image_flatten / max(image_flatten))
    i = int(i)
    x, y = np.unravel_index(i, image_copy.shape)

    if show:
        plt.imshow(image_copy)
        plt.scatter(x, y)
        plt.show()
        plt.close()
        plt.plot(image_flatten / max(image_flatten))
        plt.show()

    x += hg_x - limit + np.random.uniform(-0.5, 0.5)
    y += hg_y - limit + np.random.uniform(-0.5, 0.5)

    ra, dec = wcs_info.all_pix2world(x, y, 0)

    if path:
        image.close()

    return x, y, ra, dec


def random_ebv(tau: float = 0.67, r_v: float = 2.3):
    # Set up range of x
    x = np.arange(0, 2, 0.01)
    # Get curve of PDF.
    a_exp = st.exponential(x=x, a=0.99, c=r_v / tau, d=0)
    # Get random E(B-V) value from PDF.
    ebv = st.value_from_pdf(x, a_exp)

    return ebv


peak_mus = {'ia': -19.09, 'ib': -17.45, 'ic': -17.66, 'iib': -16.99, 'iil': -17.98, 'iip': -16.75, 'iin': -18.53}
peak_sigmas = {'ia': 0.62, 'ib': 1.12, 'ic': 1.18, 'iib': 0.92, 'iil': 0.86, 'iip': 0.98, 'iin': 1.32}


def random_peak(sn_type: str, mu: float = None, sigma: float = None):
    """
    Generate a supernova peak absolute magnitude (in B-band) from a right-handed Gumbel distribution, to account for
    the subluminous tail.
    Type Ia properties from Ashall 2016, MNRAS 460, 3529; albeit, derived as a normal distribution.
    Other properties from Richardson 2014, ApJ 147(5), 118
    :param mu: float: Mean of distribution.
    :param sigma: Standard deviation of distribution.
    :return:
    """
    if mu is None:
        mu = peak_mus[sn_type]
    if sigma is None:
        sigma = peak_sigmas[sn_type]

    return stats.gumbel_r.rvs(loc=mu, scale=sigma)


rise_time_mus = {'ia': 18.03, 'ib': 18.03, 'ic': 18.03, 'iib': 7.5, 'iil': 7.5, 'iip': 7.5, 'iin': 7.5}
rise_time_sigmas = {'ia': 0.24, 'ib': 0.24, 'ic': 0.24, 'iib': 3.5, 'iil': 3.5, 'iip': 3.5, 'iin': 3.5}


def random_rise_time(sn_type: str = 'ia', mu: float = None, sigma: float = None):
    """
    Generate a supernova rise time from a normal distribution.
    Defaults for Type Ia from Ganeshalingam et al, 2011, MNRAS 416, 2607.
    :param sn_type: Type of supernova, eg 'ia', 'iip'.
    :return: rise_time: float: in days
    """
    if sn_type not in sn_types:
        raise ValueError('Accepted sn_types are', sn_types)

    # Select distribution properties.
    if mu is None:
        mu = rise_time_mus[sn_type]
    if sigma is None:
        sigma = rise_time_sigmas[sn_type]

    # Get random value from distribution.
    rise_time = stats.norm.rvs(loc=mu, scale=sigma)
    return rise_time


def random_x1(mu: float = 0.604, sigma_minus: float = 1.029, sigma_plus: float = 0.363):
    """
    Generate a Type Ia supernova SALT2 x1 (stretch) value, using an asymmetric Gaussian distribution,
    per Scolnic et al, 2016, ApJ 822, L35. Defaults are from PS1 distribution, same paper.
    :param mu: Mean of the distribution.
    :param sigma_minus: Sigma of the negative side of the distribution.
    :param sigma_plus: Sigma of the positive side of the distribution.
    :return: x1: float
    """
    # Set up range of x
    x = np.arange(-5, 4, 0.01)
    # Get curve of PDF.
    a_gauss = st.asymmetric_gaussian(x=x, mu=mu, sigma_minus=sigma_minus, sigma_plus=sigma_plus)
    # Get random x1 value from PDF.
    x1 = st.value_from_pdf(x, a_gauss / max(a_gauss))
    return x1


def random_light_curves_type_ia(filters: list, image: Union[str, fits.HDUList], hg_ra: float, hg_dec: float,
                                z: float,
                                peak: float = None,
                                rise_time: float = None,
                                x1: float = None,
                                ebv_host: float = None,
                                ebv_mw: float = 0, day_markers: list = None,
                                fil_peak: str = 'bessellb', show: bool = False, output_path: str = None,
                                output_title: str = None,
                                x: float = None, y: float = None, ra: float = None, dec: float = None, limit: int = 10):
    """

    :param filters: Filters to obtain light curves in. Must be in the sncosmo Registry.
    :param z: Redshift of source.
    :param peak: Peak (ie lowest) absolute magnitude, in the filter given by fil_peak, to calibrate curves to.
        If not given, will be generated.
    :param rise_time: Rest-frame time, from beginning of SN to peak magnitude, in days. 
        If not given, will be generated.
    :param x1: SALT2 light curve stretch parameter. See SALT2 documentation for further information. If not given, will
        be generated.
    :param ebv_host: Reddening parameter E(B-V), using S&F11 law, for the host galaxy.
    :param ebv_mw: Reddening parameter E(B-V), using S&F11 law, for the Milky Way along the SN's line-of-sight.
    :param day_markers: List of times (in days) to mark on plot, eg observation dates.
    :param fil_peak: Filter in which to set the peak absolute magnitude; usually reported in B or V.
    :param show: Show plot onscreen?
    :param output_path: Path to which to save output. If None, does not save.
    :param output_title: Title to give output plot and table.

    :return: mag_table, model, x, y, tbl
        mag_table: astropy.table.Table: times, in days, and the magnitudes in each filter.
        model: the entire sncosmo model instance.
        x: float: x position, in pixels, of generated source.
        y: y position, in pixels, of generated source.
        tbl: astropy.table.Table containing the SN type and generated values.
    """
    # Generate a random separation using the image itself.
    if x is None or y is None:
        x, y, ra, dec = random_position(image=image, hg_ra=hg_ra, hg_dec=hg_dec, limit=limit, show=False)
    # Generate a rise time using a normal distribution.
    if rise_time is None:
        rise_time = random_rise_time(sn_type='ia')
    if x1 is None:
        # Generate an x1 value using an asymmetric Gaussian distribution
        x1 = random_x1()
    if peak is None:
        peak = random_peak(sn_type='ia')
    if ebv_host is None:
        ebv_host = random_ebv()

    mags_filters, model = light_curves_salt2(z=z, filters=filters, peak=peak, rise_time=rise_time, ebv_mw=ebv_mw,
                                             ebv_host=ebv_host, x1=x1, day_markers=day_markers, fil_peak=fil_peak,
                                             show=show, output_path=output_path, output_title=output_title)

    tbl = table.Table()

    tbl['ra'] = [ra]
    tbl['dec'] = [dec]
    tbl['type'] = ['ia']
    tbl['rise_time'] = [rise_time]
    tbl['x1'] = [x1]
    tbl['peak'] = [peak]
    tbl['ebv_host'] = [ebv_host]

    return mags_filters, model, x, y, tbl


def random_light_curves(sn_type: str, filters: list, image: Union[str, fits.HDUList], hg_ra: float, hg_dec: float,
                        z: float, source: str = None,
                        peak: float = None,
                        rise_time: float = None,
                        ebv_host: float = None,
                        ebv_mw: float = 0, day_markers: list = None,
                        fil_peak: str = 'bessellb', show: bool = False, output_path: str = None,
                        limit: int = 10, x: float = None, y: float = None, ra: float = None, dec: float = None,
                        output_title: str = None, sources: dict = None):
    """

    :param filters: Filters to obtain light curves in. Must be in the sncosmo Registry.
    :param z: Redshift of source.
    :param peak: Peak (ie lowest) absolute magnitude, in the filter given by fil_peak, to calibrate curves to.
        If not given, will be generated.
    :param rise_time: Rest-frame time, from beginning of SN to peak magnitude, in days.
        If not given, will be generated.
    :param ebv_host: Reddening parameter E(B-V), using S&F11 law, for the host galaxy.
    :param ebv_mw: Reddening parameter E(B-V), using S&F11 law, for the Milky Way along the SN's line-of-sight.
    :param day_markers: List of times (in days) to mark on plot, eg observation dates.
    :param fil_peak: Filter in which to set the peak absolute magnitude; usually reported in B or V.
    :param show: Show plot onscreen?
    :param output_path: Path to which to save output. If None, does not save.
    :param output_title: Title to give output plot and table.

    :return: mag_table, model, x, y, tbl
        mag_table: astropy.table.Table: times, in days, and the magnitudes in each filter.
        model: the entire sncosmo model instance.
        x: float: x position, in pixels, of generated source.
        y: y position, in pixels, of generated source.
        tbl: astropy.table.Table containing the SN type and generated values.
    """

    days = mags_filters = model = None

    while days is None or mags_filters is None or model is None:

        if source is None:
            if sources is None:
                sources = p.sncosmo_models()
            choose_sources = sources['type_' + sn_type]

            i = np.random.randint(low=0, high=len(choose_sources))
            source_gen = choose_sources[i]
        else:
            source_gen = source

        # Generate a random separation using the galaxy's image as the basis for the distribution.
        if x is None or y is None:
            x, y, ra, dec = random_position(image=image, hg_ra=hg_ra, hg_dec=hg_dec, limit=limit, show=False)
        # Generate a rise time using a normal distribution.
        if rise_time is None:
            rise_time_gen = random_rise_time(sn_type=sn_type)
        else:
            rise_time_gen = rise_time

        if peak is None:
            peak_gen = random_peak(sn_type=sn_type)
        else:
            peak_gen = peak

        if ebv_host is None:
            ebv_host_gen = random_ebv()
        else:
            ebv_host_gen = ebv_host

        days, mags_filters, model = light_curves(source=source_gen, z=z, filters=filters, peak=peak_gen,
                                                 rise_time=rise_time_gen,
                                                 ebv_mw=ebv_mw, ebv_host=ebv_host_gen, output_path=output_path,
                                                 output_title=output_title, day_markers=day_markers,
                                                 fil_peak=fil_peak, show=show)

    tbl = table.Table()

    tbl['ra'] = [ra]
    tbl['dec'] = [dec]
    tbl['type'] = [sn_type]
    tbl['source'] = source_gen
    tbl['rise_time'] = [rise_time_gen]
    tbl['peak'] = [peak_gen]
    tbl['ebv_host'] = [ebv_host_gen]

    return mags_filters, model, x, y, tbl


def magnitude_at_epoch(epoch, days, mags):
    arg_day, _ = u.find_nearest(days, epoch)
    mag = float(mags[arg_day])
    return mag


def register_filter(f: str, instrument: str = 'FORS2'):
    data = p.filter_params(instrument=instrument, f=f)
    bandpass = sncosmo.Bandpass(data['wavelengths_filter_only'], data['transmissions_filter_only'])
    sncosmo.register(bandpass, f)
