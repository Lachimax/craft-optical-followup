# Code by Lachlan Marnoch, 2019

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import photutils
import os

from astropy import wcs as wcs
from astropy.visualization import (ImageNormalize, LogStretch, SqrtStretch, ZScaleInterval, MinMaxInterval,
                                   PowerStretch, wcsaxes)
from astropy.table import Table
from typing import Union

from craftutils import fits_files as ff
from craftutils import params as p
from craftutils import astrometry as am


def plot_kron(fig: plt.figure, data_title: str, instrument: str, f: str, index: Union[int, list], catalogue: str,
              n: int, n_x: int, n_y: int,
              image_name: str, frame: Union[int, float], cmap: str = 'viridis', vmin: float = None, vmax: float = None,
              des: bool = False, offset_ra: int = 0, offset_dec: int = 0):
    table = Table().read(catalogue, format='ascii.csv')

    if type(index) is int:
        index = [index]

    centre_ra = float(np.mean(table['ra']))
    centre_dec = float(np.mean(table['dec']))
    centre_dec += offset_dec / 3600
    centre_dec += offset_ra / 3600

    fig, hdu = plot_galaxy(data_title=data_title,
                           instrument=instrument,
                           f=f,
                           ra=centre_ra,
                           dec=centre_dec,
                           frame=frame,
                           fig=fig,
                           n=n,
                           n_x=n_x,
                           n_y=n_y,
                           cmap=cmap,
                           show_cbar=False,
                           vmin=vmin,
                           vmax=vmax,
                           show_filter=False,
                           image_name=image_name,
                           show_instrument=False,
                           object_name=None,
                           ticks=None,
                           world_frame=True,
                           interval='zscale'
                           )

    for ind in index:

        ob = table[ind]

        if des:
            plot_gal_params(hdu=hdu,
                            ras=[ob['ra_des']], decs=[ob['dec_des']],
                            a=[ob['a_des'] / 3600],
                            b=[ob['b_des'] / 3600],
                            theta=[ob['theta_des']],
                            colour='blue',
                            world_axes=True)
            plot_gal_params(hdu=hdu,
                            ras=[ob['ra_des']], decs=[ob['dec_des']],
                            a=[ob['kron_radius_des'] * ob['a_des'] / 3600],
                            b=[ob['kron_radius_des'] * ob['b_des'] / 3600],
                            theta=[ob['theta_des']],
                            colour='red',
                            world_axes=True)
        else:
            plot_gal_params(hdu=hdu,
                            ras=[ob['ra']], decs=[ob['dec']],
                            a=[ob['a'] / 3600],
                            b=[ob['b'] / 3600],
                            theta=[ob['theta']],
                            colour='blue')
            plot_gal_params(hdu=hdu,
                            ras=[ob['ra']], decs=[ob['dec']],
                            a=[ob['kron_radius'] * ob['a'] / 3600],
                            b=[ob['kron_radius'] * ob['b'] / 3600],
                            theta=[ob['theta']],
                            colour='red')


def plot_difference(fig: plt.figure, path: str, obj: str, instrument: str,
                    frame: Union[int, float], world_frame: bool = True,
                    n: int = 1, n_y: int = 1, show_title: bool = True,
                    cmap: str = 'viridis', show_cbar: bool = False, stretch: str = 'sqrt', vmin: float = None,
                    vmax: float = None, show_grid: bool = False,
                    ticks: int = None, interval: str = 'minmax',
                    show_coords=False, show_frb: bool = False,
                    font_size: int = 12
                    ):
    params = p.object_params_frb(obj=obj)
    ra = params['hg_ra']
    dec = params['hg_dec']

    titles = ['FORS2', instrument, f'FORS2 - {instrument}']

    difference_path = path + filter(lambda f: "difference.fits" in f, os.listdir(path)).__next__()

    template = list(filter(lambda f: "template_tweaked.fits" in f, os.listdir(path)))
    if not template:
        template_file = filter(lambda f: "template_aligned.fits" in f, os.listdir(path)).__next__()
        print('Using reprojected template')
    else:
        template_file = template[0]
        print('Using aligned template')

    template_path = path + template_file

    comparison = list(filter(lambda f: "comparison_tweaked.fits" in f, os.listdir(path)))
    if not comparison:
        comparison_file = filter(lambda f: "comparison_aligned.fits" in f, os.listdir(path)).__next__()
        print('Using reprojected comparison')
    else:
        comparison_file = comparison[0]
        print('Using aligned comparison')

    comparison_path = path + comparison_file

    paths = [comparison_path, template_path, difference_path]
    path_names = ['Comparison', 'Template', 'Difference']

    for i, image_path in enumerate(paths):
        print()
        print(path_names[i])
        n_x = i + 1

        print(image_path)

        hdu = fits.open(image_path)
        header = hdu[0].header

        ylabel = None

        if show_title:
            title = titles[i]
            if n_x != 3:
                title = title + ', MJD ' + str(int(np.round(header['MJD-OBS'])))
            if show_frb and n_x == 1:
                ylabel = obj.replace('FRB', 'FRB\,')
        else:
            title = None

        plot, hdu = plot_subimage(fig=fig, hdu=hdu, ra=ra, dec=dec, frame=frame, world_frame=world_frame,
                                  title=title,
                                  n=(n - 1) * 3 + n_x, n_x=3, n_y=n_y, cmap=cmap, show_cbar=show_cbar, stretch=stretch,
                                  vmin=vmin,
                                  vmax=vmax, show_grid=show_grid, ticks=ticks, interval=interval,
                                  show_coords=show_coords, ylabel=ylabel, font_size=font_size)

        hdu.close()

    return fig


def plot_subimage(fig: plt.figure, hdu: Union[str, fits.HDUList], ra: float, dec: float,
                  frame: Union[int, float], world_frame: bool = False, title: str = None,
                  n: int = 1, n_x: int = 1, n_y: int = 1,
                  cmap: str = 'viridis', show_cbar: bool = False, stretch: str = 'sqrt', vmin: float = None,
                  vmax: float = None,
                  show_grid: bool = False,
                  ticks: int = None, interval: str = 'minmax',
                  show_coords: bool = True, ylabel: str = None,
                  font_size: int = 12,
                  reverse_y=False):
    """

    :param fig:
    :param hdu:
    :param ra:
    :param dec:
    :param frame: in pixels, or in arcsecs (?) if world_frame is True.
    :param world_frame:
    :param title:
    :param n:
    :param n_x:
    :param n_y:
    :param cmap:
    :param show_cbar:
    :param stretch:
    :param vmin:
    :param vmax:
    :param show_grid:
    :param ticks:
    :param interval:
    :param show_coords:
    :param ylabel:
    :param font_size:
    :param reverse_y:
    :return:
    """
    print(hdu)
    hdu, path = ff.path_or_hdu(hdu=hdu)
    print(hdu[0].data.shape)

    hdu_cut = ff.trim_frame_point(hdu=hdu, ra=ra, dec=dec, frame=frame, world_frame=world_frame)
    wcs_cut = wcs.WCS(header=hdu_cut[0].header)

    print(n_y, n_x, n)
    if show_coords:
        plot = fig.add_subplot(n_y, n_x, n, projection=wcs_cut)
        if ticks is not None:
            lat = plot.coords[0]
            lat.set_ticks(number=ticks)
    else:
        plot = fig.add_subplot(n_y, n_x, n)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.set_yticks([])
        # frame1.axes.get_yaxis().set_visible(False)

    if show_grid:
        plt.grid(color='black', ls='dashed')

    if type(vmin) is str:
        if vmin == 'median_full':
            vmin = np.nanmedian(hdu[0].data)
        elif vmin == 'median_cut':
            vmin = np.nanmedian(hdu_cut[0].data)
        else:
            raise ValueError('Unrecognised vmin string argument.')

    if interval == 'minmax':
        interval = MinMaxInterval()
    elif interval == 'zscale':
        interval = ZScaleInterval()
    else:
        raise ValueError('Interval not recognised.')

    print(hdu_cut[0].data.shape)
    if stretch == 'log':
        norm = ImageNormalize(hdu_cut[0].data, interval=interval, stretch=LogStretch(), vmin=vmin, vmax=vmax)
    elif stretch == 'sqrt':
        norm = ImageNormalize(hdu_cut[0].data, interval=interval, stretch=SqrtStretch(), vmin=vmin, vmax=vmax)
    else:
        raise ValueError('Stretch not recognised.')

    plot.title.set_text(title)
    plot.title.set_size(font_size)
    print(ylabel)
    if ylabel is not None:
        plot.set_ylabel(ylabel, size=12)

    im = plt.imshow(hdu_cut[0].data, norm=norm, cmap=cmap)
    if reverse_y:
        plot.invert_yaxis()
    c_ticks = np.linspace(norm.vmin, norm.vmax, 5, endpoint=True)
    if show_cbar:
        cbar = plt.colorbar(im)  # ticks=c_ticks)

    return plot, hdu_cut


def plot_galaxy(fig: plt.figure, data_title: str, instrument: str, f: str, ra: float, dec: float,
                frame: Union[int, float], world_frame: bool = False,
                n: int = 1, n_x: int = 1, n_y: int = 1,
                cmap: str = 'viridis', show_cbar: bool = False, stretch: str = 'sqrt', vmin: float = None,
                vmax: float = None,
                show_filter: bool = True, show_instrument: bool = False,
                show_grid: bool = False,
                image_name: str = 'astrometry_image',
                object_name: str = None, ticks: int = None, interval: str = 'minmax',
                show_coords=True,
                reverse_y=False):
    instrument = instrument.lower()
    instruments = {'fors2': 'FORS2', 'imacs': 'IMACS', 'xshooter': 'X-shooter', 'gmos': 'GMOS'}

    if instrument == 'imacs':
        f_0 = f
    else:
        f_0 = f[0]

    epoch_properties = p.object_params_instrument(obj=data_title, instrument=instrument)
    paths = p.object_output_paths(obj=data_title, instrument=instrument)

    if 'sdss' in image_name:
        instrument_formal = 'SDSS'
    elif 'des' in image_name:
        instrument_formal = 'DES'
    else:
        instrument_formal = instruments[instrument]

    print(f_0)
    print(image_name)
    print(f_0 + '_' + image_name)
    if f_0 + '_' + image_name in paths:
        path = paths[f_0 + '_' + image_name]
    elif f_0 + '_' + image_name in epoch_properties:
        path = epoch_properties[f_0 + '_' + image_name]
    else:
        raise ValueError(f'No path for {f_0}_{image_name} found.')

    title = object_name
    if title is None:
        title = ''
        if show_instrument:
            title += f'({instrument_formal}'
        if show_filter:
            if instrument == 'imacs':
                title += f'${f[-1]}$-band'
            else:
                title += f'${f_0}$-band'

    else:
        if show_filter:
            if instrument == 'imacs':
                title += f' (${f[-1]}$-band)'
            else:
                title += f' (${f_0}$-band)'
        if show_instrument:
            title = title.replace('(', f'({instrument_formal}, ')

    plot, hdu_cut = plot_subimage(fig=fig, hdu=path, ra=ra, dec=dec,
                                  frame=frame, world_frame=world_frame, title=title,
                                  n=n, n_x=n_x, n_y=n_y,
                                  cmap=cmap, show_cbar=show_cbar, stretch=stretch, vmin=vmin,
                                  vmax=vmax,
                                  show_grid=show_grid,
                                  ticks=ticks, interval=interval,
                                  show_coords=show_coords,
                                  reverse_y=reverse_y)

    return plot, hdu_cut


def plot_hg(data_title: str, instrument: str, f: str, frame: int,
            fig: plt.figure, n: int = 1, n_x: int = 1, n_y: int = 1,
            show_frb: Union[bool, str] = False, ellipse_colour: str = 'white',
            cmap: str = 'viridis', show_cbar: bool = False, stretch: str = 'sqrt', vmin: float = None,
            vmax: float = None,
            bar_colour: str = 'white',
            show_filter: bool = True,
            show_hg: bool = False, show_grid: bool = False, show_z: bool = True, z_colour: str = 'white',
            image_name: str = 'astrometry_image',
            show_instrument: bool = False,
            ticks: int = None,
            show_distance: bool = True, bar_position: str = 'left',
            show_coords: bool = True,
            show_name: bool = True,
            reverse_y=False):
    instrument = instrument.lower()
    instruments = {'fors2': 'FORS2', 'imacs': 'IMACS', 'xshooter': 'X-shooter', 'gmos': 'GMOS'}
    if instrument not in instruments:
        raise ValueError('Instrument not recognised.')

    burst_name = data_title[:-2]
    name = burst_name[3:]
    burst_properties = p.object_params_frb(burst_name)

    hg_ra = burst_properties['hg_ra']
    hg_dec = burst_properties['hg_dec']

    burst_ra = burst_properties['burst_ra']
    burst_dec = burst_properties['burst_dec']

    ang_size_distance = burst_properties['ang_size_distance']

    if show_name:
        object_name = f'HG\,{name}'
    else:
        object_name = ''

    plot, hdu_cut = plot_galaxy(data_title=data_title, instrument=instrument, f=f, ra=hg_ra, dec=hg_dec, frame=frame,
                                fig=fig,
                                n=n, n_x=n_x, n_y=n_y, cmap=cmap, show_cbar=show_cbar, stretch=stretch, vmin=vmin,
                                vmax=vmax,
                                show_grid=show_grid,
                                show_filter=show_filter, image_name=image_name, show_instrument=show_instrument,
                                object_name=object_name, ticks=ticks, show_coords=show_coords, reverse_y=reverse_y)

    if show_z:
        if reverse_y:
            plt.text(frame / 15, frame / 5, f'z = {burst_properties["z"]}', color=z_colour)
        else:
            plt.text(frame / 15, frame * 2 - frame / 5, f'z = {burst_properties["z"]}', color=z_colour)

    # cbar.set_label('ADU/s', rotation=270)

    if show_distance:
        if bar_position == 'left':

            distance_bar(hdu=hdu_cut, ang_size_distance=ang_size_distance, angle_length=1.0, x=frame / 15,
                         y=frame / 4.5,
                         colour=bar_colour, spread=frame / 10, reverse_y=reverse_y, frame=frame)
        elif bar_position == 'right':
            distance_bar(hdu=hdu_cut, ang_size_distance=ang_size_distance, angle_length=1.0, x=frame * 2 - frame / 1.8,
                         y=frame / 4.5,
                         colour=bar_colour, spread=frame / 10, reverse_y=reverse_y, frame=frame)
        else:
            raise ValueError('Bar position not recognised.')

    if show_frb is True:
        show_frb = 'quadrature'

    if show_frb == 'all':
        # Statistical
        a, b, theta = am.calculate_error_ellipse(burst_properties, error='statistical')
        plot_gal_params(hdu=hdu_cut, ras=[burst_ra], decs=[burst_dec], a=[a],
                        b=[b],
                        theta=[-theta], colour=ellipse_colour, line_style='-')
        # Systematic
        a, b, theta = am.calculate_error_ellipse(burst_properties, error='systematic')
        plot_gal_params(hdu=hdu_cut, ras=[burst_ra], decs=[burst_dec], a=[a],
                        b=[b],
                        theta=[-theta], colour=ellipse_colour, line_style='--')
        # Quadrature
        a, b, theta = am.calculate_error_ellipse(burst_properties, error='quadrature')
        plot_gal_params(hdu=hdu_cut, ras=[burst_ra], decs=[burst_dec], a=[a],
                        b=[b],
                        theta=[-theta], colour=ellipse_colour, line_style=':')

    elif show_frb is not False:
        a, b, theta = am.calculate_error_ellipse(burst_properties, error=show_frb)
        plot_gal_params(hdu=hdu_cut, ras=[burst_ra], decs=[burst_dec], a=[a],
                        b=[b],
                        theta=[-theta], colour=ellipse_colour, line_style='-')

        # burst_x, burst_y = wcs_cut.all_world2pix(burst_ra, burst_dec, 0)
        # plt.scatter(burst_x, burst_y)
    if show_hg:
        wcs_cut = wcs.WCS(header=hdu_cut[0].header)
        hg_x, hg_y = wcs_cut.all_world2pix(hg_ra, hg_dec, 0)
        plt.scatter(hg_x, hg_y, marker='x', c='r')

    return fig, hdu_cut


def distance_bar(hdu: fits.hdu.HDUList, ang_size_distance: float, x: float, y: float, frame: int,
                 colour: str = 'white', length: float = None, angle_length: float = None, spread: float = 1.,
                 reverse_y=False):
    """
    Draw a projected distance bar on your plot.
    :param hdu:
    :param ang_size_distance: Cosmological angular size distance, in parsecs.
    :param length: In parsecs, the length of bar you want to show. One of
    :param angle_length: In arcsecs, the length of the bar you want to show.
    :param x: Position of bar in plot.
    :param y: Position of bar in plot.
    :param colour: Colour of bar and text.
    :return:
    """

    if angle_length is length is None:
        raise ValueError('Either length or angle_length must be provided.')
    if angle_length is not None and length is not None:
        raise ValueError('Only one of length or angle_length can be provided.')

    # Get the angular pixel scale (degrees per pixel)
    _, pix_angle_scale = ff.get_pixel_scale(hdu)
    # Get the distance pixel scale (parsecs per pixel)
    pix_length_scale = ff.projected_pix_scale(hdu=hdu, ang_size_distance=ang_size_distance)

    if angle_length is None:
        # Get the length of the bar in pixels.
        pix_length = length / pix_length_scale
        # Use the angular pixel scale to get the angular length of the bar
        angle_length = pix_length * pix_angle_scale

    else:
        # Convert to degrees.
        angle_length /= 3600
        # Get the length of the bar in pixels.
        pix_length = angle_length / pix_angle_scale
        # Use the distance pixel scale to get the projected length of the bar.
        length = pix_length * pix_length_scale

    angle_length *= 3600

    print('Projected length:', length, 'pc')
    print('Angular size:', angle_length, 'arcsecs')

    if reverse_y:
        plt.plot((x, x + pix_length), (2 * frame - y, 2 * frame - y), c=colour)
        plt.text(x, 2 * frame - y - spread, f'{np.round(length / 1000, 1)} kpc', color=colour)
        plt.text(x, 2 * frame - y + 1.7 * spread, f'{int(angle_length)} arcsec', color=colour)
    else:
        plt.plot((x, x + pix_length), (y, y), c=colour)
        plt.text(x, y + spread, f'{np.round(length / 1000, 1)} kpc', color=colour)
        plt.text(x, y - 1.7 * spread, f'{int(angle_length)} arcsec', color=colour)


def nice_norm(image: np.ndarray):
    return ImageNormalize(image, interval=ZScaleInterval(), stretch=SqrtStretch())


def log_norm(image: np.ndarray, a=2000):
    return ImageNormalize(image, interval=ZScaleInterval(), stretch=LogStretch(a))


def latex_setup():
    # Matplotlib and pyplot settings
    plt.rc('font', family='serif')

    plt.rc('font', weight='bold')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath} ']  # \usepackage{sfmath} \boldmath

    plt.rcParams['font.serif'] = ['Times']
    plt.rcParams['axes.linewidth'] = 2.


def plot_file(path: str, label: str = None, colour: str = None, show: bool = False):
    """
    Plots a simple two-column ascii file, with the first column as x and the second column as y.
    :param path: path of the file.
    :param label: label for the plot.
    :param colour: colour for the plot.
    :return:
    """
    file = np.genfromtxt(path)
    file = file.transpose()
    wavelengths = file[0]
    throughput = file[1]
    plt.plot(wavelengths, throughput, label=label, c=colour)
    if show:
        plt.legend()
        plt.show()


def plot_gal_params(hdu: fits.HDUList, ras: Union[list, np.ndarray, float], decs: Union[list, np.ndarray, float],
                    a: Union[list, np.ndarray, float], b: Union[list, np.ndarray, float],
                    theta: Union[list, np.ndarray, float], colour: str = 'white',
                    show_centre: bool = False,
                    label: str = None, world: bool = True, world_axes: bool = True, line_style='-'):
    """

    :param hdu:
    :param ras: In degrees.
    :param decs: In degrees.
    :param a: In degrees.
    :param b: In degrees.
    :param theta: In degrees, apparently.
    :param colour:
    :param offset:
    :param show_centre:
    :return:
    """
    # TODO: Rename parameters to reflect acceptance of pixel coordinates.

    _, pix_scale = ff.get_pixel_scale(hdu)
    n_y, n_x = hdu[0].data.shape
    header = hdu[0].header
    wcs_image = wcs.WCS(header=header)
    if world:
        xs, ys = wcs_image.all_world2pix(ras, decs, 0)
    else:
        xs = np.array(ras)
        ys = np.array(decs)

    theta = np.array(theta)
    theta = -theta + ff.get_rotation_angle(header)
    # Convert to radians
    theta = theta * np.pi / 180

    for i, x in enumerate(xs):
        if a[i] != 0 and b[i] != 0:
            if world_axes:
                ellipse = photutils.EllipticalAperture((x, ys[i]), a=a[i] / pix_scale, b=b[i] / pix_scale,
                                                       theta=theta[i])
            else:
                ellipse = photutils.EllipticalAperture((x, ys[i]), a=a[i], b=b[i], theta=theta[i])
            ellipse.plot(color=colour, label=label, ls=line_style)
            line_label = None
        else:
            line_label = label
        if show_centre:
            plt.plot((0.0, n_x), (ys[i], ys[i]), c=colour, label=line_label)
            plt.plot((x, x), (0.0, n_y), c=colour)


def plot_all_params(image: Union[str, fits.hdu.HDUList], cat: Union[str, Table, np.ndarray], show: bool = True,
                    cutout: bool = False, ra_key: str = "ALPHA_SKY", dec_key: str = "DELTA_SKY", a_key: str = "A_WORLD",
                    b_key: str = "B_WORLD", theta_key: str = "THETA_WORLD", kron: bool = False,
                    kron_key: str = "KRON_RADIUS"):
    """
    Plots
    :param image:
    :param cat:
    :param cutout: Plots a cutout of the image centre. Forces 'show' to True.
    :return:
    """

    if cutout:
        show = True

    image, path = ff.path_or_hdu(image)

    data = image[0].data

    wcs_image = wcs.WCS(header=image[0].header)

    plt.subplot(projection=wcs_image)
    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=SqrtStretch())
    plt.imshow(data, origin='lower', norm=norm, )
    plot_gal_params(hdu=image, ras=cat[ra_key], decs=cat[dec_key], a=cat[a_key],
                    b=cat[b_key],
                    theta=cat[theta_key], colour='red')
    if kron:
        plot_gal_params(hdu=image, ras=cat[ra_key], decs=cat[dec_key], a=cat[kron_key] * cat[a_key],
                        b=cat[kron_key] * cat[b_key],
                        theta=cat[theta_key], colour='violet')

    if show:
        plt.show()

    if cutout:
        n_x = data.shape[1]
        n_y = data.shape[0]

        mid_x = int(n_x / 2)
        mid_y = int(n_y / 2)

        left = mid_x - 45
        right = mid_x + 45
        bottom = mid_y - 45
        top = mid_y + 45

        gal = ff.trim(hdu=image, left=left, right=right, bottom=bottom, top=top)
        plt.imshow(gal[0].data)
        plot_gal_params(hdu=gal, ras=cat[ra_key], decs=cat[dec_key], a=cat[a_key],
                        b=cat[b_key],
                        theta=cat[theta_key], colour='red')
        if kron:
            plot_gal_params(hdu=image, ras=cat[ra_key], decs=cat[dec_key], a=cat[kron_key] * cat[a_key],
                            b=cat[kron_key] * cat[b_key],
                            theta=cat[theta_key], colour='violet')

        if show:
            plt.show()

    if path:
        image.close()
