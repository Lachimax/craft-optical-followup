import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from craftutils import params as p
from craftutils import utils as u
from craftutils import stats

matplotlib.rcParams.update({'errorbar.capsize': 3})


def main(epoch, show, write):

    print("\nExecuting Python script pipeline_fors2/9-extinction_atmospheric.py, with:")
    print(f"\tepoch {epoch}")
    print(f"\tshow {show}")
    print(f"\twrite {write}")
    print()

    filters = p.instrument_all_filters('FORS2')
    epoch_params = p.object_params_fors2(obj=epoch)
    output_values = p.object_output_params(obj=epoch, instrument='FORS2')

    filters_known = np.array(['b_HIGH', 'v_HIGH', 'R_SPECIAL', 'I_BESS'])
    filters_find = np.array(['u_HIGH', 'g_HIGH', 'z_GUNN'])

    output = {}

    mjd = output_values['mjd_obs']
    print()
    print('EPOCH:', epoch)
    print('MJD:', mjd)
    lambda_effs_known = []
    extinctions_known = []
    extinctions_known_err = []
    zeropoints = []
    zeropoints_err = []
    for f in filters_known:
        print(f)
        f_params = filters[f]
        lambda_effs_known.append(f_params['lambda_eff'])
        i, nrst = u.find_nearest(f_params['mjd'], mjd)
        print('Nearest MJD in table:', nrst)
        extinctions_known.append(f_params['extinction'][i])
        extinctions_known_err.append(f_params['extinction_err'][i])
        zeropoints.append(f_params['zeropoint'][i])
        zeropoints_err.append(f_params['zeropoint_err'][i])

    lambda_effs_find = []
    for f in filters_find:
        f_params = filters[f]
        lambda_effs_find.append(f_params['lambda_eff'])

    interp_err = max(extinctions_known_err)

    print('EXTINCTIONS:')
    for i, f in enumerate(filters_known):
        print(f, ':', extinctions_known[i], '+/-', extinctions_known_err[i])
        output[f + '_extinction_provided'] = float(extinctions_known[i])
        output[f + '_extinction_provided_err'] = float(extinctions_known_err[i])

    # try:
    exp, cov = curve_fit(stats.exponential, lambda_effs_known, extinctions_known, p0=(1, 1e-2, 0))
    # except RuntimeError:

    extinctions_find = stats.exponential(np.array(lambda_effs_find), *exp)
    extinctions_known_exp = stats.exponential(np.array(lambda_effs_known), *exp)
    residuals = np.abs(extinctions_known_exp - np.array(extinctions_known))

    extinctions_find_err = []

    for i in range(len(filters_find)):
        if min(lambda_effs_known) <= lambda_effs_find[i] <= max(lambda_effs_known):
            extinctions_find_err.append(interp_err + np.max(residuals))
        else:
            extinctions_find_err.append(2 * (interp_err + np.max(residuals)))

    for i, f in enumerate(filters_find):
        print(f, ':', extinctions_find[i], '+/-', extinctions_find_err[i])
        output[f + '_extinction_interpolated'] = float(extinctions_find[i])
        output[f + '_extinction_interpolated_err'] = float(extinctions_find_err[i])

    print('ZEROPOINTS:')
    for i, f in enumerate(filters_known):
        print(f, ':', zeropoints[i], '+/-', zeropoints_err[i])
        output[f + '_zeropoint_provided'] = float(zeropoints[i])
        output[f + '_zeropoint_provided_err'] = float(zeropoints_err[i])

    ext_poly_space = np.poly1d(np.polyfit(lambda_effs_known, extinctions_known, deg=3))
    lambda_eff_fit = np.linspace(300, 1000)

    plot_params = p.plotting_params()
    size_font = plot_params['size_font']
    size_label = plot_params['size_label']
    size_legend = plot_params['size_legend']
    weight_line = plot_params['weight_line']
    width = plot_params['a4_width']

    major_x_ticks = np.arange(200, 1500, 100)
    minor_x_ticks = np.arange(200, 1500, 10)
    major_y_ticks = np.arange(-1, 1, 0.2)
    minor_y_ticks = np.arange(-1, 1, 0.1)

    fig = plt.figure(figsize=(width, 4))
    plot = fig.add_subplot(1, 1, 1)
    plt.xticks(major_x_ticks, labels=np.arange(2000, 15000, 1000))
    plot.set_yticks(major_y_ticks)
    plot.set_xticks(minor_x_ticks, minor=True)
    plot.set_yticks(minor_y_ticks, minor=True)

    plot.tick_params(axis='x', labelsize=size_label, pad=5)
    plot.tick_params(axis='y', labelsize=size_label)
    plot.tick_params(which='both', width=2)
    plot.tick_params(which='major', length=4)
    plot.tick_params(which='minor', length=2)

    plot.plot(lambda_eff_fit, ext_poly_space(lambda_eff_fit), label='Polynomial fit', c='violet', lw=weight_line,
              zorder=1)
    plot.plot(lambda_eff_fit, stats.exponential(lambda_eff_fit, *exp), label='Exponential fit', c='green',
              lw=weight_line, zorder=2)
    plot.scatter(lambda_effs_find, ext_poly_space(lambda_effs_find), c='violet', zorder=3)
    plot.scatter(lambda_effs_find, np.interp(lambda_effs_find, lambda_effs_known, extinctions_known),
                 label='Interpolated', c='cyan', zorder=4)
    plot.scatter(lambda_effs_find, stats.exponential(np.array(lambda_effs_find), *exp), c='green', zorder=5)
    # plt.title(epoch + ' Extinction Interpolation')
    plot.scatter(lambda_effs_known, extinctions_known, label='Provided by ESO', c='b', zorder=6)
    plot.set_xlabel('Filter $\lambda_\mathrm{eff}$ (angstrom)', fontsize=size_font, fontweight='bold')
    plot.set_ylabel('Extinction coefficient $k$', fontsize=size_font, fontweight='bold')
    plt.legend(fontsize=size_legend)
    fig.savefig(epoch_params['data_dir'] + '9-zeropoint/atmospheric_extinction.png', bbox_inches='tight')
    if show:
        plt.show()

    if write:
        update_dict = {}
        for i, f in enumerate(filters_find):
            if f in epoch_params['filters']:
                update_dict[f[0] + '_extinction'] = float(extinctions_find[i])
                update_dict[f[0] + '_extinction_err'] = float(extinctions_find_err[i])
        for i, f in enumerate(filters_known):
            if f in epoch_params['filters']:
                update_dict[f[0] + '_extinction'] = float(extinctions_known[i])
                update_dict[f[0] + '_extinction_err'] = float(extinctions_known_err[i])
                update_dict[f[0] + '_zeropoint_provided'] = float(zeropoints[i])
                update_dict[f[0] + '_zeropoint_provided_err'] = float(zeropoints_err[i])
        p.add_output_values(obj=epoch, params=update_dict, instrument='fors2')
    p.add_params(file=epoch_params['data_dir'] + '9-zeropoint/calibrations.yaml', params=output)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Write ')

    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('-show', action='store_true')
    parser.add_argument('-write', action='store_true')

    args = parser.parse_args()
    main(epoch=args.op, show=args.show, write=args.write)
