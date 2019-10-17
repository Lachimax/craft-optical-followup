# Code by Lachlan Marnoch, 2019
from astropy import table
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def clean_residuals(tbl, x_col: str, y_col: str, mag_col: str, bad=False):
    x_mode = stats.mode(np.round(tbl[x_col]))[0]
    y_mode = stats.mode(np.round(tbl[y_col]))[0]
    mag_mode = stats.mode(np.round(tbl[mag_col]))[0]

    remove = (np.round(tbl[x_col]) == x_mode) & (np.round(tbl[y_col]) == y_mode) & (np.round(tbl[mag_col]) == mag_mode)
    # print(np.sum(remove))

    if not bad:
        tbl = tbl[np.invert(remove)]

    else:
        tbl = tbl[remove]

    keep = np.invert(np.isnan(tbl[mag_col]))
    tbl = tbl[keep]

    return tbl, x_mode, y_mode, mag_mode


def percentage(tbl, failure=True, filters=['g_HIGH', 'I_BESS'], tolerance=3600 ** 2):
    percentages = {filters[0]: 0, filters[1]: 0}
    x_modes = {filters[0]: 0, filters[1]: 0}
    y_modes = {filters[0]: 0, filters[1]: 0}
    mag_modes = {filters[0]: 0, filters[1]: 0}
    tbls = {filters[0]: 0, filters[1]: 0}

    smallest = 100
    smallest_tbl = None
    largest = 0
    largest_tbl = None
    for f in percentages:
        x_col = f + '_nuclear_offset_pix_x'
        y_col = f + '_nuclear_offset_pix_y'
        mag_col = f + '_mag_recovered'
        tbl_new, x_mode, y_mode, mag_mode = clean_residuals(tbl, x_col, y_col, mag_col)
        # print(len(tbl_new))
        x_modes[f] = x_mode
        y_modes[f] = y_mode
        mag_modes[f] = mag_mode
        tbl_new = tbl_new[tbl_new[f + '_matching_distance_arcsec'] * tolerance < 1.0]
        # print(len(tbl_new))

        if len(tbl_new) <= smallest:
            smallest = len(tbl_new)
            smallest_tbl = tbl_new
        if len(tbl_new) >= largest:
            largest = len(tbl_new)
            largest_tbl = tbl_new

        if not failure:
            percentages[f] = len(tbl_new)
        else:
            percentages[f] = 100 - len(tbl_new)

        tbls[f] = tbl_new

    both = np.sum(np.isin(smallest_tbl['name'], largest_tbl['name']))
    if failure:
        both = 100 - both
    percentages['Both'] = both

    return percentages, tbls, x_modes, y_modes, mag_modes


path = "/home/user/Data/SS/"
path_180924 = path + "FRB180924/subtraction/"
path_181112 = path + "FRB181112/subtraction/"
path_190102 = path + "FRB190102/subtraction/"
path_190608 = path + "FRB190608/subtraction/"

ia_180924 = table.Table.read(path_180924 + "FORS2_1-FORS2_4_multi_sn_random_ia/recovery_table.csv", format='ascii.csv')
ia_181112 = table.Table.read(path_181112 + "FORS2_1-FORS2_3_multi_sn_random_ia/recovery_table.csv", format='ascii.csv')
ia_190102 = table.Table.read(path_190102 + "FORS2_1-FORS2_5_multi_sn_random_ia/recovery_table.csv", format='ascii.csv')
ia_190608 = table.Table.read(path_190608 + "XSHOOTER_1_nosub_multi_sn_random_ia/recovery_table.csv", format='ascii.csv')

ib_180924 = table.Table.read(path_180924 + "FORS2_1-FORS2_4_multi_sn_random_ib/recovery_table.csv", format='ascii.csv')
ib_181112 = table.Table.read(path_181112 + "FORS2_1-FORS2_3_multi_sn_random_ib/recovery_table.csv", format='ascii.csv')
ib_190102 = table.Table.read(path_190102 + "FORS2_1-FORS2_5_multi_sn_random_ib/recovery_table.csv", format='ascii.csv')

ic_180924 = table.Table.read(path_180924 + "FORS2_1-FORS2_4_multi_sn_random_ic/recovery_table.csv", format='ascii.csv')
ic_181112 = table.Table.read(path_181112 + "FORS2_1-FORS2_3_multi_sn_random_ic/recovery_table.csv", format='ascii.csv')
ic_190102 = table.Table.read(path_190102 + "FORS2_1-FORS2_5_multi_sn_random_ic/recovery_table.csv", format='ascii.csv')

iil_180924 = table.Table.read(path_180924 + "FORS2_1-FORS2_4_multi_sn_random_iil/recovery_table.csv",
                              format='ascii.csv')
iil_181112 = table.Table.read(path_181112 + "FORS2_1-FORS2_3_multi_sn_random_iil/recovery_table.csv",
                              format='ascii.csv')
iil_190102 = table.Table.read(path_190102 + "FORS2_1-FORS2_5_multi_sn_random_iil/recovery_table.csv",
                              format='ascii.csv')
iil_190608 = table.Table.read(path_190608 + "XSHOOTER_1_nosub_multi_sn_random_iil/recovery_table.csv",
                              format='ascii.csv')

iip_180924 = table.Table.read(path_180924 + "FORS2_1-FORS2_4_multi_sn_random_iip/recovery_table.csv",
                              format='ascii.csv')
iip_181112 = table.Table.read(path_181112 + "FORS2_1-FORS2_3_multi_sn_random_iip/recovery_table.csv",
                              format='ascii.csv')
iip_190102 = table.Table.read(path_190102 + "FORS2_1-FORS2_5_multi_sn_random_iip/recovery_table.csv",
                              format='ascii.csv')
iip_190608 = table.Table.read(path_190608 + "XSHOOTER_1_nosub_multi_sn_random_iip/recovery_table.csv",
                              format='ascii.csv')

iin_180924 = table.Table.read(path_180924 + "FORS2_1-FORS2_4_multi_sn_random_iin/recovery_table.csv",
                              format='ascii.csv')
iin_181112 = table.Table.read(path_181112 + "FORS2_1-FORS2_3_multi_sn_random_iin/recovery_table.csv",
                              format='ascii.csv')
iin_190102 = table.Table.read(path_190102 + "FORS2_1-FORS2_5_multi_sn_random_iin/recovery_table.csv",
                              format='ascii.csv')
iin_190608 = table.Table.read(path_190608 + "XSHOOTER_1_nosub_multi_sn_random_iin/recovery_table.csv",
                              format='ascii.csv')

fors2_180924 = table.Table.read(path_180924 + "FORS2_1-FORS2_4_multi_frb_range/recovery_table.csv", format='ascii.csv')
fors2_181112 = table.Table.read(path_181112 + "FORS2_1-FORS2_3_multi_frb_range/recovery_table.csv", format='ascii.csv')
fors2_190102 = table.Table.read(path_190102 + "FORS2_1-FORS2_5_multi_frb_range/recovery_table.csv", format='ascii.csv')

xshooter_180924 = table.Table.read(path_180924 + "FORS2_1-XSHOOTER_3_multi_frb_range/recovery_table.csv",
                                   format='ascii.csv')
xshooter_181112 = table.Table.read(path_181112 + "FORS2_1-XSHOOTER_2_multi_frb_range/recovery_table.csv",
                                   format='ascii.csv')
xshooter_190102 = table.Table.read(path_190102 + "FORS2_1-XSHOOTER_4_multi_frb_range/recovery_table.csv",
                                   format='ascii.csv')

big_dict = {'Type Ia': {'180924': ia_180924,
                        '181112': ia_181112,
                        '190102': ia_190102,
                        '190608': ia_190608},
            'Type Ib': {'180924': ib_180924,
                        '181112': ib_181112,
                        '190102': ib_190102},
            'Type Ic': {'180924': ic_180924,
                        '181112': ic_181112,
                        '190102': ic_190102},
            'Type IIn': {'180924': iin_180924,
                         '181112': iin_181112,
                         '190102': iin_190102,
                         '190608': iin_190608},
            'Type II-L': {'180924': iil_180924,
                          '181112': iil_181112,
                          '190102': iil_190102,
                          '190608': iil_190608},
            'Type II-P': {'180924': iip_180924,
                          '181112': iip_181112,
                          '190102': iip_190102,
                          '190608': iin_190608}}

for sn_type in big_dict:
    print(sn_type)
    combined = {'g': 1, 'I': 1}
    for obj in big_dict[sn_type]:
        if obj == '190608':
            filters = ['g_prime', 'I']
            tolerance = 1
        else:
            filters = ['g_HIGH', 'I_BESS']
            tolerance = 3600 ** 2
        percent, tbl, x_mode, y_mode, mag_mode = percentage(big_dict[sn_type][obj], failure=True, filters=filters,
                                                            tolerance=tolerance)
        print(obj, ':', percent)
        for f in filters:
            combined[f[0]] *= percent[f] / 100
    print('Combined:', combined)
