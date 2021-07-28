# Code by Lachlan Marnoch, 2019-2020
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os.path import isfile

from astropy.table import Table

from craftutils import params as p
from craftutils.retrieve import update_frb_irsa_extinction


def main(obj: str):
    matplotlib.rcParams.update({'errorbar.capsize': 3})

    frb_params = p.object_params_frb(obj=obj)

    print("\nExecuting Python script extinction_galactic.py, with:")
    print(f"\tobj {obj}")

    # TODO: Scriptify

    lambda_eff_interp = p.instrument_filters_single_param(param="lambda_eff", instrument='FORS2', sort_value=True)
    filters_interp = list(lambda_eff_interp.keys())
    lambda_eff_interp = list(lambda_eff_interp.values())

    extinction_path = frb_params['data_dir'] + 'galactic_extinction.txt'
    if not isfile(extinction_path):
        print("Extinction bandpass data not found. Attempting retrieval from IRSA Dust Tool...")
        update_frb_irsa_extinction(frb=obj)

    tbl = Table.read(frb_params['data_dir'] + 'galactic_extinction.txt', format='ascii')
    tbl.sort('LamEff')
    lambda_eff_tbl = tbl['LamEff'] * 1000
    extinctions = tbl['A_SandF']

    extinctions_interp = np.interp(lambda_eff_interp, lambda_eff_tbl, extinctions)

    plt.errorbar(lambda_eff_tbl, extinctions, label='Calculated by IRSA', fmt='o')
    plt.errorbar(lambda_eff_interp, extinctions_interp, label='Numpy Interpolated', fmt='o')
    plt.title('Extinction Interpolation for FRB190102')
    plt.xlabel(r'Filter $\lambda_\mathrm{eff}}$ (nm)')
    plt.ylabel(r'Extinction (magnitude)')
    plt.legend()
    plt.show()

    to_write = {}
    for i, f in enumerate(filters_interp):
        value = float(extinctions_interp[i])
        to_write[f"{f}_ext_gal"] = value
    p.add_output_values_frb(obj=obj, params=to_write)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Retrieve galactic extinction values from the IRSA dust tool and use '
                                                 'to interpolate the value at a given effective wavelength.')
    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)

    args = parser.parse_args()
    main(obj=args.op)
