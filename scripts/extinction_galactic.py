# Code by Lachlan Marnoch, 2019
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from astropy.table import Table

matplotlib.rcParams.update({'errorbar.capsize': 3})

# TODO: Write something that gets these numbers automatically
# TODO: Scriptify

# Effective wavelengths
lambda_u = 0.361  # u_HIGH
lambda_b = 0.440  # b_HIGH
lambda_g = 0.470  # g_HIGH
lambda_v = 0.557  # v_HIGH
lambda_R = 0.655  # R_SPECIAL
lambda_I = 0.768  # I_BESS
lambda_z = 0.910  # z_GUNN

filters_interp = ['u', 'b', 'g', 'v', 'R', 'I', 'z']
lambda_eff_int = np.array([lambda_u, lambda_b, lambda_g, lambda_v, lambda_R, lambda_I, lambda_z])

# FRB190102
print('FRB190102')

ext_190102 = Table.read('/home/lachlan/Data/FRB190102/galactic_extinction.txt', format='ascii')

ext_190102.sort('LamEff')
lambda_eff_tbl = ext_190102['LamEff']
extinctions_tbl = ext_190102['A_SandF']

extinctions_interp = np.interp(lambda_eff_int, lambda_eff_tbl, extinctions_tbl)

print(lambda_eff_tbl, extinctions_tbl)

print(filters_interp)
print(lambda_eff_int)
print(extinctions_interp)

plt.errorbar(lambda_eff_tbl, extinctions_tbl, label='Calculated by IRSA', fmt='o')
plt.errorbar(lambda_eff_int, extinctions_interp, label='Numpy Interpolated', fmt='o')
plt.title('Extinction Interpolation for FRB190102')
plt.xlabel(r'Filter $\lambda_\mathrm{eff}}$ (nm)')
plt.ylabel(r'Extinction (magnitude)')
plt.legend()
plt.show()

# FRB180924
print('FRB180924')

ext_180924 = Table.read('/home/lachlan/Data/FRB180924/galactic_extinction.txt', format='ascii')

ext_180924.sort('LamEff')
lambda_eff_tbl = ext_180924['LamEff']
extinctions_tbl = ext_180924['A_SandF']

extinctions_interp = np.interp(lambda_eff_int, lambda_eff_tbl, extinctions_tbl)

print(lambda_eff_tbl, extinctions_tbl)

print(filters_interp)
print(lambda_eff_int)
print(extinctions_interp)

plt.errorbar(lambda_eff_tbl, extinctions_tbl, label='Calculated by IRSA', fmt='o')
plt.errorbar(lambda_eff_int, extinctions_interp, label='Numpy Interpolated', fmt='o')
plt.title('Extinction Interpolation for FRB180924')
plt.xlabel(r'Filter $\lambda_\mathrm{eff}}$ (nm)')
plt.ylabel(r'Extinction (magnitude)')
plt.legend()
plt.show()

# FRB181112
print('FRB181112')

ext_181112 = Table.read('/home/lachlan/Data/FRB181112/galactic_extinction.txt', format='ascii')

ext_181112.sort('LamEff')
lambda_eff_tbl = ext_181112['LamEff']
extinctions_tbl = ext_181112['A_SandF']

extinctions_interp = np.interp(lambda_eff_int, lambda_eff_tbl, extinctions_tbl)

print(lambda_eff_tbl, extinctions_tbl)

print(filters_interp)
print(lambda_eff_int)
print(extinctions_interp)

plt.errorbar(lambda_eff_tbl, extinctions_tbl, label='Calculated by IRSA', fmt='o')
plt.errorbar(lambda_eff_int, extinctions_interp, label='Numpy Interpolated', fmt='o')
plt.title('Extinction Interpolation for FRB181112')
plt.xlabel(r'Filter $\lambda_\mathrm{eff}}$ (nm)')
plt.ylabel(r'Extinction (magnitude)')
plt.legend()
plt.show()


# FRB190608
print('FRB190608')

ext_190608 = Table.read('/home/lachlan/Data/FRB190608/galactic_extinction.txt', format='ascii')

ext_190608.sort('LamEff')
lambda_eff_tbl = ext_190608['LamEff']
extinctions_tbl = ext_190608['A_SandF']

extinctions_interp = np.interp(lambda_eff_int, lambda_eff_tbl, extinctions_tbl)

print(lambda_eff_tbl, extinctions_tbl)

print(filters_interp)
print(lambda_eff_int)
print(extinctions_interp)

plt.errorbar(lambda_eff_tbl, extinctions_tbl, label='Calculated by IRSA', fmt='o')
plt.errorbar(lambda_eff_int, extinctions_interp, label='Numpy Interpolated', fmt='o')
plt.title('Extinction Interpolation for FRB190608')
plt.xlabel(r'Filter $\lambda_\mathrm{eff}}$ (nm)')
plt.ylabel(r'Extinction (magnitude)')
plt.legend()
plt.show()

# FRB191001
print('FRB191001')

ext_191001 = Table.read('/home/lachlan/Data/FRB191001/galactic_extinction.txt', format='ascii')

ext_191001.sort('LamEff')
lambda_eff_tbl = ext_191001['LamEff']
extinctions_tbl = ext_191001['A_SandF']

extinctions_interp = np.interp(lambda_eff_int, lambda_eff_tbl, extinctions_tbl)

print(lambda_eff_tbl, extinctions_tbl)

print(filters_interp)
print(lambda_eff_int)
print(extinctions_interp)

plt.errorbar(lambda_eff_tbl, extinctions_tbl, label='Calculated by IRSA', fmt='o')
plt.errorbar(lambda_eff_int, extinctions_interp, label='Numpy Interpolated', fmt='o')
plt.title('Extinction Interpolation for FRB191001')
plt.xlabel(r'Filter $\lambda_\mathrm{eff}}$ (nm)')
plt.ylabel(r'Extinction (magnitude)')
plt.legend()
plt.show()


