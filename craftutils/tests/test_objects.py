from craftutils.observation import objects
import astropy.units as units
from astropy.coordinates import SkyCoord
import numpy as np

# Number of digits to round to before checking values
tolerance = 0

test_frb = objects.FRB(
    dm=1234.5 * objects.dm_units,
    position=SkyCoord("23h23m10.424s -30d24m19.55s")
)


# def test_frb_dm_mw_halo():
#     dm_mws = test_frb.dm_mw_halo(
#         model="all",
#         zero_distance=10*units.kpc
#     )
#     assert dm_mws["dm_halo_mw_yf17"].round(tolerance) == 68 * objects.dm_units
#     assert dm_mws["dm_halo_mw_pz19"].round(tolerance) == 39 * objects.dm_units
#     assert dm_mws["dm_halo_mw_mb15"].round(tolerance) == 9 * objects.dm_units


# def test_frb_dm_mw_ism():
#     assert test_frb.dm_mw_ism_ne2001().round(tolerance) == 32 * objects.dm_units
#     assert test_frb.dm_mw_ism_ymw16().round(tolerance) == 17 * objects.dm_units