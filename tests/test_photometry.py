import craftutils.photometry as ph
import astropy.units as units
import numpy as np

# Number of digits to round to before checking values
tolerance = 15


def test_magnitude_uncertainty():
    flux = [4051519.0,
            3501612.0]
    flux_err = [28118.24,
                70699.83]
    exp_time = [903.0,
                610.0]
    exp_time_err = [0.0,
                    0.0]

    mag_expected = [-9.129825324003573,
                    -9.397345467302534]
    err_expected = [0.007535196350326441,
                    0.02192172208149988]

    for i in range(len(flux)):
        mag, err = ph.magnitude_uncertainty(flux=flux[i] * units.ct,
                                            flux_err=flux_err[i] * units.ct,
                                            exp_time=exp_time[i] * units.second,
                                            exp_time_err=exp_time_err[i] * units.second)
        assert mag.round(tolerance) == (mag_expected[i] * units.mag).round(tolerance)
        assert err.round(tolerance) == (err_expected[i] * units.mag).round(tolerance)
