from astropy.coordinates import SkyCoord
import astropy.units as units

import craftutils.astrometry as ast

position = SkyCoord("21h44m25.255s -40d54m00.10s")
tolerance = 1e-8 * units.deg


def test_attempt_skycoord():
    assert position == ast.attempt_skycoord(position.copy())
    assert position == ast.attempt_skycoord("21h44m25.255s -40d54m00.10s")
    assert position.separation(ast.attempt_skycoord((326.10522917, -40.90002778))) < tolerance
    assert position == ast.attempt_skycoord(("21h44m25.255s", "-40d54m00.10s"))
    assert position.separation(ast.attempt_skycoord(("326.10522917", "-40.90002778"))) < tolerance
    assert position.separation(ast.attempt_skycoord((326.10522917, "-40d54m00.10s"))) < tolerance
