"""Normalization contract: coordinate/θ round-trips and scale invariance."""

import numpy as np

from tdp.model.normalization import (
    BOARD_ASPECT,
    FrameNorm,
    cond_vector,
    dT_scale_from_sensors,
    px_to_xy,
    temp_from_theta,
    theta_from_temp,
    xy_to_px,
)


def test_px_xy_roundtrip():
    rows = np.arange(0, 157, 13)
    cols = np.arange(0, 103, 11)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    xy = px_to_xy(rr, cc)
    assert xy[..., 0].max() < BOARD_ASPECT
    assert xy[..., 1].max() < 1.0
    r2, c2 = xy_to_px(xy)
    np.testing.assert_allclose(r2, rr, atol=1e-9)
    np.testing.assert_allclose(c2, cc, atol=1e-9)


def test_theta_roundtrip_and_floor():
    norm = FrameNorm(t_amb=26.5, dT_scale=dT_scale_from_sensors(np.array([27.0]), 26.5),
                     aspect=BOARD_ASPECT)
    assert norm.dT_scale == 1.0  # floored: idle sensors barely above ambient
    T = np.array([26.5, 30.0, 79.2])
    np.testing.assert_allclose(temp_from_theta(theta_from_temp(T, norm), norm), T)


def test_amplitude_scale_invariance():
    """Doubling the field amplitude leaves the normalized inputs unchanged."""
    t_amb = 26.0
    field = np.array([30.0, 40.0, 60.0])
    for gain in (1.0, 2.0, 5.0):
        T = t_amb + (field - t_amb) * gain
        norm = FrameNorm(t_amb, dT_scale_from_sensors(T, t_amb), BOARD_ASPECT)
        theta = theta_from_temp(T, norm)
        np.testing.assert_allclose(
            theta, (field - t_amb) / (field.max() - t_amb), atol=1e-12)


def test_cond_vector_shapes():
    c = cond_vector(2.5, 1.3, BOARD_ASPECT, True)
    assert c.shape == (4,) and np.isfinite(c).all() and c[3] == 1.0
    c = cond_vector(2.5, None, BOARD_ASPECT, False)
    assert c[1] == 0.0 and c[3] == 0.0
