import numpy as np
import pytest
import scipy
import pandas as pd

# Import the functions from your utils module.
# Adjust the module path as needed.
from vtide.utils import (
    meanvar, normalis, unnorm, logdet, bayes_linear_fit_ard,
    vt_E, get_basis_function, comp_uncert_err, amps_phases,
    comp_outliers, compute_amplitude_and_angle
)


###########################
# Test for bayes_linear_fit_ard
###########################
def test_bayes_linear_fit_ard():
    # Create synthetic linear data.
    np.random.seed(0)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2 * X + np.random.normal(0, 0.5, size=X.shape)
    w, V, invV, logdetv, an, bn, E_a, L = bayes_linear_fit_ard(X, y)
    # Check dimensions and that the variational bound L is finite.
    assert w.shape[0] == X.shape[1]
    assert np.isfinite(L)

###########################
# Test for vt_E
###########################
def test_vt_E():
    t = np.array([0, 1, 2])
    tref = 0
    frq = np.array([0.1, 0.2])
    lind = np.array([0, 1])
    lat = 45
    # Set flags so that condition (ngflgs[1] and ngflgs[3]) is True.
    ngflgs = [False, True, False, True]
    prefilt = None
    A, B = vt_E(t, tref, frq, lind, lat, ngflgs, prefilt)
    assert A.shape == (len(t), len(frq))
    assert B.shape == (len(t), len(frq))

###########################
# Test for get_basis_function
###########################
def test_get_basis_function():
    # Use synthetic data. (This test assumes that utide functions work in your environment.)
    times = np.linspace(0, 1, 10)
    observations = np.sin(2 * np.pi * times)
    lat = 45
    constity = ['M2']  # Use a valid constituent name.
    nodal = False
    A, B = get_basis_function(times, observations, lat, constity, nodal)
    # Expect A and B to be 1D arrays of length equal to times.
    assert A.ndim == 1
    assert B.ndim == 1
    assert len(A) == len(times)
    assert len(B) == len(times)

###########################
# Test for comp_uncert_err
###########################
def test_comp_uncert_err():
    w1, w2 = 1.0, 1.0
    v1, v2 = 0.1, 0.1
    s2 = 1.0
    std_amp, std_phase = comp_uncert_err(w1, w2, v1, v2, s2)
    assert std_amp > 0
    assert std_phase > 0

###########################
# Test for amps_phases
###########################
def test_amps_phases():
    # For two constituents.
    constituents = ['M2', 'S2']
    # Construct a weight vector for two constituents: for first, w1=3, w2=4; for second, w1=6, w2=8.
    w = np.array([3, 4, 6, 8])
    # Provide corresponding variances.
    v = np.array([0.01, 0.01, 0.02, 0.02])
    s2 = 1.0
    amplitudes, phases, amp_uncerts, phase_uncerts = amps_phases(constituents, w, v, s2)
    # Check that outputs match the number of constituents.
    assert len(amplitudes) == 2
    # For the first constituent, amplitude should be ~5 and phase ~arctan2(4,3)
    np.testing.assert_almost_equal(amplitudes[0], 5, decimal=1)
    expected_phase = np.degrees(np.arctan2(4, 3))
    np.testing.assert_almost_equal(phases[0], expected_phase, decimal=1)

###########################
# Test for comp_outliers
###########################
def test_comp_outliers():
    sd = np.ones(5)
    yp = np.zeros(5)
    actual = np.array([0, 0, 5, 0, 0])
    thresh = 3
    out = comp_outliers(sd, yp, actual, thresh)
    np.testing.assert_array_equal(out, np.array([2]))

###########################
# Test for compute_amplitude_and_angle
###########################
def test_compute_amplitude_and_angle():
    A_val = 3
    B_val = 4
    amp, angle = compute_amplitude_and_angle(A_val, B_val)
    np.testing.assert_almost_equal(amp, 5)
    expected_angle = np.degrees(np.arctan2(4, 3))
    np.testing.assert_almost_equal(angle, expected_angle, decimal=1)
