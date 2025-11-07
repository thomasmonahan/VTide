# utils.py — revised
import sys
import warnings
import pdb

import numpy as np
import pandas as pd
import scipy
import scipy.special
import scipy.linalg

from astropy.time import Time  # for np_datetime64_to_mjd

from utide._solve import _slvinit
from utide.constituent_selection import ut_cnstitsel
from utide._solve import _process_opts
from utide.harmonics import FUV


##########################################
def meanvar(D):
    m = np.mean(D, axis=0)
    s = np.shape(D)
    if len(s) == 1:
        c = D.var()
        if c == 0:
            c = 1.0
    elif (s[0] == 1) + (s[1] == 1):
        c = D.var()
        if c == 0:
            c = 1.0
    else:
        c = np.diag(np.cov(D, rowvar=False)).copy()
        c[c == 0] = 1
    return m, c


##########################################
def normalis(X, D):
    m, c = meanvar(D)
    return (X - m) / np.sqrt(c)


##########################################
def unnorm(X, D):
    m, c = meanvar(D)
    Y = np.multiply(X, np.ones([np.shape(X)[0], 1]) * np.sqrt(c))
    Y = Y + np.dot(np.ones([np.shape(X)[0], 1]), m)
    return Y


##########################################
def logdet(a):
    if not np.allclose(a.T, a):
        print("MATRIX NOT SYMMETRIC")
    # Second make sure that matrix is positive definite:
    eigenvalues = np.linalg.eigvalsh(a)
    if min(eigenvalues) <= 0:
        print("Matrix is NOT positive-definite")
        print("   min eigv = %.16f" % min(eigenvalues))
    step1 = np.linalg.cholesky(a)
    step2 = np.diag(step1.T)
    out = 2.0 * np.sum(np.log(step2), axis=0)
    return out


##########################################
def bayes_linear_fit_ard(X, y):
    # uninformative priors under assumption of N(0,1) incoming data
    # expects X,y to be matrices
    X = np.matrix(X)
    y = np.matrix(y)
    a0 = 1e-2
    b0 = 1e-4
    c0 = 1e-2
    d0 = 1e-4
    # pre-process data
    [N, D] = np.shape(X)
    X_corr = X.T * X
    Xy_corr = X.T * y
    an = a0 + N / 2.0
    gammaln_an = scipy.special.gammaln(an)
    cn = c0 + 1 / 2.0
    D_gammaln_cn = D * scipy.special.gammaln(cn)
    # iterate to find hyperparameters
    L_last = -sys.float_info.max
    max_iter = 500
    E_a = np.matrix(np.ones(D) * c0 / d0).T
    for iter in range(max_iter):
        # covariance and weight of linear model
        invV = np.matrix(np.diag(np.array(E_a)[:, 0])) + X_corr
        V = np.matrix(np.linalg.inv(invV))
        logdetV = -logdet(invV)
        w = np.dot(V, Xy_corr)[:, 0]
        # parameters of noise model (an remains constant)
        sse = np.sum(np.power(X * w - y, 2), axis=0)

        if np.imag(sse) == 0:
            sse = np.real(sse)[0]
        else:
            print("Something went wrong")
            pdb.set_trace()
        bn = b0 + 0.5 * (sse + np.sum((np.array(w)[:, 0] ** 2) * np.array(E_a)[:, 0], axis=0))
        E_t = an / bn
        # hyperparameters of covariance prior (cn remains constant)
        dn = d0 + 0.5 * (E_t * (np.array(w)[:, 0] ** 2) + np.diag(V))
        E_a = np.matrix(cn / dn).T
        # variational bound, ignoring constant terms for now
        L = -0.5 * (E_t * sse + np.sum(np.multiply(X, X * V))) + 0.5 * logdetV - b0 * E_t + gammaln_an - an * np.log(bn) + an + D_gammaln_cn - cn * np.sum(np.log(dn))
        # variational bound must grow!
        if L_last > L:
            # if this happens, then something has gone wrong....
            file = open("ERROR_LOG", "w")
            file.write("Last bound %6.6f, current bound %6.6f" % (L, L_last))
            file.close()
            break
        # stop if change in variation bound is < 0.001%
        if abs(L_last - L) < abs(0.00001 * L):
            break
        # print L, L_last
        L_last = L
    if iter == max_iter:
        warnings.warn("Bayes:maxIter ... Bayesian linear regression reached maximum number of iterations.")
    # augment variational bound with constant terms
    L = L - 0.5 * (N * np.log(2 * np.pi) - D) - scipy.special.gammaln(a0) + a0 * np.log(b0) + D * (-scipy.special.gammaln(c0) + c0 * np.log(d0))
    return w, V, invV, logdetV, an, bn, E_a, L


def vt_E(t, tref, frq, lind, lat, ngflgs, prefilt):
    """
    Compute real quadrature basis functions (cosine and sine), consistent with
    the complex exponential basis E = cos + i*sin.

    Returns
    -------
    C, S : arrays (nt, nc)
        C = F * cos((U+V)*2*pi)
        S = F * sin((U+V)*2*pi)
        where F is the nodal amplitude factor from FUV
    """
    t = np.atleast_1d(t)
    frq = np.atleast_1d(frq)
    lind = np.atleast_1d(lind)
    nt = len(t)
    nc = len(frq)

    # If both nodal and gwch flags indicate trivial factors, simplify
    if ngflgs[1] and ngflgs[3]:
        F = np.ones((nt, nc))
        U = np.zeros((nt, nc))
        V = np.dot(24 * (t - tref)[:, None], frq[:, None].T)
    else:
        F, U, V = FUV(t, tref, lind, lat, ngflgs)

    phase = (U + V) * 2.0 * np.pi
    C = F * np.cos(phase)  # cosine (real part of exp(i*phase))
    S = F * np.sin(phase)  # sine   (imag part of exp(i*phase))

    return C, S


def get_basis_function(times, observations, lat, constity, nodal):
    default_opts = {
        "constit": constity,
        "order_constit": None,
        "conf_int": "none",
        "method": "ols",
        "trend": False,
        "phase": "Greenwich",
        "nodal": False,
        "infer": None,
        "MC_n": 200,
        "Rayleigh_min": 1,
        "robust_kw": {"weight_function": "cauchy"},
        "white": False,
        "verbose": True,
        "epoch": None,
    }
    options = _process_opts(default_opts, is_2D=False)

    tin = times
    uin = observations
    vin = None
    lat = lat
    packed = _slvinit(tin, uin, vin, lat, **options)
    tin, t, u, v, tref, lor, elor, opt = packed
    nt = len(t)
    if opt["cnstit"] == ["NR"]:
        opt["cnstit"] = "auto"
    cnstit, coef = ut_cnstitsel(tref, opt["rmin"] / (24 * lor), opt["cnstit"], opt["infer"])

    coef.aux.opt = opt
    coef.aux.lat = lat

    ngflgs = [opt["nodsatlint"], opt["nodsatnone"], opt["gwchlint"], opt["gwchnone"]]

    E_args = (lat, ngflgs, opt.prefilt)

    # Make the model array, starting with the harmonics.
    C, S = vt_E(t, tref, cnstit.NR.frq, cnstit.NR.lind, *E_args)
    # Return each basis flattened in the same order your regression expects.
    # Many regressors use ordering [cos1, sin1, cos2, sin2, ...]
    return C.flatten(), S.flatten()


def comp_uncert_err(w1, w2, v1, v2, s2):
    """
    Compute amplitude and phase uncertainties given quadrature coefficients.

    Arguments
    ---------
    w1, w2 : floats
        coefficients for cos and sin respectively (i.e. model is w1*cos + w2*sin)
    v1, v2 : floats
        variances of w1 and w2 (NOT stddevs). This matches the diagonal entries of V.
    s2 : float
        aleatoric variance contribution for this constituent (variance, not stddev).

    Returns
    -------
    std_amplitude, std_phase:
        std_amplitude: standard deviation of amplitude R = sqrt(w1^2 + w2^2)
        std_phase: standard deviation of phase (radians)
    """
    # amplitude
    R = np.sqrt(w1 ** 2 + w2 ** 2)

    # total variances for each quadrature component by adding aleatoric term
    tot_v1 = v1 + s2
    tot_v2 = v2 + s2

    # variance of amplitude R using a first-order (delta) approximation
    # var(R) ≈ (w1/R)^2 * var(w1) + (w2/R)^2 * var(w2)
    if R == 0:
        var_R = (tot_v1 + tot_v2) / 2.0
    else:
        var_R = (w1 ** 2 / (R ** 2)) * tot_v1 + (w2 ** 2 / (R ** 2)) * tot_v2

    # phase theta = atan2(w2, w1)
    # variance of theta via delta method:
    # var(theta) ≈ (w2^2/(w1^2 + w2^2)^2) * var(w1) + (w1^2/(w1^2 + w2^2)^2) * var(w2)
    denom = (w1 ** 2 + w2 ** 2) ** 2
    if denom == 0:
        var_theta = (tot_v1 + tot_v2) / (1.0 + 1.0)
    else:
        var_theta = (w2 ** 2 / denom) * tot_v1 + (w1 ** 2 / denom) * tot_v2

    std_amplitude = np.sqrt(var_R)
    std_phase = np.sqrt(var_theta)  # in radians

    return std_amplitude, std_phase


def amps_phases(constituents, w, v, s2):
    """
    Compute amplitudes, phases and their uncertainties.

    Arguments
    ---------
    constituents: list-like of constituents, length nc
    w: 1D array of regression weights ordered [cos1, sin1, cos2, sin2, ...]
    v: 1D array of variances for each weight (diagonal of posterior V), same order as w
    s2: scalar estimate of aleatoric variance (noise variance). We will apportion this
        across constituents proportional to their amplitudes for additional aleatoric weighting.

    Returns
    -------
    amplitudes, phases (degrees), amp_uncerts, phase_uncerts (phase_uncerts returned in degrees)
    """
    amplitudes = []
    phases = []
    amp_uncerts = []
    phase_uncerts = []

    # compute amplitudes and phases
    n_const = len(constituents)
    for i in range(n_const):
        a_idx = 2 * i
        b_idx = 2 * i + 1
        a = w[a_idx]
        b = w[b_idx]
        amp, phase_deg = compute_amplitude_and_angle(a, b)
        amplitudes.append(amp)
        phases.append(phase_deg)

    # compute uncertainties
    total_amp = np.sum(amplitudes) if np.sum(amplitudes) != 0 else 1.0
    for i in range(n_const):
        a_idx = 2 * i
        b_idx = 2 * i + 1
        a = w[a_idx]
        b = w[b_idx]
        # variances (diagonal entries from V) — pass variances, not stddevs
        var_a = v[a_idx]
        var_b = v[b_idx]
        # apportion aleatoric variance to this constituent proportional to its amplitude
        alloc_s2 = s2 * (amplitudes[i] / total_amp)
        amp_uncert, phase_uncert_rad = comp_uncert_err(a, b, var_a, var_b, alloc_s2)
        amp_uncerts.append(amp_uncert)
        phase_uncerts.append(np.degrees(phase_uncert_rad))  # convert to degrees

    return amplitudes, phases, amp_uncerts, phase_uncerts


def comp_outliers(sd, yp, actual, thresh):
    combo1 = yp.flatten()
    actual = actual.flatten()
    above = np.where(actual > yp.flatten() + thresh * sd.flatten())[0]
    below = np.where(actual < yp.flatten() - thresh * sd.flatten())[0]
    return np.array(list(above) + list(below))


def compute_amplitude_and_angle(a, b):
    """
    For model: a*cos(ωt) + b*sin(ωt)
    amplitude R = sqrt(a^2 + b^2)
    phase (degrees) phi = atan2(b, a)  (returns degrees)
    """
    R = np.sqrt(a ** 2 + b ** 2)
    theta_deg = np.degrees(np.arctan2(b, a))
    return R, theta_deg


def np_datetime64_to_mjd(dates):
    """
    Convert a list/array of np.datetime64 objects to Modified Julian Dates (MJD).
    """
    datetimes = [pd.to_datetime(date).to_pydatetime() for date in dates]
    mjd_list = []
    for date in datetimes:
        t = Time(date, format="datetime")
        mjd_list.append(t.mjd)
    return np.array(mjd_list)
