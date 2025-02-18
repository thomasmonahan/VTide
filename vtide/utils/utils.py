import scipy
import scipy.special
import scipy.linalg
import numpy
import pdb
import sys
import warnings
import pandas as pd
import numpy as np

from utide._solve import _slvinit
from utide.constituent_selection import ut_cnstitsel
from utide._solve import _process_opts
from utide.harmonics import FUV


##########################################
def meanvar(D):
    m = np.mean(D, axis = 0)
    s = np.shape(D)
    if len(s) == 1:
        c = D.var()
        if c == 0: c = 1.0
    elif (s[0] == 1) + (s[1] == 1):
        c = D.var()
        if c == 0: c = 1.0
    else:
        c = np.diag(np.cov(D, rowvar = False)).copy()
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
    if np.allclose(a.T,a) == False:
        print('MATRIX NOT SYMMETRIC')
    # Second make sure that matrix is positive definite:
    eigenvalues = np.linalg.eigvalsh(a)
    if min(eigenvalues) <=0:
        print('Matrix is NOT positive-definite')
        print('   min eigv = %.16f' % min(eigenvalues))
    step1 = np.linalg.cholesky(a)
    step2 = np.diag(step1.T)
    out = 2. * np.sum(np.log(step2), axis=0)
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
    #X_corr = np.dot(X.T , X)
    Xy_corr = X.T * y
    #Xy_corr = np.dot(X.T , y)
    an = a0 + N / 2.
    gammaln_an = scipy.special.gammaln(an)
    cn = c0 + 1 / 2.
    D_gammaln_cn = D * scipy.special.gammaln(cn)
    # iterate to find hyperparameters
    L_last = -sys.float_info.max
    max_iter = 500
    E_a = np.matrix(np.ones(D) * c0 / d0).T
    for iter in range(max_iter):
        # covariance and weight of linear model
        invV = np.matrix(np.diag(np.array(E_a)[:,0])) + X_corr
        V = np.matrix(np.linalg.inv(invV))
        logdetV = -logdet(invV)
        w = np.dot(V, Xy_corr)[:,0]
        # parameters of noise model (an remains constant)
        sse = np.sum(np.power(X*w-y, 2), axis=0)

        if np.imag(sse)==0:
            sse = np.real(sse)[0]
        else:
            print('Something went wrong')
            pdb.set_trace()
        bn = b0 + 0.5 * (sse + np.sum((np.array(w)[:,0]**2) * np.array(E_a)[:,0], axis=0))
        E_t = an / bn
        # hyperparameters of covariance prior (cn remains constant)
        dn = d0 + 0.5 * (E_t * (np.array(w)[:,0]**2) + np.diag(V))
        E_a = np.matrix(cn / dn).T
        # variational bound, ignoring constant terms for now
        L = -0.5 * (E_t*sse + np.sum(np.multiply(X,X*V))) + \
            0.5 * logdetV - b0 * E_t + gammaln_an - an * np.log(bn) + an + \
            D_gammaln_cn - cn * np.sum(np.log(dn))
        # variational bound must grow!
        if L_last > L:
            # if this happens, then something has gone wrong....
            file = open('ERROR_LOG','w')
            file.write('Last bound %6.6f, current bound %6.6f' % (L, L_last))
            file.close()
            #raise Exception('Variational bound should not reduce - see ERROR_LOG')
            #return
            break
        # stop if change in variation bound is < 0.001%
        if abs(L_last - L) < abs(0.00001 * L):
            break
        # print L, L_last
        L_last = L
    if iter == max_iter:
        warnings.warn('Bayes:maxIter ... Bayesian linear regression reached maximum number of iterations.')
    # augment variational bound with constant terms
    L = L - 0.5 * (N * np.log(2 * np.pi) - D) - scipy.special.gammaln(a0) + \
        a0 * np.log(b0) + D * (-scipy.special.gammaln(c0) + c0 * np.log(d0))
    return w, V, invV, logdetV, an, bn, E_a, L



def vt_E(t, tref, frq, lind, lat, ngflgs, prefilt):
    """
    Compute complex exponential basis function. (Adapted from UTide)

    Parameters
    ----------
    t : array_like or float (nt,)
        time in days
    tref : float
        reference time in days
    frq : array_like or float (nc,)
        frequencies in cph
    lind : array_like or int (nc,)
        indices of constituents
    lat : float
        latitude, degrees N
    nflgs : array_like, bool
        [NodsatLint NodsatNone GwchLint GwchNone]
    prefilt: Bunch
        not implemented

    Returns
    -------
    E : array (nt, nc)
        complex exponential basis function; always returned as 2-D array
    """

    t = np.atleast_1d(t)
    frq = np.atleast_1d(frq)
    lind = np.atleast_1d(lind)
    nt = len(t)
    nc = len(frq)

    if ngflgs[1] and ngflgs[3]:
        F = np.ones((nt, nc))
        U = np.zeros((nt, nc))
        V = np.dot(24 * (t - tref)[:, None], frq[:, None].T)
    else:
        F, U, V = FUV(t, tref, lind, lat, ngflgs)

    #E = F * np.exp(1j * (U + V) * 2 * np.pi)
    A = np.sin((U + V)*2*np.pi)
    B = np.cos((U + V)*2*np.pi)

    return A,B

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
  options = _process_opts(default_opts, is_2D = False)

  tin = times
  uin = observations
  vin = None
  lat = lat
  packed = _slvinit(tin, uin, vin, lat, **options)
  tin, t, u, v, tref, lor, elor, opt = packed
  nt = len(t)
  if opt['cnstit'] == ['NR']:
    opt['cnstit'] = 'auto'
  cnstit, coef = ut_cnstitsel(
      tref,
      opt["rmin"] / (24 * lor),
      opt["cnstit"],
      opt["infer"],
  )

  coef.aux.opt = opt
  coef.aux.lat = lat


  ngflgs = [opt["nodsatlint"], opt["nodsatnone"], opt["gwchlint"], opt["gwchnone"]]

  E_args = (lat, ngflgs, opt.prefilt)

  # Make the model array, starting with the harmonics.
  A,B = vt_E(t, tref, cnstit.NR.frq, cnstit.NR.lind, *E_args)
  return A.flatten(), B.flatten()

def comp_uncert_err(w1,w2, v1,v2, s2):    
    ## w,v,s2 come straight from VB_linreg_ard    
    ## idx is the index of the constituent you want to use: e.g. if I have [M2,N2,S2] and want S2 uncertainty idx = 2    
    ## GETTING NORMALIZED AMPLITUDES TO WEIGHT ALEATORIC UNCERTAINTY    
    A = np.sqrt(w1**2 + w2**2)    
    tot_v1 = v1  + s2 ## weighting alteatoric uncertainty by the constituent magnitude    
    tot_v2 = v2  + s2    
    var_A = (w1**2 / (w2**2 + w1**2)) * tot_v1 + (w2**2 / (w1**2 + w2**2)) * tot_v2    # Compute phase (theta) and its variance    
    theta = np.arctan2(w2, w1)    
    var_theta = (w2**2 / (w1**2 + w2**2)**2) * tot_v1 + (w1**2 / (w1**2 + w2**2)**2) * tot_v2    # Uncertainties     ## THESE CAN BE USED DIRECTLY FOR AMPLITUDE/PHASE UNCERTAINTIES    
    std_amplitude = np.sqrt(var_A)     
    std_phase = np.sqrt(var_theta)    
    return std_amplitude, std_phase

def amps_phases(constituents, w, v,s2):
  amplitudes = []
  phases = []
  amp_uncerts = []
  phase_uncerts = []
  ## Computing Amplitudes and Phases
  for i in range(len(constituents)):
    amp, phase  = compute_amplitude_and_angle(w[2*i], w[2*i +1])
    amplitudes.append(amp)
    phases.append(phase)
  for i in range(len(constituents)):
    amp_uncert, phase_uncert =  comp_uncert_err(w[2*i], w[2*i +1], np.sqrt(v[2*i]), np.sqrt(v[2*i +1]), s2*(amplitudes[i] / np.sum(amplitudes)))
    amp_uncerts.append(amp_uncert)
    phase_uncerts.append(phase_uncert)
  return amplitudes, phases, amp_uncerts, phase_uncerts

def comp_outliers(sd, yp, actual, thresh):
  combo1 = yp.flatten()
  actual = actual.flatten()
  above = np.where(actual > yp.flatten() + thresh*sd.flatten())[0]
  below = np.where(actual < yp.flatten() - thresh*sd.flatten())[0]

  return np.array(list(above) + list(below))

def compute_amplitude_and_angle(A, B):
    # Compute the amplitude
    R = np.sqrt(A**2 + B**2)

    # Compute the angle in radians
    theta = np.arctan2(B, A) * 180 / np.pi

    return R, theta
    
def np_datetime64_to_mjd(dates):
    """
    Convert a list of np.datetime64 objects to Modified Julian Dates (MJD).

    Parameters:
    dates (list): A list of np.datetime64 objects.

    Returns:
    list: A list of corresponding Modified Julian Dates (MJD).
    """
    # Convert np.datetime64 objects to datetime.datetime using pandas
    datetimes = [pd.to_datetime(date).to_pydatetime() for date in dates]

    # Convert datetime.datetime objects to MJD using astropy
    mjd_list = []
    for date in datetimes:
        t = Time(date, format='datetime')
        mjd = t.mjd
        mjd_list.append(mjd)

    return np.array(mjd_list)
