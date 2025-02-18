from .utils import *
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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



class VTide:
    """
    VTide class for variational Bayesian tidal estimation
    
    https://doi.org/10.22541/essoar.172072179.92243659/v1
    
    Parameters
    ----------
    ts : DataFrame
      time series data containing the observations and optionally other input 
      functions. Must have datetime index, though data can be unevenly spaced.
    lat: float
      latitude of the station
    lon: float
      longitude of the station 
    
    Examples
    ----------
    >>> from vtide import VTide
    >>> import pandas as pd
    >>> import numpy as np
    >>> constit_freqs = [8.05114007e-02, 7.89992487e-02, 8.33333333e-02, 8.20235526e-02]
    >>> mplitudes = [2.69536915e+00, 5.72051193e-01, 4.07506583e-01, 1.77388101e-01]
    >>> phases = [100.17759146,  67.89243773, 139.00250991, 137.86574464]
    >>> times = pd.date_range(start = '2024-01-01 00:00:00+00:00', periods=24*365, freq="1H",tz='UTC')
    >>> time_vals = times.to_julian_date().to_numpy()
    >>> individual_tides = []
    >>> for i,j in enumerate(constit_freqs):
    >>>    individual_tides.append(amplitudes[i]*np.cos(24*j*np.array(time_vals) * 2 * np.pi + (phases[i] *np.pi /180)))
    >>> tide = np.sum(individual_tides, axis = 0) + 0.1*np.random.normal(len(individual_tides))
    >>> df = pd.DataFrame({'observations':tide}, index = times)
    >>> model = VTide(df, lat = 44.9062, lon = -66.996201)
    >>> model.Solve()
    >>> model.Predict(test_times)
    >>> harmonics = model.harmonics
    """
    def __init__(self, ts, lat, lon):
      self.ts = ts # time series

      self.times = ts.index.to_numpy()
      self.obs = ts.observations.to_numpy()
      self.lat = lat # latitude
      self.lon = lon # longitude

      self._validate_inputs()
  
    def _validate_inputs(self):
      if not isinstance(self.ts.index, pd.DatetimeIndex):
          raise ValueError("The DataFrame does not have a DatetimeIndex.")
      if not isinstance(self.lat, (int, float)):
          raise ValueError("Latitude must be a number.")
      if not isinstance(self.lon, (int, float)):
          raise ValueError("Longitude must be a number.")
    
    def Compute_Basis_Function(self, times, obs, const):
        A,B = get_basis_function(times, obs, self.lat, [const], nodal = self.nodal)
        return A,B
    
    def Prepare_Matrix(self, times, obs):
      xs = []
      for const in self.constituents:
          A,B = self.Compute_Basis_Function(times, obs, const)
          xs.append(A)
          xs.append(B)
      if self.trend:
          xs.append(minmax_scaling(self.times.to_julian_date().to_numpy()))

      x = np.array(xs).T
      x = np.matrix(x)
      return x
    def VB_linreg_ard(self, xtr, ttr, xte, BIAS, NORM):
        # expects xtr, ttr and xte to be matrix class. Returns yp,sd & factor weights, w as arrays
        # BIAS, NORM == 1 or 0
    
        xtr = np.matrix(xtr)
        ttr = np.matrix(ttr)
        xte = np.matrix(xte)
    
        ntr = np.shape(xtr)[0]
        nnew = np.shape(xte)[0]
    
        if (NORM==1):
            # normalise everything in the training data to mn 0, var 1
            xn = normalis(xtr, xtr)
            t = normalis(ttr, ttr)
        else:
            xn = xtr
            t = ttr
    
        if (BIAS==1):
            xn = np.concatenate([xn, np.ones([ntr,1])], axis=1) # add in the bias term
    
        # run the Bayes linear model with shrinkage
        [w, v, invv, logdetv, an, bn, e_a, l] = bayes_linear_fit_ard(xn, t)
        self.fit_params = [w, v, invv, logdetv, an, bn, e_a, l]
        if (NORM==1):
            # normalis test data w.r.t. training set
            xtest = normalis(xte, xtr)
        else:
            xtest = xte
    
        if (BIAS==1):
            xtest = np.concatenate([xtest, np.ones([nnew,1])], axis=1) #add in the bias term
    
        # infer the predictive dist for the test data then un-normalise it
        yp = xtest * w # expectation of y
        e_tau = an / bn # expectation of noise precision
        s2 = 1. / e_tau
        wu = np.matrix(np.zeros([np.shape(xtest)[0],1]))
        for n in range(np.shape(xtest)[0]):    # model uncertainty [weight uncert]
            wu[n, :] = xtest[n,:] * v * xtest[n,:].T
        sd = np.sqrt(s2 + wu)
        if (NORM==1):
            yp = unnorm(yp, ttr) # undo the normalisation step
            sd = np.multiply(sd, np.matrix(np.tile(np.std(ttr,axis=0),[np.shape(yp)[0],1])))
    
        return np.array(yp), np.array(sd), np.array(w), np.diag(v), s2
        
    def Compute_VB(self, x, obs, times, linear_trend, prune = False):
      ## Fitting VBayes
      yp,sd,w,v,s2 = self.VB_linreg_ard(x,np.array(obs).reshape(len(times),1),x, BIAS = 1,NORM=0)
      x_original = x
      bad_timestamps = []
      if prune:
        ## Remove erroneous value until either (i) no values remain, or (ii) 5 loops complete
        loops = 0
        outlier = comp_outliers(sd, yp, np.array(obs), prune)
        while (len(outlier) > 0) and (loops < 5):
          bad_timestamps.extend(times[outlier]) ## getting list of bad data points
          obs = np.delete(obs, outlier)
          times = np.delete(times, outlier)
          
          x = self.Prepare_Matrix(times, obs) 
          yp,sd,w,v,s2 = self.VB_linreg_ard(x,np.array(obs).reshape(len(times),1), x,BIAS=1,NORM=0)
          outlier = comp_outliers(sd, yp, np.array(obs),prune)
          loops += 1
        
        ## Recomputing predictions for original time-series
        yp,sd,w,v,s2 = self.VB_linreg_ard(x,np.array(obs).reshape(len(times),1), x_original, BIAS=1,NORM=0)
        bad_timestamps = sorted(set(bad_timestamps))
        self.bad_indices = [np.where(self.times == ts)[0][0] for ts in sorted(set(bad_timestamps))]
      else:
        self.bad_indices = []
      return yp,sd,w,v,s2,bad_timestamps
        
    def Plot_Constituents(self):
        consts = list(self.harmonics.keys())
        amps = np.array([self.harmonics[const]['amp'] for const in consts])
        amp_uncerts = np.array([self.harmonics[const]['amp_uncert'] for const in consts])
    
        # Sort based on amplitudes in descending order
        sorted_indices = np.argsort(amps)[::-1]  
        sorted_consts = np.array(consts)[sorted_indices]  # Convert to NumPy array for indexing
        sorted_amps = amps[sorted_indices]
        sorted_amp_uncerts = amp_uncerts[sorted_indices]
    
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(sorted_consts, sorted_amps, yerr=sorted_amp_uncerts, capsize=5, color='steelblue', alpha=0.75)
    
        # Labels and title
        ax.set_xlabel("Constituents", fontsize = 20)
        ax.set_ylabel("Amplitude", fontsize = 20)
        ax.set_xticks(range(len(sorted_consts)))  
        ax.set_xticklabels(sorted_consts, rotation=90, ha="center")
    
        # Grid and formatting
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
    
        # Show plot
        plt.show()

    def Visualize_Predictions(self):

        yp = self.train_predictions['tide']
        observations = self.obs
        sd = self.train_predictions['sd']
        times = self.times
        residuals = yp - observations  # Compute residuals
        if len(self.bad_indices) > 0:
            residuals_minus_outliers = np.delete(residuals, self.bad_indices)
        else:
            residuals_minus_outliers = residuals
        
        
        # Create figure with gridspec
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])  # 2 columns, 2 rows
        
        # Time Series Plot (Spanning 2 rows)
        ax1 = fig.add_subplot(gs[:, 0])  # Span both rows
        ax1.plot(times, observations, label="Observed", color="k", alpha=0.7)
        ax1.plot(times, yp, label="Predicted", color="C0")
        
        # Confidence Interval (±2*sd)
        ax1.fill_between(times, yp - 2*sd, yp + 2*sd, color="C0", alpha=0.3, label="2$\sigma$")
        
        # Format X-Axis
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())  # Auto tick placement
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # Format as YYYY-MM-DD
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")  # Rotate labels
        if len(self.bad_timestamps) >0:
            for i,ti in enumerate(self.bad_timestamps):
                if i == 0:
                    ax1.axvline(ti, color = 'red', label = 'Outlier(s)')
                else:
                    ax1.axvline(ti, color = 'red')
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Sea Level")
        ax1.legend()
        ax1.set_xlim(times[0], times[-1])
        
        # KDE Plot 1 (Residuals)
        ax2 = fig.add_subplot(gs[0, 1])
        sns.kdeplot(x=residuals, ax=ax2, fill=True, color="C0", alpha=0.3)
        ax2.set_ylabel("Density")
        ax2.set_xlabel("Residuals")
        
        # KDE Plot 2 (Standardized Residuals)
        ax3 = fig.add_subplot(gs[1, 1])
        sns.kdeplot(x=residuals_minus_outliers, ax=ax3, fill=True, color="C1", alpha=0.3)
        ax3.set_ylabel("Density")
        ax3.set_xlabel("Residuals (without outliars)")
        
        plt.tight_layout()
        plt.show()
        
    def Solve(self, constituents = 'auto', nodal = False, trend = False, prune = False, aleatoric_uncert = True):
        """
        Function to estimate tidal constituents using variational Bayesian HA. 
        
        Parameters
        ----------
        constituents : List or None
          constituents to include in the analysis. If None, computed from rayleigh criterion with R=1
        nodal: Boolean
          Whether or not to include nodal corrections
        trend: Boolean
          Whether or not to compute linear sea-level trend
        prune: False or float
          If not false, the provided float determines the threshold for values to remove in standard deviations. 
          3$\sigma$ = 99.7, 4$\sigma$ = 99.99, etc. 

        returns 
        ---------
        self.harmonics : dictionary
            dictionary containg harmonic constituents amplitudes (amps), phases (phase), and associated uncertainties
            (amp_uncerts) and (phase_uncerts)
        yp : np.array
            tidal predictions
        sd : np.array
            standard deviation of predictions
        """

        if not (prune == False or isinstance(prune, (int, float))):
          raise ValueError("prune must either be False, or a float representing the number of standard deviations to use as a threshold")
        if not isinstance(nodal,bool):
          raise ValueError("nodal must be either True or False")
        if not (constituents == 'auto' or isinstance(constituents, list)):
          raise ValueError("if not 'auto', constituents must be a list containing strings of different constituents")
        if not isinstance(trend,bool):
          raise ValueError("trend must be either True or False")
        if not isinstance(aleatoric_uncert,bool):
          raise ValueError("aleatoric_uncert must be either True or False")

        self.nodal = nodal
        self.trend = trend

        if constituents == 'auto':
            ### Using UTide constituent selection
            tref = 0.5 * (self.ts.index.to_julian_date().to_numpy()[0] + self.ts.index.to_julian_date().to_numpy()[-1])
            rayleigh_constits, coef = ut_cnstitsel(
                 tref,
                1 / (24 * np.ptp(self.ts.index.to_julian_date().to_numpy())),
                'auto',
                None)
            self.constituents = rayleigh_constits.NR.name
        else:
            self.constituents = constituents

        x = self.Prepare_Matrix(self.times, self.obs)

        ## Computing VB estimate with optional pruning
        
        yp, sd, self.w, self.v, self.s2, self.bad_timestamps = self.Compute_VB(x, self.obs, self.times, self.trend, prune)
        
        if aleatoric_uncert:
            s2 = self.s2
        else:
            s2 = np.matrix(np.array([0]))
        ## Computing Amps/Phases
        amplitudes, phases, amp_uncerts, phase_uncerts = amps_phases(self.constituents, self.w, self.v, s2)

        self.harmonics = {}
        for i,const in enumerate(self.constituents):
            self.harmonics[const] = {
                'amp': amplitudes[i][0],
                'phase': phases[i][0],
                'amp_uncert': np.array(amp_uncerts[i])[0][0],
                'phase_uncert': np.array(phase_uncerts[i])[0][0]
            }
        self.train_predictions = {'tide': yp.flatten(), 'sd': sd.flatten()}
        return yp.flatten(),sd.flatten()

    def Predict(self, test_times):
        """
        Function to generate tidal predictions at any time using the learned model. Automatically uses
        the same parameters which were included to solve the model. 
        
        Parameters
        ----------
        test_times: list or array or dataframe with datetime index
            timestamps to generate tidal predictions

        Returns
        ----------
        yp : array
            tidal predictions
        sd : array
            associated prediction standard deviations
        """

        try:
            self.fit_params
        except:
            raise AttributeError("Model has not been fit yet, call Solve() first before predicting")
        if isinstance(test_times, (pd.DataFrame, pd.Series)):
            test_times = test_times.index
        xte = self.Prepare_Matrix(test_times, np.ones(len(test_times)))
        xte = np.matrix(xte)
        nnew = np.shape(xte)[0]
    
        # run the Bayes linear model with shrinkage
        [w, v, invv, logdetv, an, bn, e_a, l] = self.fit_params

        xtest = xte
    
        xtest = np.concatenate([xtest, np.ones([nnew,1])], axis=1) #add in the bias term
    
        # infer the predictive dist for the test data then un-normalise it
        yp = xtest * w # expectation of y
        e_tau = an / bn # expectation of noise precision
        s2 = 1. / e_tau
        wu = np.matrix(np.zeros([np.shape(xtest)[0],1]))
        for n in range(np.shape(xtest)[0]):    # model uncertainty [weight uncert]
            wu[n, :] = xtest[n,:] * v * xtest[n,:].T
        sd = np.sqrt(s2 + wu)
        return np.array(yp).flatten(), np.array(sd).flatten()