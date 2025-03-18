A Python implementation of the variational Bayesian harmonic analysis proposed in https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JC021533. 

# Installation
```
pip install vtide
```
# Usage
## Standard Analysis
Suppose we have a pandas dataframe containing sea-level measurements with a date-time index containing the time of each measurement. A complete RTide analysis can be run in just three lines of code: 
```
df = pd.DataFrame({'observations': observations}, index = times)

model = VTide(df, lat, lon)
yp, sd = model.Solve()
```
A model has now been trained, and the constituents estimated from the provided data. Solve() will return two arrays, yp and sd which correspond to the tidal predictions and the associated standard deviations respectively. Solve has several additional functionalities, the most useful is the `prune' feature which will perform probabilistic outlier removal based on a user defined threshold. The fit results can be visualized using:
```
model.Visualize_Residuals()
```
If pruning it used, the bottom KDE plot will show the distribution of residuals without the outliers. This should be approximately Gaussian if things have gone right! We can now use the learned model to generate probabilistic predictions whenever we want. All we need is to provide a pandas time-index with the desired times:
```
yp, sd = model.Predict(test_times)
```

## Constituents
After calling Solve(), the derived constituents and associated uncertainty estimates can be accessed directly through a dictionary using:
```
consts = model.harmonics

amplitudes = [consts[const]['amp'] for const in consts]
amplitude_uncertainty = [consts[const]['amp_uncert'] for const in consts]
phases = [consts[const]['phase'] for const in consts]
phase_uncertainty = [consts[const]['phase_uncert'] for const in consts]
```
Constituents and confidence intervals can quickly be visualized by calling:
```
model.Visualize_Constituents()
```







