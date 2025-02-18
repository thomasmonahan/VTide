# tests/test_vtide.py

import pytest
import numpy as np
import pandas as pd
import datetime
import matplotlib
# Use a non-interactive backend for testing plots.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

# Import your VTide class from your package.
# Adjust the import statement as needed.
from vtide import VTide

# ----------------------------
# Fixture for sample data
# ----------------------------
@pytest.fixture
def sample_data():
    """
    Returns a DataFrame with a DatetimeIndex and a simple sine‐wave (plus noise)
    as 'observations'. (Hourly data for 7 days.)
    """
    constit_freqs = [8.05114007e-02, 7.89992487e-02, 8.33333333e-02, 8.20235526e-02]
    amplitudes = [2.69536915e+00, 5.72051193e-01, 4.07506583e-01, 1.77388101e-01]
    phases = [100.17759146, 67.89243773, 139.00250991, 137.86574464]
    times = pd.date_range(start='2024-01-01 00:00:00+00:00', periods=24*7, freq="1H", tz='UTC')
    time_vals = times.to_julian_date().to_numpy()
    individual_tides = []
    for i, j in enumerate(constit_freqs):
        individual_tides.append(amplitudes[i]*np.cos(24*j*np.array(time_vals) * 2 * np.pi + (phases[i] * np.pi / 180)))
    tide = np.sum(individual_tides, axis=0) + 0.1*np.random.normal(len(individual_tides))
    df = pd.DataFrame({'observations': tide}, index=times)
    return df

# ----------------------------
# Test input validations
# ----------------------------
def test_input_validation(sample_data):
    # Non-DatetimeIndex should raise ValueError
    df = sample_data.copy()
    df.index = np.arange(len(df))
    with pytest.raises(ValueError):
        VTide(df, lat=45, lon=-67)
        
    # Non-numeric latitude
    with pytest.raises(ValueError):
        VTide(sample_data, lat="not a number", lon=-67)
        
    # Non-numeric longitude
    with pytest.raises(ValueError):
        VTide(sample_data, lat=45, lon="not a number")

# ----------------------------
# Test Solve and Predict functionality
# ----------------------------
def test_solve_and_predict(sample_data):
    model = VTide(sample_data, lat=44.9, lon=-67)
    
    # Use auto constituents and no pruning.
    yp, sd = model.Solve(constituents='auto', nodal=False, trend=False, prune=False, aleatoric_uncert=True)
    
    # Ensure predictions have the same length as input observations.
    assert len(yp) == len(sample_data)
    assert len(sd) == len(sample_data)
    
    # Test that Predict() raises an error if the model has not been solved.
    model2 = VTide(sample_data, lat=44.9, lon=-67)
    with pytest.raises(AttributeError):
        model2.Predict(sample_data.index)
        
    # Now, after solving, call Predict on the same timestamps.
    model.Solve(constituents='auto', nodal=False, trend=False, prune=False, aleatoric_uncert=True)
    yp_pred, sd_pred = model.Predict(sample_data.index)
    assert len(yp_pred) == len(sample_data.index)
    assert len(sd_pred) == len(sample_data.index)

# ----------------------------
# Test pruning functionality
# ----------------------------
def test_prune_functionality(sample_data):
    # Insert artificial outliers into the observations.
    df = sample_data.copy()
    obs = df['observations'].values.copy()
    # For example, set index 5 and 15 to extreme values.
    obs[5] += 10
    obs[15] -= 10
    df['observations'] = obs
    
    model = VTide(df, lat=44.9, lon=-67)
    # Solve with a prune threshold (e.g., 3 standard deviations)
    yp, sd = model.Solve(constituents='auto', nodal=False, trend=False, prune=3, aleatoric_uncert=True)
    
    # After pruning, the model should have recorded some bad indices.
    assert hasattr(model, 'bad_indices')
    assert len(model.bad_indices) > 0

# ----------------------------
# Test plotting functions (visual tests)
# ----------------------------
def test_plot_functions(sample_data):
    model = VTide(sample_data, lat=44.9, lon=-67)
    model.Solve(constituents='auto', nodal=False, trend=False, prune=False, aleatoric_uncert=True)
    
    # Test that Plot_Constituents runs without error.
    try:
        model.Plot_Constituents()
    except Exception as e:
        pytest.fail("Plot_Constituents() raised an exception: {}".format(e))
        
    # Test that Visualize_Predictions runs without error.
    try:
        model.Visualize_Predictions()
    except Exception as e:
        pytest.fail("Visualize_Predictions() raised an exception: {}".format(e))
