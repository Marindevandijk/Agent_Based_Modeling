import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from itertools import combinations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import EpsteinCivilViolence
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import csv
import tempfile
from multiprocessing import Lock, Manager
import gc

# Define the parameters and their bounds for Sobol analysis
problem = {
    'num_vars': 3,
    'names': [
        'cop_density',
        'legitimacy',
        'active_threshold'
    ],
    'bounds': [
        [0.01, 0.1],     # cop_density
        [0.5, 0.99],     # legitimacy
        [0.05, 0.2]      # active_threshold
    ]
}

replicates = 10  #10
max_steps = 100  #100
distinct_samples = 16 #5 #has to be even for Sobol analysis!

# Generate parameter samples
param_values = saltelli.sample(problem, distinct_samples)

# Load the CSV file
df = pd.read_csv('3param_sobol_results.csv')

# Sort by the first column (assumed to be 'index' or unnamed)
first_col = df.columns[0]
df_sorted = df.sort_values(by=first_col).reset_index(drop=True)


def run_model_and_save(args):
    vals = args
    params = {
        'height': 40,
        'width': 40,
        'citizen_density': 0.7,
        'cop_density': vals[0],
        'cop_vision': 7,
        'citizen_vision': 7,
        'legitimacy': vals[1],
        'max_jail_term': 30,
        'active_threshold': vals[2],
        'arrest_prob_constant': 2.3,
        'movement': True,
        'max_iters': 100,
        'networked': True,
    }
    model = EpsteinCivilViolence(**params)
    model.run_model()
    
    # Calculate coefficient of variation (CV) for 'active'
    active_series = model.datacollector.get_model_vars_dataframe()['active'][20:]
    mean_active = np.mean(active_series)
    std_active = np.std(active_series)
    cv = std_active / mean_active if mean_active != 0 else 0
    return cv

indices = df_sorted[first_col].values
is_sequential = True
for i in range(1, len(indices)):
    if indices[i] != indices[i-1] + 1:
        print(f"Sequentiality breaks at line {i}: {indices[i-1]} -> {indices[i]}")
        is_sequential = False
        
        # Find the missing index
        missing_idx = indices[i-1] + 1

        # Run the model for the missing parameter set
        # Each unique parameter set is repeated 'replicates' times in the CSV
        # To find the correct parameter set for the missing index:
        param_set_idx = missing_idx // replicates
        missing_param = param_values[param_set_idx]
        cv = run_model_and_save(missing_param)

        # Create a new row with the missing index, parameter values, and cv
        row = [missing_idx] + list(missing_param) + [cv]
        new_row = pd.DataFrame([row], columns=df_sorted.columns)

        # Insert the new row into the DataFrame at the correct position
        df_sorted = pd.concat([
            df_sorted.iloc[:i],
            new_row,
            df_sorted.iloc[i:]
        ]).reset_index(drop=True)

        print(f"Inserted missing row for index {missing_idx}")

df_sorted.to_csv('fixed_data.csv', index=False)
        