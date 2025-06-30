"""
    Sobol analysis and plotting from batched results CSV
"""
import numpy as np
import pandas as pd
from SALib.analyze import sobol
from itertools import combinations
import matplotlib.pyplot as plt
import os

# Define the parameters and their bounds for Sobol analysis (from batched script)
problem = {
    'num_vars': 4,
    'names': [
        'citizen_density',
        'cop_density',
        'legitimacy',
        'active_threshold',
    ],
    'bounds': [
        [0.4, 0.8],
        [0.01, 0.1],
        [0.2, 0.99],
        [0.05, 0.5],
    ]
}

# Path to the batched results CSV file
csv_path = './batched_sobol_results_waittime30.csv'

# Read results from CSV
data = pd.read_csv(csv_path)

# If there are replicates, average over them for each parameter set
# (Assume index + 4 params define a unique run)
if 'index' in data.columns:
    group_cols = ['citizen_density', 'cop_density', 'legitimacy', 'active_threshold']
    data_grouped = data.groupby(group_cols, as_index=False)['cv'].mean()
    y = data_grouped['cv'].values.astype(float)
else:
    y = data['cv'].values.astype(float)

# Sobol analysis
Si = sobol.analyze(problem, y, print_to_console=True)

# Plotting function
def plot_index(s, params, i, title='', filename=None):
    if i == '2':
        p = len(params)
        params2 = list(combinations(params, 2))
        indices = s['S' + i].reshape((p ** 2))
        indices = indices[~np.isnan(indices)]
        errors = s['S' + i + '_conf'].reshape((p ** 2))
        errors = errors[~np.isnan(errors)]
        labels = [f'{a},{b}' for a, b in params2]
    else:
        indices = s['S' + i]
        errors = s['S' + i + '_conf']
        labels = params
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.ylim([-0.2, len(indices) - 1 + 0.2])
    plt.yticks(range(len(indices)), labels)
    plt.errorbar(indices, range(len(indices)), xerr=errors, linestyle='None', marker='o')
    plt.axvline(0, c='k')
    plt.tight_layout()
    if filename:
        outdir = 'epstein_civil_violence/Figures/sobol_analysis'
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f'{outdir}/{filename}.pdf')
    plt.show()

# Plot and save 1st, 2nd, and total-order sensitivity for 'cv'
for i, label, fname in zip(['1', '2', 'T'],
                           ['First order', 'Second order', 'Total order'],
                           ['sobol_first_order', 'sobol_second_order', 'sobol_total_order']):
    plot_index(Si, problem['names'], i, f'{label} sensitivity', filename=fname)
