import numpy as np
import pandas as pd
from SALib.analyze import sobol
from itertools import combinations
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from multiprocessing import Lock, Manager

csv_path = 'fixed_data.csv'

problem = {
    'num_vars': 3,
    'names': [
        'Cop Density',
        'Legitimacy',
        'Active threshold'
    ],
    'bounds': [
        [0.01, 0.1],     # cop_density
        [0.5, 0.99],     # legitimacy
        [0.05, 0.2]      # active_threshold
    ]
}

# Read results from CSV
data = pd.read_csv(csv_path)
print(data)

# Sobol analysis
Si_active = sobol.analyze(problem, data['cv'].values.astype(float), print_to_console=True)

# Plotting function
def plot_index(s, params, i, title='', filename=None):
    """
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.
    """
    if i == '2':
        p = len(params)
        params = list(combinations(params, 2))
        indices = s['S' + i].reshape((p ** 2))
        indices = indices[~np.isnan(indices)]
        errors = s['S' + i + '_conf'].reshape((p ** 2))
        errors = errors[~np.isnan(errors)]
    else:
        indices = s['S' + i]
        errors = s['S' + i + '_conf']
        plt.figure(figsize=(12, 8))  # Wider plot

    l = len(indices)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylim([-0.2, l - 0.8])  # Reduce extra space at the bottom and top
    plt.yticks(range(l), params, fontsize=14, fontweight='bold')
    plt.errorbar(
        indices, range(l), xerr=errors,
        linestyle='None', marker='o',
        markersize=10, markeredgewidth=2, markerfacecolor='blue',
        elinewidth=3, capsize=6, capthick=2
    )
    plt.axvline(0, c='k', linewidth=2)
    plt.tight_layout(pad=0.2)  # Reduce padding further
    
    if filename:
        plt.savefig(f'Figures/{filename}.png')

# Plot and save 1st, 2nd, and total-order sensitivity for 'Active'
for i, label, fname in zip(['1', '2', 'T'],
                            ['First order', 'Second order', 'Total order'],
                            ['sobol_first_order', 'sobol_second_order', 'sobol_total_order']):
    plot_index(Si_active, problem['names'], i, f'{label} sensitivity', filename=fname)
    plt.show()