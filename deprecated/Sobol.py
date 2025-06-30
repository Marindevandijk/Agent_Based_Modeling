"""
    Sobol sensitivity analysis for Epstein Civil Violence model
"""
import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from itertools import combinations
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

def run_model_and_save(index_and_args):
    index, args = index_and_args
    vals, max_steps, csv_path, lock = args
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
        'max_iters': max_steps,
        'networked': True,
    }
    model = EpsteinCivilViolence(**params)
    model.run_model()
    
    # Calculate coefficient of variation (CV) for 'active'
    active_series = model.datacollector.get_model_vars_dataframe()['active'][20:]
    mean_active = np.mean(active_series)
    std_active = np.std(active_series)
    cv = std_active / mean_active if mean_active != 0 else 0
    row = [index] + list(vals) + [cv]
    
    # Write to CSV with lock to avoid race conditions
    with lock:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            
    # Explicitly delete objects and collect garbage
    del model
    gc.collect()
    
    return None  # No need to return results

if __name__ == "__main__":
    # Prepare CSV file
    columns = ['index'] + problem['names'] + ['cv']
    csv_dir = os.path.join(os.getcwd(), 'epstein_civil_violence', 'Results')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'sobol_results.csv')
    # Write header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

    manager = Manager()
    lock = manager.Lock()
    all_args = [(vals, max_steps, csv_path, lock) for _ in range(replicates) for vals in param_values]
    print(f"Total runs: {len(all_args)}")
    with Pool(processes=4) as pool:
        list(tqdm(pool.imap_unordered(run_model_and_save, [(i, args) for i, args in enumerate(all_args)]), total=len(all_args)))

    # Read results from CSV
    data = pd.read_csv(csv_path)
    print(data)

    # Sobol analysis
    Si_active = sobol.analyze(problem, data['Active'].values.astype(float), print_to_console=True)

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
        plt.title(title)
        plt.ylim([-0.2, l - 1 + 0.2])
        plt.yticks(range(l), params)
        plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
        plt.axvline(0, c='k')
        plt.tight_layout()
        if filename:
            plt.savefig(f'epstein_civil_violence/Figures/{filename}.pdf')

    # Plot and save 1st, 2nd, and total-order sensitivity for 'Active'
    for i, label, fname in zip(['1', '2', 'T'],
                               ['First order', 'Second order', 'Total order'],
                               ['sobol_first_order', 'sobol_second_order', 'sobol_total_order']):
        plot_index(Si_active, problem['names'], i, f'{label} sensitivity', filename=fname)
        plt.show()
