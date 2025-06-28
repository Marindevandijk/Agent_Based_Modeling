"""
    Sobol sensitivity analysis for Epstein Civil Violence model
"""
import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from itertools import combinations
from model import EpsteinCivilViolence
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import gc
print(f"Objects in memory: {len(gc.get_objects())}") 

# Define the parameters and their bounds for Sobol analysis
problem = {
    'num_vars': 5,
    'names': [
        'citizen_density',
        'cop_density',
        'legitimacy',
        'max_jail_term',
        'active_threshold'
    ],
    'bounds': [
        [0.6, 0.9],      # citizen_density
        [0.01, 0.1],     # cop_density
        [0.5, 0.99],     # legitimacy
        [1, 30],         # max_jail_term
        [0.05, 0.2]      # active_threshold
    ]
}

replicates = 10  #10
max_steps = 100  #100
distinct_samples = 32 #32 #has to be even for Sobol analysis!

# Generate parameter samples
param_values = saltelli.sample(problem, distinct_samples)

def run_model(args):
    vals, max_steps = args
    params = {
        'height': 40,
        'width': 40,
        'citizen_density': vals[0],
        'cop_density': vals[1],
        'legitimacy': vals[2],
        'max_jail_term': int(round(vals[3])),
        'active_threshold': vals[4],
        'arrest_prob_constant': 2.3,
        'movement': True,
        'max_iters': max_steps,
        'networked': True,
    }
    #print(f"Running with parameters: {params}")
    model = EpsteinCivilViolence(**params)
    model.run_model()
    active_series = model.datacollector.get_model_vars_dataframe()['active']
    mu = active_series.mean()
    sigma = active_series.std()
    cv = sigma / mu if mu != 0 else 0
    del model # clear memory
    gc.collect() # empty garbage collector
    return list(vals) + [cv]

if __name__ == "__main__":
    import csv
    all_args = [(vals, max_steps) for _ in range(replicates) for vals in param_values]
    print(f"Total runs: {len(all_args)}")
    output_file = "epstein_civil_violence/sobol_results.csv"
    columns = problem['names'] + ['CV']
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        with Pool(processes=cpu_count()) as pool:
            for result in tqdm(pool.imap(run_model, all_args), total=len(all_args)):
                writer.writerow(result)

    # Load results from CSV for analysis
    data = pd.read_csv(output_file)
    print(data)

    # Sobol analysis
    Si_cv = sobol.analyze(problem, data['CV'].values.astype(float), print_to_console=True)

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

    # Plot and save 1st, 2nd, and total-order sensitivity for 'CV'
    for i, label, fname in zip(['1', '2', 'T'],
                               ['First order', 'Second order', 'Total order'],
                               ['sobol_first_order', 'sobol_second_order', 'sobol_total_order']):
        plot_index(Si_cv, problem['names'], i, f'{label} sensitivity (CV)', filename=fname)
        plt.show()