"""
    Sobol sensitivity analysis for Epstein Civil Violence model
"""
import numpy as np
from SALib.sample import saltelli
from model.model import EpsteinCivilViolence
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import csv
from multiprocessing import Lock, Manager
import gc

# Define the parameters and their bounds for Sobol analysis
problem = {
    'num_vars': 4,
    'names': [
        'citizen_density',
        'cop_density',
        'legitimacy',
        'active_threshold'
    ],
    'bounds': [
        [0.6, 0.9],      # citizen_density
        [0.01, 0.1],     # cop_density
        [0.5, 0.99],     # legitimacy
        [0.05, 0.2]      # active_threshold
    ]
}

replicates = 10  #10
max_steps = 150  #100
distinct_samples = 16 #5 #has to be even for Sobol analysis!

# Generate parameter samples
param_values = saltelli.sample(problem, distinct_samples)

def run_model_and_save(args):
    vals, max_steps, csv_path, lock = args
    params = {
        'height': 40,
        'width': 40,
        'citizen_density': vals[0],
        'cop_density': vals[1],
        'cop_vision': 7,
        'citizen_vision': 7,
        'legitimacy': vals[2],
        'max_jail_term': 30,
        'active_threshold': vals[3],
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
    row = list(vals) + [cv]
    
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
    columns = problem['names'] + ['cv']
    csv_dir = os.path.join(os.getcwd(), 'epstein_civil_violence', 'Results')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'snellius_sobol_results.csv')
    # Write header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

    manager = Manager()
    lock = manager.Lock()
    all_args = [(vals, max_steps, csv_path, lock) for _ in range(replicates) for vals in param_values]
    print(f"Total runs: {len(all_args)}")
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(run_model_and_save, all_args), total=len(all_args)))

    