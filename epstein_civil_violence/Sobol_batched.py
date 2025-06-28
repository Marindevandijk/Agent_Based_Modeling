from SALib.sample import saltelli
from model import EpsteinCivilViolence
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import gc
import csv
import os

# Define the parameters and their bounds for Sobol analysis
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

''' # keep these saved 
        [0.4, 0.8],
        [0.01, 0.1],
        [0.2, 0.99],
        [0.05, 0.5],
'''
replicates = 10
max_steps = 100
distinct_samples = 32
batch_size = 100 # 1000
output_file = "./batched_sobol_results_waittime30.csv"

param_values = saltelli.sample(problem, distinct_samples)
all_args = [((i, args), max_steps) for i, args in enumerate([(vals, max_steps) for _ in range(replicates) for vals in param_values])]
total_runs = len(all_args)

def run_model(index_args):
    (index, args), max_steps = index_args
    vals, _ = args
    params = {
        'height': 40,
        'width': 40,
        'citizen_density': vals[0],
        'cop_density': vals[1],
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
    active_series = model.datacollector.get_model_vars_dataframe()['active']
    mu = active_series.mean()
    sigma = active_series.std()
    cv = sigma / mu if mu != 0 else 0
    del model
    gc.collect()
    return [index] + list(vals) + [cv]

if __name__ == "__main__":
    
    columns = ['index'] + problem['names'] + ['cv']
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            completed = sum(1 for _ in f) - 1  # minus header
    else:
        with open(output_file, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
        completed = 0

    print(f"Total runs: {total_runs}, Completed: {completed}")
    all_args = all_args[completed:]

    for i in range(0, len(all_args), batch_size):
        print(f"Running batch {i+completed} to {min(i + batch_size + completed, total_runs)}...")
        batch_args = all_args[i:i + batch_size]
        with Pool(processes=4) as pool:
            with open(output_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                for result in tqdm(pool.imap(run_model, batch_args), total=len(batch_args)):
                    writer.writerow(result)
        gc.collect()