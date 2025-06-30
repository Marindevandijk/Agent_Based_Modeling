'''
Sobol materials and runner.

run_model() defines a worker to obtain data for each sobol parameter combination and runs the collection multiprocess.
analyze_and_plot_sobol() analyzes and plots sobol analysis results
'''

from SALib.sample import saltelli
from model.model import EpsteinCivilViolence
from multiprocessing import Pool
from tqdm import tqdm
import gc
import csv
import os

import numpy as np
import pandas as pd
from SALib.analyze import sobol
import matplotlib.pyplot as plt

def analyze_and_plot_sobol(problem, csv_path):
    """
    Takes the saltelli sample matrix and runs sobol analysis.

    Parameters
    ----------
    problem : ndarray
        saltelli sample matrix
        
    csv_path : string
    """    
    
    csv_paths = {
    'model 1': csv_path,
    }

    results = {}
    for label, path in csv_paths.items():
        data = pd.read_csv(path)
        Si = sobol.analyze(problem, data['cv'].values.astype(float), print_to_console=False, calc_second_order=True)
        results[label] = Si

    params = problem['names']
    x = np.arange(len(params))
    width = 0.25

    orders = [
        ('S1', 'First-order Sobol index'),
        ('ST', 'Total-order Sobol index')
    ]

    for order, ylabel in orders:
        _, ax = plt.subplots(figsize=(7, 6))
        for i, (label, color) in enumerate(zip(['10', '2'], ['#1f77b4', '#ff7f0e'])):
            Si = results[label]
            ax.bar(x + i*width, Si[order], width, yerr=Si[f'{order}_conf'], label=f'm={label}', color=color, capsize=5, edgecolor='black')
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(params, fontsize=13)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=13)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'Figures/sobol_plots/sobol_comparison_{order.lower()}_order.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        

def run_model(index_args):
    """
    Defines a worker for use in multiprocessing Sobol sampling. 

    Parameters
    ----------
    index_args : tuple containing (index, args)
        
        index : int  
            identification index for the run (helps to keep track of which sample of the saltelli matrix is used)  
              
        args : array  
            contains the model parameters:  

                citizen_density : float  
                cop_density : float  
                legitimacy : float  
                active_threshold : float  
            
    Returns
    -------
    Array to be written as a row in the csv for Sobol analysis:
    [index] + list(vals) + [cv]
    """    
    
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
        'm': 10
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
    
    # Define the parameters and their bounds for the construction of saltelli matrix
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

    ''' # used for report 
            [0.4, 0.8],
            [0.01, 0.1],
            [0.2, 0.99],
            [0.05, 0.5],
    '''
    replicates = 10
    max_steps = 100
    distinct_samples = 32
    batch_size = 100
    output_file = "Data/batched_sobol_results.csv"

    param_values = saltelli.sample(problem, distinct_samples)
    # add an index to each parameter combination for tracking purposes
    all_args = [((i, args), max_steps) for i, args in enumerate([(vals, max_steps) for _ in range(replicates) for vals in param_values])]
    total_runs = len(all_args)

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

    # runs are divided up into batches for sake of RAM
    for i in range(0, len(all_args), batch_size):
        print(f"Running batch {i+completed} to {min(i + batch_size + completed, total_runs)}...")
        batch_args = all_args[i:i + batch_size]
        with Pool(processes=4) as pool: # num_processes set to 4 due to limited RAM
            with open(output_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                for result in tqdm(pool.imap(run_model, batch_args), total=len(batch_args)):
                    writer.writerow(result)
        gc.collect()
        
    analyze_and_plot_sobol(problem)