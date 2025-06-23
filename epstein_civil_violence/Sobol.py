"""
    Sobol sensitivity analysis for Epstein Civil Violence model
"""
from SALib.sample import saltelli
from SALib.analyze import sobol
#from mesa.batchrunner import BatchRunner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from model import EpsteinCivilViolence

# Define the parameters and their bounds for Sobol analysis
problem = {
    'num_vars': 2,
    'names': ['legitimacy', 'cop_density'],
    'bounds': [[0.5, 0.99], [0.01, 0.1]]
}

replicates = 10
max_steps = 100
distinct_samples = 10

# Generate parameter samples
param_values = saltelli.sample(problem, distinct_samples)

# Model reporters: collect number of active agents at the end
def get_active_agents(model):
    return model.datacollector.get_model_vars_dataframe()['active'].iloc[-1]

model_reporters = {"Active": get_active_agents}

# Prepare DataFrame to store results
data = pd.DataFrame(index=range(replicates * len(param_values)),
                    columns=problem['names'] + ['Active'])

count = 0
for i in range(replicates):
    for vals in param_values:
        vals = list(vals)
        # If you have integer parameters, cast them here (not needed for legitimacy/cop_density)
        params = {
            'height': 40,
            'width': 40,
            'citizen_density': 0.8,
            'cop_density': vals[1],
            'citizen_vision': 7,
            'cop_vision': 7,
            'legitimacy': vals[0],
            'max_jail_term': 30,
            'active_threshold': 0.1,
            'arrest_prob_constant': 2.3,
            'movement': True,
            'max_iters': max_steps,
        }
        model = EpsteinCivilViolence(**params)
        model.run_model()
        data.iloc[count, 0:2] = vals
        data.iloc[count, 2] = get_active_agents(model)
        count += 1
        print(f'{count / (len(param_values) * replicates) * 100:.2f}% done')

print(data)

# Sobol analysis
Si_active = sobol.analyze(problem, data['Active'].values.astype(float), print_to_console=True)

# Plotting function
def plot_index(s, params, i, title=''):
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
        plt.figure()

    l = len(indices)
    plt.title(title)
    plt.ylim([-0.2, len(indices) - 1 + 0.2])
    plt.yticks(range(l), params)
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
    plt.axvline(0, c='k')

# Plot 1st, 2nd, and total-order sensitivity for 'Active'
for i, label in zip(['1', '2', 'T'], ['First order', 'Second order', 'Total order']):
    plot_index(Si_active, problem['names'], i, f'{label} sensitivity')
    plt.show()
