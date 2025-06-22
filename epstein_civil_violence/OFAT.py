import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import EpsteinCivilViolence

# Define the parameters and their bounds for OFAT
problem = {
    'num_vars': 3,
    'names': ['legitimacy', 'cop_density' ], #, 'citizen_density'],
    'bounds': [[0.5, 0.99], [0.01, 0.1]] #] , [0.6, 0.9]]
}

replicates = 10
max_steps = 100
distinct_samples = 10

# Output: number of active agents at the end
def get_active_agents(model):
    return model.datacollector.get_model_vars_dataframe()['active'].iloc[-1]

data = {}

for i, var in enumerate(problem['names']):
    samples = np.linspace(*problem['bounds'][i], num=distinct_samples)
    results = []

    for value in samples:
        replicate_results = []
        for _ in range(replicates):
            params = {
                'height': 40,
                'width': 40,
                'citizen_density': 0.8,
                'cop_density': 0.074,
                'citizen_vision': 7,
                'cop_vision': 7,
                'legitimacy': 0.8,
                'max_jail_term': 30,
                'active_threshold': 0.1,
                'arrest_prob_constant': 2.3,
                'movement': True,
                'max_iters': max_steps,
            }
            params[var] = value  # Vary only this parameter
            model = EpsteinCivilViolence(**params)
            model.run_model()
            replicate_results.append(get_active_agents(model))
        results.append(replicate_results)
    # Store as DataFrame for easier plotting
    df = pd.DataFrame(results, index=samples)
    data[var] = df

# Plotting
for var in problem['names']:
    df = data[var]
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    plt.figure()
    plt.errorbar(df.index, means, yerr=1.96*stds/np.sqrt(replicates), fmt='-o')
    plt.xlabel(var)
    plt.ylabel('Active agents at end')
    plt.title(f'OFAT Sensitivity: {var}')
    plt.tight_layout()
    plt.savefig(f'epstein_civil_violence/Figures/OFAT_{var}.pdf')
    plt.show()