import numpy as np
import pandas as pd
from SALib.analyze import sobol
import seaborn as sns
import matplotlib.pyplot as plt

csv_paths = {
    '10': 'batched_sobol_results_m10.csv',
    '2': 'batched_sobol_results_m2.csv'
}

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

results = {}
for label, path in csv_paths.items():
    data = pd.read_csv(path)
    Si = sobol.analyze(problem, data['cv'].values.astype(float), print_to_console=False, calc_second_order=True)
    results[label] = Si

params = problem['names']
x = np.arange(len(params))
width = 0.25

orders = [
    ('S1', 'First-order', 'First-order Sobol index'),
    ('ST', 'Total-order', 'Total-order Sobol index')
]

for order, title, ylabel in orders:
    fig, ax = plt.subplots(figsize=(7, 6))
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
    plt.savefig(f'Figures/sobol_comparison_{order.lower()}_order.pdf', dpi=300, bbox_inches='tight')
    plt.show()