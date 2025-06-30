# Agent-Based Modeling of Civil Violence

This repository implements the Epstein Civil Violence model in Python using the Mesa framework, with extensions for networked agent interactions and global sensitivity analysis (Sobol). The codebase supports both grid-based and network-based simulations, batch experiments, and comprehensive result analysis and visualization.

---

## Features

- **Network-Extended Epstein Civil Violence Model**: Simulates citizens and cops on a grid or network, following the rules from Epstein (2002).
- **Network Dynamics**: Optionally considers citizen communication on a superimposed scale-free network (Barabási–Albert) to study social structure effects.
- **Experimentation**: Run the original experiments from the Epstein (2002) paper, with or without the network dynamics.
- **Sobol Sensitivity Analysis**: Quantifies the influence of model parameters using SALib.
- **Visualization**: Generates plots for time series, waiting times, tension, and Sobol indices.
- **Reproducibility**: Supports fixed seeds and outputs all results to CSV for further analysis.

---

## Directory Structure

```
.
├── Data/
│   ├── batched_sobol_results.csv (example)
│   ├── batched_sobol_results_m2.csv (example)
│   ├── output_networked.csv (example)
│   └── output_non_networked.csv (example)
├── Figures/
│   ├── networked/
│   ├── non_networked/
│   └── sobol_plots/
├── model/
│   ├── agents.py
│   ├── model.py
│   └── __init__.py
├── deprecated/
│   ├── snellius_sobol.py
│   ├── testing.py
│   └── env_snellius.yml
├── experiments.py
├── requirements.txt
└── sobol.py
```

---

## Recommended usage

### 1. Install Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 2. Run Experiments

To run the main experiments and generate output data (circa 2 mins):

```bash
python experiments.py --generate
```

If you wish to simply view previous results:
```bash
python experiments.py
```

This will produce CSV files in the `Data/` directory and figures in `Figures/networked/` and `Figures/non_networked/`.

To edit the model parameters, see run_experiment() in experiments.py

### 3. Run Sobol Sensitivity Analysis


To perform a sobol sensitivity analysis using previous data: 

```bash
python sobol.py --no_run
```



If you wish to generate your own data first:  
NOTE: with current settings this takes circa 3 hours
```bash
python sobol.py
```

### 4. Visualize Results

Figures are saved in the `Figures/` subdirectories. You can also use the plotting functions in `experiments.py` or `experiment_plotter.py` to generate additional plots.

---

## Model Parameters

Key parameters (see `model/model.py`):

- `citizen_density`: Proportion of citizens on the 40x40 grid.
- `cop_density`: Proportion of cops on the grid.
- `legitimacy`: Regime legitimacy.
- `active_threshold`: Citizen activation threshold.
- `max_jail_term`: Maximum jail time for arrested citizens.
- `networked`: Whether to use a network structure for citizens.
- `m`: Number of edges to attach from a new node in the Barabási–Albert network.

---

## Data & Output

- **CSV files** in `Data/` contain time series and batch experiment results.
- **Figures** in `Figures/` show time series, waiting time histograms, tension plots, and Sobol sensitivity indices.

---

## References

- Epstein, J. M. (2002). Modeling civil violence: An agent-based computational approach. *PNAS*, 99(suppl 3), 7243-7250.
- [Mesa: Agent-based modeling in Python](https://mesa.readthedocs.io/)
- [SALib: Sensitivity Analysis Library in Python](https://salib.readthedocs.io/)

---

## License

MIT License

---