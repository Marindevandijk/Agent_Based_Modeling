# Epstein Civil Violence Model with Social Network Extension

## Summary

This project extends Joshua Epstein’s original model of civil violence by integrating **network structures** to better simulate the influence of **social media and peer communication**. In this networked version, citizen agents are influenced not only by their immediate vision range but also by their connections in a social network. This addition allows us to study how digital communication channels affect the dynamics of protest and suppression.

## Files

* ``model.py``: Core model code.
* ``agent.py``: Agent classes.
* ``app.py``: Sets up the interactive visualization.
* ``experiment_runner.py``: Use this file to run the model with chosen parameters.
* ``experiment_plotter.py``: After running `experiment_runner.py`, this file plots the simulation results.
* ``Sobol.py``: Runs Sobol sensitivity analysis on key model parameters.

## Requirements

To install the required packages:

```bash
$ pip install -U mesa
$ pip install -U mesa[rec]
$ pip install numpy pandas matplotlib tqdm networkx SALib 
```

## How to Run

To run the model interactively, in this directory, run the following command

```
    $ solara run app.py
```

## Further Reading

This model is based adapted from:

[Epstein, J. “Modeling civil violence: An agent-based computational approach”, Proceedings of the National Academy of Sciences, Vol. 99, Suppl. 3, May 14, 2002](http://www.pnas.org/content/99/suppl.3/7243.short)

A similar model is also included with NetLogo:

Wilensky, U. (2004). NetLogo Rebellion model. http://ccl.northwestern.edu/netlogo/models/Rebellion. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.
