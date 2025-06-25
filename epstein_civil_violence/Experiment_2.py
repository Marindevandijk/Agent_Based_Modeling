"""
This script runs the Epstein Civil Violence model and analyzes the waiting times
between outbursts of civil violence.
"""

from model import EpsteinCivilViolence
import matplotlib.pyplot as plt
import csv
import numpy as np

model = EpsteinCivilViolence(
    height=40,
    width=40,
    citizen_density=0.7,
    cop_density=0.04,
    citizen_vision=7,
    cop_vision=7,
    legitimacy=0.75,
    max_jail_term=30,
    active_threshold=0.1,
    arrest_prob_constant=2.3,
    movement=True,
    max_iters=1000,
    seed=3, # Set a seed for reproducibility
) 

model.run_model()

model_out = model.datacollector.get_model_vars_dataframe()
active = np.array(model_out['active'])
waiting_time = []
last_end = 0
outburst = False
for i, a in enumerate(active):
    if not outburst and a > 50:
        if last_end != 0:
            waiting_time.append(i - last_end)
        outburst = True
    if outburst and a <= 50:
        last_end = i
        outburst = False

with open('waiting_time.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(waiting_time)

plt.hist(waiting_time, bins=50)
plt.xlabel("Waiting time between outbursts")
plt.ylabel("Frequency")
plt.title("Distribution of Waiting Times")
plt.savefig('Figures/Waiting time distribution.pdf')
plt.show()