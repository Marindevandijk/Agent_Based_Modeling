"""
This script runs the Epstein Civil Violence model and generates plots
to visualize the number of active agents over time, demonstrating the concept of punctuated equilibrium.
"""


from model import EpsteinCivilViolence
import matplotlib.pyplot as plt

model = EpsteinCivilViolence(
    height=40,
    width=40,
    citizen_density=0.8,
    cop_density=0.074,
    citizen_vision=7,
    cop_vision=7,
    legitimacy=0.8,
    max_jail_term=0,
    active_threshold=0.1,
    arrest_prob_constant=2.3,
    movement=True,
    max_iters=1000,
    seed=3,  # Set a seed for reproducibility
)

model.run_model()

model_out = model.datacollector.get_model_vars_dataframe()
active = model_out['active']
time = list(range(len(active)))

# Plotting the number of active agents over time for the first 1000 steps only
#plt.plot(time[:1000], active[:1000])
#plt.title('Punctuated equilibrium',size = 14)
#plt.xlabel('Time',size = 12)
#plt.ylabel('Number of active agents',size = 12)
#plt.savefig('epstein_civil_violence/Figures/punct_equilibrium1.pdf')
#plt.show()


plt.plot(time, active)
plt.title('Punctuated equilibrium',size = 14)
plt.xlabel('Time',size = 12)
plt.ylabel('Figures/Number of active agents',size = 12)
plt.savefig('epstein_civil_violence/Figures/punct_equilibrium2.pdf')
plt.show()
