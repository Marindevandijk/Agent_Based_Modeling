"""
This script runs the Epstein Civil Violence model with specified parameters and plots the results. To plot the tension and active agents over time.
"""


from model import EpsteinCivilViolence
import matplotlib.pyplot as plt


"""
model = EpsteinCivilViolence(max_iters=10)  
model.run_model()
print(model.datacollector.get_model_vars_dataframe().head())

"""

model = EpsteinCivilViolence(
    height=40,
    width=40,
    citizen_density=0.7,
    cop_density=0.04,
    citizen_vision=7,
    cop_vision=7,
    legitimacy=0.82,
    max_jail_term=30,  # infinite jail term
    active_threshold=0.1,
    arrest_prob_constant=2.3,
    movement=True,
    max_iters=1000,
    seed=1,  # Set a seed for reproducibility
)

model.run_model()
df = model.datacollector.get_model_vars_dataframe()

plt.plot(df.index, df["tension"], label="Tension",  color="blue")
plt.plot(df.index, df["active"]/1122,  label="Active agents", color="red")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.savefig('epstein_civil_violence/Figures/Tension vs Actives.pdf')
plt.show()
