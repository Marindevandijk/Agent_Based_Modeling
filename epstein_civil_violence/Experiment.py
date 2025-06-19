
from model import EpsteinCivilViolence
import matplotlib.pyplot as plt

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
    seed=None,
)

model.run_model()

model_out = model.datacollector.get_model_vars_dataframe()
active = model_out['active']
time = list(range(len(active)))
plt.plot(time[:1000], active[:1000])
plt.title('Punctuated equilibrium',size = 14)
plt.xlabel('Time',size = 12)
plt.ylabel('Number of active agents',size = 12)
plt.savefig('Figures/Experiment_fig3.png')
plt.show()

plt.plot(time, active)
plt.title('Punctuated equilibrium',size = 14)
plt.xlabel('Time',size = 12)
plt.ylabel('Figures/Number of active agents',size = 12)
plt.savefig('Experiment_fig4.png')
plt.show()
