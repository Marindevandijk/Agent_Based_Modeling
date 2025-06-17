
from model import EpsteinCivilViolence
import matplotlib.pyplot as plt


model = EpsteinCivilViolence(
    height=40,
    width=40,
    citizen_density=0.7,
    cop_density=0.074,
    citizen_vision=7,
    cop_vision=7,
    legitimacy=0.8,
    max_jail_term=1000,
    max_iters=1000,
)  # cap the number of steps the model takes
model.run_model()

model_out = model.datacollector.get_model_vars_dataframe()

ax = model_out.plot()
ax.set_title("Citizen Condition Over Time")
ax.set_xlabel("Step")
ax.set_ylabel("Number of Citizens")
_ = ax.legend(bbox_to_anchor=(1.35, 1.025))

plt.show()
