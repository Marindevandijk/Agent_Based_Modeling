import pandas as pd

import matplotlib.pyplot as plt

# region EXPERIMENT 1 ---------------
df = pd.read_csv('Data/model_output.csv')
active = df['active']
time = range(len(active))

plt.plot(time, active)
plt.title('Punctuated equilibrium', size=14)
plt.xlabel('Time', size=12)
plt.ylabel('Figures/Number of active agents', size=12)
plt.savefig('Figures/punct_equilibrium2.pdf')
plt.show()

# region EXPERIMENT 2 ------------------

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

plt.hist(waiting_time, bins=50)
plt.xlabel("Waiting time between outbursts")
plt.ylabel("Frequency")
plt.title("Distribution of Waiting Times")
plt.savefig('Figures/Waiting time distribution.pdf')
plt.show()

# region EXPERIMENT 3 -------------------

plt.plot(df.index, df["tension"], label="Tension",  color="blue")
plt.plot(df.index, df["active"]/1122,  label="Active agents", color="red")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.savefig('Figures/Tension vs Actives.pdf')
plt.show()