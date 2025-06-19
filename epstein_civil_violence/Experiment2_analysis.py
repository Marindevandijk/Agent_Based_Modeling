import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('waiting_time.csv', header=None).T.squeeze()

plt.hist(df, bins=25)
plt.xlabel("Waiting time between outbursts")
plt.ylabel("Frequency")
plt.title("Distribution of Waiting Times")
plt.savefig('Figures/Experiment_6.png')
plt.show()

df2 = df[df > 30]
print(df2)

plt.hist(df2, bins=25, log=True)
plt.xlabel("Waiting time between outbursts")
plt.ylabel("Frequency (log scale)")
plt.title("Distribution of Waiting Times")
plt.savefig('Figures/Experiment_7.png')
plt.show()