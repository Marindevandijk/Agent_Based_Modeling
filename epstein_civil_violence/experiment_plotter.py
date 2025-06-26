import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_experiments(data_file_path, networked=False):

    # Determine the Figures subdirectory based on 'networked'
    figures_dir = os.path.join('Figures', 'networked' if networked else 'non_networked')
    os.makedirs(figures_dir, exist_ok=True)

    # region EXPERIMENT 1 ---------------
    df = pd.read_csv(data_file_path)
    active = df['active']
    time = range(len(active))

    plt.plot(time[:1000], active[:1000])
    plt.xlabel('Time', size=12)
    plt.ylabel('Number of active agents', size=12)
    plt.savefig(os.path.join(figures_dir, f'puncEq_net_{networked}.pdf'))
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

    min_wt, max_wt = min(waiting_time), max(waiting_time)
    bins = np.arange(min_wt, max_wt + 2) - 0.5  # bin edges centered on integers
    plt.hist(waiting_time, bins=bins)
    
    plt.xlabel("Waiting time between outbursts")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(figures_dir, f'Wait_Time_net_{networked}.pdf'))
    plt.show()

    # region EXPERIMENT 3 -------------------

    plt.plot(df.index[:500], df["tension"][:500], label="Tension",  color="blue")
    tot_citizens = int(df.iloc[0, :3].sum())
    plt.plot(df.index[:500], df["active"][:500]/tot_citizens,  label="Active agents", color="red")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f'tension_net_{networked}.pdf'))
    plt.show()
    
if __name__ == "__main__":
    plot_experiments("Data/output_networked.csv", networked=True)
    plot_experiments("Data/output_non_networked.csv", networked=False)