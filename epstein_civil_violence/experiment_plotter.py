import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_experiments(data_file_path, networked):

    # Determine the Figures subdirectory based on 'networked'
    figures_dir = os.path.join('Figures', 'networked' if networked else 'non_networked')
    os.makedirs(figures_dir, exist_ok=True)

    # region EXPERIMENT 1 ---------------
    df = pd.read_csv(data_file_path)
    active = df['active']
    time = range(len(active))

    plt.plot(time[:500], active[:500])
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


    waiting_time = np.array(waiting_time)
    waiting_time = waiting_time[waiting_time<280]
    min_wt, max_wt = min(waiting_time), max(waiting_time)
    bin_width = 3
    bins = np.arange(min_wt, max_wt + bin_width, bin_width) - 0.5
    print('mean',np.mean(waiting_time))
    print('std',np.std(waiting_time))
    cv = np.std(waiting_time) / np.mean(waiting_time)
    print('CV', cv)
    
    plt.hist(waiting_time, bins=bins,edgecolor="black")
    plt.xlabel("Waiting time between outbursts")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(figures_dir, f'Wait_Time_net_{networked}.pdf'))
    plt.show()

    # truncated waiting time from values above 30 
    
    waiting_time_trunc = waiting_time[waiting_time>30]
    min_wt, max_wt = min(waiting_time_trunc), max(waiting_time_trunc)
    bin_width = 3

    counts, bin_edges = np.histogram(waiting_time_trunc, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    nonzero = counts > 0
    counts = counts[nonzero]
    bin_centers = bin_centers[nonzero]

    log_counts = np.log10(counts)
    slope, intercept = np.polyfit(bin_centers, log_counts, 1)
    fit_x = np.linspace(min(bin_centers), max(bin_centers), 100)
    fit_y = 10 ** (slope * fit_x + intercept)
    print(f"Slope of plot: {slope:.3f}")
    
    bins = np.arange(min_wt, max_wt + bin_width, bin_width) - 0.5
    mid_x = 0.5 * (min(fit_x) + max(fit_x))
    mid_y = 10 ** (slope * mid_x + intercept+ 0.5)

    plt.hist(waiting_time_trunc, bins=bins,log=True,edgecolor="black")
    plt.text(mid_x, mid_y, f"$y = {slope:.2f}x + {intercept:.2f}$", fontsize=10, color='red')
    plt.plot(fit_x, fit_y, linestyle='dotted', color='red', label=f"Slope ≈ {slope:.2f}")
    plt.xlabel("Waiting time between outbursts")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(figures_dir, f'Wait_Time_trunced_net_{networked}.pdf'))
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

    