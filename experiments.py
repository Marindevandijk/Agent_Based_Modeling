'''
Recreates the experiments from the original 2002 epstein paper. Both with a non-networked and with a networked model.
'''

from model.model import EpsteinCivilViolence
import time
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def run_experiment(output_path, networked):
    """
    Runs civil violence model and writes results containing # of actives and tension to csv at output_path.
    All model variables except for networked are hard-coded.

    Parameters
    ----------
    output_path : string
    networked : boolean
        whether the model ran is the networked version or not
    """    
    model = EpsteinCivilViolence(
        height=40,
        width=40,
        citizen_density=0.7,
        cop_density=0.06,
        citizen_vision=7,
        cop_vision=7,
        legitimacy=0.82,
        max_jail_term=30,
        active_threshold=0.1,
        arrest_prob_constant=2.3,
        movement=True,
        max_iters=500,
        seed=42,
        networked=networked
    )

    start_time = time.time()
    model.run_model()
    end_time = time.time()
    print(f"Model run time: {end_time - start_time:.2f} seconds")
    print(f"{(end_time - start_time )/model.max_iters:.2f} sec per iteration")

    model_out = model.datacollector.get_model_vars_dataframe()
    model_out.to_csv(output_path, index=False)
    
def plot_experiments(data_file_path, networked):
    """
    Plots epstein experiment results.

    Parameters
    ----------
    data_file_path : string
    networked : boolean
        whether the model is networked or not, will store figures in the appropriate Figures subdirectory
    """    
    

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
    # uncomment to generate own data
    #run_experiment("Data/output_networked.csv", networked=True)
    #run_experiment("Data/output_non_networked.csv", networked=False)
    
    plot_experiments("Data/output_networked.csv", networked=True)
    plot_experiments("Data/output_non_networked.csv", networked=False)