from model.model import EpsteinCivilViolence
from experiment_plotter import plot_experiments
import time

def run_experiment(output_path, networked):
    # using punctuated equilibrium settings
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
        max_iters=100,
        seed=42,  # Set a seed for reproducibility
        networked=networked
    )

    start_time = time.time()
    model.run_model()
    end_time = time.time()
    print(f"Model run time: {end_time - start_time:.2f} seconds")
    print(f"{(end_time - start_time )/model.max_iters:.2f} sec per iteration")

    model_out = model.datacollector.get_model_vars_dataframe()
    model_out.to_csv(output_path, index=False)

if __name__ == "__main__":
    run_experiment("Data/testing.csv", networked=True)
    plot_experiments("Data/testing.csv", networked=True)
    
    run_experiment("Data/testing.csv", networked=False)
    plot_experiments("Data/testing.csv", networked=False)