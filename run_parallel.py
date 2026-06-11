import os
import subprocess
from dask.distributed import Client, LocalCluster


def run_simulation(config_file, output_file):
    # Copy current environment
    env = os.environ.copy()

    # Limit threading for the simulation
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"

    print(f"Running {config_file} → {output_file}")

    subprocess.run(
        ["python", "run.py", "--props", config_file, "--output", output_file],
        env=env,
        check=True
    )

    print(f"Finished {config_file}")


if __name__ == "__main__":
    # Create Dask cluster
    cluster = LocalCluster(
        n_workers=4,          # 4 simulations in parallel
        threads_per_worker=1, # IMPORTANT: avoid nested threading
        processes=True
    )

    client = Client(cluster)

    print(client)

    # Define jobs
    configs = [
        #"properties.json",
        #"properties.json",
        #"properties.json",
        "properties.json"
    ]

    outputs = [
        #"results1.nc",
        #"results2.nc",
        #"results3.nc",
        "results_2t.nc"
    ]

    # Submit tasks
    futures = [
        client.submit(run_simulation, cfg, out)
        for cfg, out in zip(configs, outputs)
    ]

    # Wait for completion
    client.gather(futures)

    print("All simulations finished.")