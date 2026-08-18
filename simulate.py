import os
import pandas as pd
import numpy as np
from datetime import datetime


""" 
sim.py

Purpose:
    Stand in for a real underwriting intake feed. Since there is no live
    policyholder API for this project, this script fabricates new applicant
    rows and drops them as CSV batches into a landing folder that the
    Airflow DAG's extract task will poll.
Usage:
    Run manually multiple times to simulate several arrivals before testing
    the DAG's extract task, or call generate_batch() directly from a
    notebook cell for quick one-off testing.
"""

sim_data = "data/sim_data"

os.makedirs(sim_data, exist_ok=True)

def generate_batch(n=5, seed=None):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(18, 65, n),
        "sex": rng.choice(["male", "female"], n),
        "bmi": rng.normal(30.6, 6.1, n).round(2),
        "children": rng.integers(0, 6, n),
        "smoker": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
        "region": rng.choice(["southwest", "southeast", "northwest", "northeast"], n),
    })

if __name__ == "__main__":
    batch = generate_batch(n=5)
    fname = f"{sim_data}/batch_{datetime.now():%Y%m%d_%H%M%S}.csv"
    batch.to_csv(fname, index=False)
    print(f"wrote {fname}")