import numpy as np
import pandas as pd
import os

def generate_fgts_data():
    T = 500
    d = 10
    K = 5
    output_dir = "data/fgts_exp_1"
    
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((T, d))

    weights = rng.uniform(-1, 1, (d, K))
    Y = X @ weights + rng.normal(0, 0.1, (T, K))

    pd.DataFrame(X).to_csv(os.path.join(output_dir, "X.csv"), index=False)
    pd.DataFrame(Y).to_csv(os.path.join(output_dir, "Y.csv"), index=False)

if __name__ == "__main__":
    generate_fgts_data()