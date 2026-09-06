import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# utils_fun.py

import json
from pathlib import Path
from datetime import datetime

def save_experiment(config, outputs, evaluation, root="experiments"):
    """
    Salva configurazione, predizioni e valutazione in una cartella timestampata.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(root) / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    #salva config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    #salva outputs
    with open(exp_dir / "outputs.json", "w") as f:
        json.dump(outputs, f, indent=2)

    #salva valutazione
    with open(exp_dir / "evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)

    return str(exp_dir)

def compute_quantile_table(values, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    values = np.array(values)
    qvals = np.quantile(values, quantiles)
    return pd.DataFrame({
        "quantile": quantiles,
        "value": qvals
    })
def plot_quantile(values, title):
    #Ordina i valori
    sorted_vals = np.sort(values)

    #Normalizza l’asse X in [0,1]
    quantiles = np.linspace(0, 1, len(sorted_vals))

    plt.figure(figsize=(6,4))
    plt.plot(quantiles, sorted_vals, marker='o', markersize=3)
    plt.title(title)
    plt.xlabel("Quantile")
    plt.ylabel("Score")
    plt.grid(True)
    plt.show()

    #Esempio: quantile plot per categoria Mesial
    #plot_quantile(results["ap_per_scan"]["Mesial"], "Quantile plot – AP Mesial")