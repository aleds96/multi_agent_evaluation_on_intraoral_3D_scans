import numpy as np

def evaluate_agent(outputs):
    """
    Calcola metriche di valutazione per il single agent.
    outputs: lista di dict con:
        - scan
        - real_quality
        - pred_quality
        - motivation
    """

    real = np.array([o["real_quality"] for o in outputs])
    pred = np.array([o["pred_quality"] for o in outputs])

    mae = float(np.mean(np.abs(real - pred)))
    acc = float(np.mean(real == pred))

    return {
        "mae": mae,
        "accuracy": acc,
        "n_samples": len(outputs),
    }
