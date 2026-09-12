import numpy as np
#Calcola metriche di valutazione per il single agent.
def evaluate_agent(outputs):
 
    real = np.array([o["real_quality"] for o in outputs])
    pred = np.array([o["pred_quality"] for o in outputs])

    mae = float(np.mean(np.abs(real - pred)))
    acc = float(np.mean(real == pred))

    return {
        "mae": mae,
        "accuracy": acc,
        "n_samples": len(outputs),
    }
