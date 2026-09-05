#!/usr/bin/env python
# coding: utf-8

# In[18]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[37]:


import sys
from pathlib import Path
import os

PROJECT_ROOT = Path.cwd().parent
print("PROJECT ROOT ==>", PROJECT_ROOT)

sys.path.append(str(PROJECT_ROOT))

from src.landmark_eval import evaluate_all_scans, evaluate_dataset, load_gt, load_pred
from src.pred_gt_viewer_viz import visualize_pred_and_gt,load_pred_landmarks
from src.mesh_fun import (
    load_mesh, load_landmarks, create_landmark_spheres, visualize
)
from src.utils_fun import compute_quantile_table

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import open3d as o3d


# In[25]:


LANDMARK_PALETTE = {
    "Mesial":        [1.0, 0.0, 0.0],   #rosso
    "Distal":        [0.0, 1.0, 0.0],   #verde
    "Cusp":          [0.0, 0.0, 1.0],   #blu
    "InnerPoint":    [1.0, 1.0, 0.0],   #giallo
    "OuterPoint":    [0.0, 1.0, 1.0],   #ciano
    "FacialPoint":    [1.0, 0.0, 1.0],   #magenta
}

CATEGORIES = list(LANDMARK_PALETTE.keys())
print("CATEGORIES ==>", CATEGORIES)


# In[21]:


DATA_ROOT = PROJECT_ROOT / "dataset"
SCANS = DATA_ROOT / "toothinstancenet_input"
GT_ROOT = SCANS
PRED_CSV = DATA_ROOT / "model_predictions" / "ynlab" / "predictions.csv"

print("SCANS:", SCANS)
print("GT_ROOT:", GT_ROOT)
print("PRED_CSV:", PRED_CSV)


# In[22]:


###check obj
objs = sorted(list(SCANS.glob("*.obj")))
print("Numero totale di OBJ:", len(objs))


# #### calcola mAP, mARP globale e per categoria

# In[24]:


results_global, mAP_global, mAR_global = evaluate_dataset(GT_ROOT, PRED_CSV, categories=CATEGORIES)

print("=== RISULTATI GLOBALI (ufficiali) ===")
for cat, vals in results_global.items():
    print(f"{cat}: AP={vals['AP']:.4f}, AR={vals['AR']:.4f}")

print("\nMETRICHE TOTALI:")
print(f"mAP = {mAP_global:.4f}")
print(f"mAR = {mAR_global:.4f}")


# #### calcola mAP, mARP per scan e per categoria

# In[ ]:


results = evaluate_all_scans(GT_ROOT, PRED_CSV, categories=CATEGORIES)
print('***results***', results.keys())
#Boxplot per categoria (AP)
for cat in CATEGORIES:
    plt.figure()
    plt.boxplot(results["ap_per_scan"][cat])
    plt.title(f"AP distribution for {cat}")
    plt.show()

#Boxplot globale (mAP) (mediamo rispetto a ogni scan (ap x categoria) e poi facciamo la media per scan)
plt.figure()
plt.boxplot(results["mAP_per_scan"])
plt.title("Global mAP distribution across scans")
plt.show()


# #### distribuzione map, mar via quantili

# In[ ]:


quantile_mAP = compute_quantile_table(results["mAP_per_scan"])
quantile_mAR = compute_quantile_table(results["mAR_per_scan"])

print("\n=== Quantile table global mAP per scan ===")
print(quantile_mAP)

print("\n=== Quantile table global mAR per scan ===")
print(quantile_mAR)


# In[27]:


#### come sopra, ma per categoria (AP e AR)
quantile_tables_ap = {}
quantile_tables_ar = {}

for cat in CATEGORIES:
    vals_ap = results["ap_per_scan"][cat]
    vals_ar = results["ar_per_scan"][cat]

    if len(vals_ap) > 0:
        quantile_tables_ap[cat] = compute_quantile_table(vals_ap)

    if len(vals_ar) > 0:
        quantile_tables_ar[cat] = compute_quantile_table(vals_ar)
for cat in CATEGORIES:
    print(f"\n=== Quantile table AP => {cat} ===")
    print(quantile_tables_ap[cat])

    print(f"\n=== Quantile table AR => {cat} ===")
    print(quantile_tables_ar[cat])


# ### Top & worst k scan per categoria e globalmente. Da valutare visiviamente successivamente 

# In[53]:


def top_k_scans_per_category(results, categories, k=5):
    top_ap = {}
    top_ar = {}

    for cat in categories:
        values_ap = results["ap_per_scan"][cat]
        values_ar = results["ar_per_scan"][cat]
        scans = results["scans"]

        ap_pairs = list(zip(scans, values_ap))
        ar_pairs = list(zip(scans, values_ar))

        ap_sorted = sorted(ap_pairs, key=lambda x: x[1], reverse=True)
        ar_sorted = sorted(ar_pairs, key=lambda x: x[1], reverse=True)

        top_ap[cat] = ap_sorted[:k]
        top_ar[cat] = ar_sorted[:k]

    return top_ap, top_ar

def worst_k_scans_per_category(results, categories, k=5):
    worst_ap = {}
    worst_ar = {}

    for cat in categories:
        values_ap = results["ap_per_scan"][cat]
        values_ar = results["ar_per_scan"][cat]
        scans = results["scans"]

        ap_pairs = list(zip(scans, values_ap))
        ar_pairs = list(zip(scans, values_ar))

        ap_sorted = sorted(ap_pairs, key=lambda x: x[1])
        ar_sorted = sorted(ar_pairs, key=lambda x: x[1])

        worst_ap[cat] = ap_sorted[:k]
        worst_ar[cat] = ar_sorted[:k]

    return worst_ap, worst_ar

def top_k_global(results, k=5):
    scans = results["scans"]
    mAP = results["mAP_per_scan"]
    mAR = results["mAR_per_scan"]

    ap_pairs = list(zip(scans, mAP))
    ar_pairs = list(zip(scans, mAR))

    ap_sorted = sorted(ap_pairs, key=lambda x: x[1], reverse=True)
    ar_sorted = sorted(ar_pairs, key=lambda x: x[1], reverse=True)

    return ap_sorted[:k], ar_sorted[:k]


def worst_k_global(results, k=5):
    scans = results["scans"]
    mAP = results["mAP_per_scan"]
    mAR = results["mAR_per_scan"]

    ap_pairs = list(zip(scans, mAP))
    ar_pairs = list(zip(scans, mAR))

    ap_sorted = sorted(ap_pairs, key=lambda x: x[1])
    ar_sorted = sorted(ar_pairs, key=lambda x: x[1])

    return ap_sorted[:k], ar_sorted[:k]


top_ap_cat, top_ar_cat = top_k_scans_per_category(results, CATEGORIES, k=2)
worst_ap_cat, worst_ar_cat = worst_k_scans_per_category(results, CATEGORIES, k=2)

top_global_ap, top_global_ar = top_k_global(results, k=2)
worst_global_ap, worst_global_ar = worst_k_global(results, k=10)


# In[54]:


worst_global_ap


# ### analisi visuale predizioni vs ground truth

# In[34]:


import pandas as pd
predicted_data = pd.read_csv(PRED_CSV, header=None, names=[
    "scan", "coord_x", "coord_y", "coord_z", "class", "score"
])
print(predicted_data.columns)


# In[35]:


predicted_data.head(10)


# ##### Visualizzazione interattiva (mesh singolo specificato + landmark GT)

# In[56]:


case = "AKBDPB4C"
case="JXVWXY0L"
arc = "upper"
scan_name = f"{case}_{arc}"

mesh_path = SCANS / f"{scan_name}.obj"
gt_json = GT_ROOT / f"{scan_name}__kpt.json"

coords_gt, classes_gt = load_landmarks(gt_json)
mesh = load_mesh(mesh_path)
spheres_gt = create_landmark_spheres(coords_gt, classes_gt, LANDMARK_PALETTE)

visualize(mesh, spheres_gt, view="front")


# ##### Visualizzazione interattiva (mesh singolo specificato + predizioni )

# In[ ]:


#Visualizzazione predizioni (solo predetti)(per debugging))
coords_pred, classes_pred = load_pred_landmarks(PRED_CSV, scan_name)
mesh = load_mesh(mesh_path)
spheres_pred = create_landmark_spheres(coords_pred, classes_pred, LANDMARK_PALETTE)

visualize(mesh, spheres_pred, view="front")


# In[57]:


#Visualizzazione interattiva (mesh singolo specificato +  GT + predizioni insieme (per debugging))
mesh = load_mesh(mesh_path)

spheres_gt = create_landmark_spheres(coords_gt, classes_gt, LANDMARK_PALETTE)
spheres_pred = create_landmark_spheres(coords_pred, classes_pred, LANDMARK_PALETTE)

visualize(mesh, spheres_gt + spheres_pred, view="front")


# #### Visualizzazione predizioni + GT (singole & overlay) + screenshot per modello(come Figure 11 del paper)
# 

# In[60]:


#Visualizzazione predizioni + GT + screenshot (come Figure 11 del paper) [per modello]
print('scan_name', scan_name)
out_dir = PROJECT_ROOT / "screenshots_test" / scan_name
visualize_pred_and_gt(mesh_path, PRED_CSV, gt_json, scan_name, out_dir, LANDMARK_PALETTE)

