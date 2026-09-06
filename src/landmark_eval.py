import json
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from pathlib import Path

#da 0 a==> 3 mm, step 0.1
THRESHOLDS = np.arange(0.0, 3.0 + 0.1, 0.1)  


def load_gt(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)
    return [
        {"class": obj["class"], "coord": np.array(obj["coord"])}
        for obj in data["objects"]
    ]


def load_pred(csv_path: Path, scan_name: str):
    df = pd.read_csv(csv_path, header=None,
                     names=["scan", "x", "y", "z", "class", "conf"])
    df = df[df["scan"] == scan_name]

    return [
        {"class": row["class"],
         "coord": np.array([row.x, row.y, row.z]),
         "conf": float(row.conf)}
        for _, row in df.iterrows()
    ]


def voc_ap(rec, prec):
    #append sentinel values
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    #precision envelope (smoothing)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    #points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    #area under curve
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
def voc_ar(dist_thresh_list, recall, keypoint_cat):
    #reverse arrays
    mrec = np.array(dist_thresh_list[::-1])
    mpre = np.array(recall[keypoint_cat][::-1])

    #sentinel values
    mrec = np.concatenate(([0.], mrec, [1.]))
    mpre = np.concatenate(([0.], mpre, [0.]))

    #recall envelope (same logic as precision envelope)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    #points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # area under curve
    ar = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ar
def eval_det_cls_map(pred, gt, dist_thresh, classname):
    class_recs = {}
    npos = 0

    #GT
    for mesh_name in gt.keys():
        keypoints = np.array(gt[mesh_name])  
        det = [False] * len(keypoints)
        npos += len(keypoints)
        class_recs[mesh_name] = {'kp': keypoints, 'det': det}

    #scans senza GT ma con pred
    for mesh_name in pred.keys():
        if mesh_name not in class_recs:
            class_recs[mesh_name] = {'kp': np.array([]), 'det': []}

    #flatten pred
    mesh_names = []
    confidence = []
    KP = []

    for mesh_name in pred.keys():
        for kp, score in pred[mesh_name]:
            mesh_names.append(mesh_name)
            confidence.append(score)
            KP.append(kp)

    confidence = np.array(confidence)
    KP = np.array(KP)

    sorted_ind = np.argsort(-confidence)
    KP = KP[sorted_ind]
    mesh_names = [mesh_names[i] for i in sorted_ind]

    nd = len(mesh_names)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[mesh_names[d]]
        kp = KP[d]
        KPGT = R['kp']

        if KPGT.size > 0:
            distance = np.linalg.norm(kp - KPGT, axis=1)
            dmin = np.min(distance)
            jmin = np.argmin(distance)
        else:
            dmin = np.inf

        if dmin < dist_thresh:
            if not R['det'][jmin]:
                tp[d] = 1.
                R['det'][jmin] = True
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = voc_ap(rec, prec)
    return rec, prec, ap

def eval_map(pred_all, gt_all, dist_thresh=0.1):
    rec = {}
    prec = {}
    ap = {}

    for classname in gt_all.keys():
        rec[classname], prec[classname], ap[classname] = eval_det_cls_map(
            pred_all[classname], gt_all[classname], dist_thresh, classname
        )

    return rec, prec, ap

def filter_scan(all_scans, scan_name):
    all_filtered = {}
    for category, data_dict in all_scans.items():
        all_filtered[category] = {}
        try:
            all_filtered[category][scan_name] = all_scans[category][scan_name]
        except:
            all_filtered[category][scan_name] = []
    return all_filtered
def score_(gt_all, pred_all_map):
    score_dict = {}
    dist_thresh_list = []
    recall = {cat: [] for cat in gt_all.keys()}

    for i in range(30):
        dist_thresh = 0.1 * i
        rec, prec, ap = eval_map(pred_all_map, gt_all, dist_thresh)
        score_dict[str(i)] = ap
        dist_thresh_list.append(dist_thresh)

        for cat in rec.keys():
            #Se rec[cat] è vuoto => recall = 0
            if len(rec[cat]) == 0:
                recall[cat].append(0.0)
            else:
                recall[cat].append(rec[cat][-1])

    #mAP per categoria
    class_values = {cat: [] for cat in gt_all.keys()}
    for ap_dict in score_dict.values():
        for cat in class_values.keys():
            class_values[cat].append(ap_dict[cat])

    mAP = {cat: float(np.mean(values)) for cat, values in class_values.items()}

    #mAR per categoria
    mar = {}
    dist_x = np.asarray(dist_thresh_list) 
    for cat in recall.keys():
        mar[cat] = float(voc_ar(dist_x, recall, cat))

    return {"AP": mAP, "AR": mar}


def build_gt_all(gt_root,categories):
    gt_all = {cat: {} for cat in categories}

    for gt_json in gt_root.glob("*__kpt.json"):
        scan = gt_json.stem.replace("__kpt", "").rstrip("_")
        gt = load_gt(gt_json) 

        for cat in categories:
            gt_all[cat].setdefault(scan, [])

        for g in gt:
            cls = g["class"]
            if cls in gt_all:
                coord = g["coord"]
                gt_all[cls][scan].append(coord)

    return gt_all
def build_pred_all_map(pred_csv, categories):
    pred_all_map = {cat: {} for cat in categories}

    df = pd.read_csv(pred_csv, header=None, names=[
        "scan", "coord_x", "coord_y", "coord_z", "class", "score"
    ])

    for _, row in df.iterrows():
        scan = row["scan"]
        cls  = row["class"]

        if cls not in pred_all_map:
            continue

        coord = np.array([row["coord_x"], row["coord_y"], row["coord_z"]])
        score = row["score"] if "score" in row else 1.0

        pred_all_map[cls].setdefault(scan, []).append([coord, score])

    return pred_all_map
def filter_single_scan(gt_all, pred_all_map, scan_name, categories):
    gt_scan = {cat: {scan_name: gt_all[cat].get(scan_name, [])} for cat in categories}
    pred_scan = {cat: {scan_name: pred_all_map[cat].get(scan_name, [])} for cat in categories}
    return gt_scan, pred_scan

def evaluate_dataset(gt_root, pred_csv, categories):
    gt_all = build_gt_all(gt_root, categories)          # usa g["coord"]
    pred_all_map = build_pred_all_map(pred_csv, categories)

    #filtra categorie richieste
    gt_all = {cat: gt_all[cat] for cat in categories}
    pred_all_map = {cat: pred_all_map[cat] for cat in categories}

    metrics = score_(gt_all, pred_all_map)

    results_global = {
        cat: {"AP": metrics["AP"][cat], "AR": metrics["AR"][cat]}
        for cat in categories
    }

    mAP_global = np.mean([metrics["AP"][cat] for cat in categories])
    mAR_global = np.mean([metrics["AR"][cat] for cat in categories])

    return results_global, mAP_global, mAR_global
def evaluate_single_scan(gt_all, pred_all_map, scan_name, categories):
    #filtra solo lo scan richiesto
    gt_scan, pred_scan = filter_single_scan(gt_all, pred_all_map, scan_name, categories)

    metrics = score_(gt_scan, pred_scan)

    #struttura identica a evaluate_dataset
    results_scan = {
        cat: {"AP": metrics["AP"][cat], "AR": metrics["AR"][cat]}
        for cat in categories
    }

    mAP_scan = np.mean([metrics["AP"][cat] for cat in categories])
    mAR_scan = np.mean([metrics["AR"][cat] for cat in categories])

    return results_scan, mAP_scan, mAR_scan
def evaluate_all_scans(gt_root, pred_csv, categories):
    gt_all = build_gt_all(gt_root, categories)
    pred_all_map = build_pred_all_map(pred_csv, categories)

    all_scans = set()
    for cat in categories:
        all_scans.update(gt_all[cat].keys())
        all_scans.update(pred_all_map[cat].keys())
    all_scans = sorted(list(all_scans))

    ap_per_scan = {cat: [] for cat in categories}
    ar_per_scan = {cat: [] for cat in categories}
    gt_count_per_scan = {cat: [] for cat in categories}

    mAP_per_scan = []
    mAR_per_scan = []

    for scan_name in all_scans:

        gt_scan = {cat: {scan_name: gt_all[cat].get(scan_name, [])} for cat in categories}
        pred_scan = {cat: {scan_name: pred_all_map[cat].get(scan_name, [])} for cat in categories}

        valid_categories = [cat for cat in categories if len(gt_scan[cat][scan_name]) > 0]

        if len(valid_categories) == 0:
            continue

        metrics = score_(
            {cat: gt_scan[cat] for cat in valid_categories},
            {cat: pred_scan[cat] for cat in valid_categories}
        )

        for cat in categories:
            # AP/AR = 0 se classe non valida
            if cat in valid_categories:
                ap_per_scan[cat].append(metrics["AP"][cat])
                ar_per_scan[cat].append(metrics["AR"][cat])
            else:
                ap_per_scan[cat].append(0.0)
                ar_per_scan[cat].append(0.0)

            # Conta GT
            gt_count_per_scan[cat].append(len(gt_scan[cat][scan_name]))

        mAP_scan = np.mean([metrics["AP"][cat] for cat in valid_categories])
        mAR_scan = np.mean([metrics["AR"][cat] for cat in valid_categories])

        mAP_per_scan.append(mAP_scan)
        mAR_per_scan.append(mAR_scan)

    return {
        "scans": all_scans,
        "ap_per_scan": ap_per_scan,
        "ar_per_scan": ar_per_scan,
        "gt_count_per_scan": gt_count_per_scan,
        "mAP_per_scan": mAP_per_scan,
        "mAR_per_scan": mAR_per_scan
    }
def compute_quantile_table(values, quantiles=None):
    if quantiles is None:
        quantiles = np.arange(0, 1.05, 0.05)
    return pd.DataFrame({
        "quantile": quantiles,
        "value": [np.quantile(values, q) for q in quantiles]
    })

def quality_from_map(map_value):
    if map_value < 0.55:
        return 1
    elif map_value < 0.65:
        return 2
    elif map_value < 0.69:
        return 3
    elif map_value < 0.73:
        return 4
    else:
        return 5
def compute_all_statistics(GT_ROOT, PRED_CSV, categories):
    results_global, mAP_global, mAR_global = evaluate_dataset(GT_ROOT, PRED_CSV, categories=categories)
    results = evaluate_all_scans(GT_ROOT, PRED_CSV, categories=categories)

    quantile_mAP = compute_quantile_table(results["mAP_per_scan"])

    return {
        "results": results,
        "quantile_mAP_over_scan": quantile_mAP,
        "global": {
            "mAP": mAP_global,
            "mAR": mAR_global,
            "per_class": results_global
        }
    }
def select_examples_by_quantile(results, quantile_table, quantiles, k=1, m=5):
    """
    Seleziona m scans attorno ai quantili specificati. Restituisce:
      1)primary_examples: dizionario {quantile_label: [k scans]}
      2)extra_examples:   dizionario {quantile_label: [m-k scans]}
    """
    scans = results["scans"]
    mAP = results["mAP_per_scan"]

    def closest_m_to(value, m):
        ordered = sorted(zip(scans, mAP), key=lambda x: abs(x[1] - value))
        return [s for s, _ in ordered[:m]]

    primary_examples = {}
    extra_examples = {}

    for label, q in quantiles.items():
        q_value = quantile_table.loc[quantile_table["quantile"] == q, "value"].item()
        pool = closest_m_to(q_value, m)

        primary_examples[label] = pool[:k]
        extra_examples[label] = pool[k:]

    return primary_examples, extra_examples
def compute_profiles_for_scans(results, scans, categories):
    
    #Restituisce una lista di profili per le scans fornite.
    return [
        quality_profile_for_scan(results, categories, scan_name=scan)
        for scan in scans
    ]
def quality_profile_for_scan(results, categories, scan_name):
    scan_idx = results["scans"].index(scan_name)
    #print('scan_idx', scan_idx,'scan_name', scan_name)
    mAP_scan = results["mAP_per_scan"][scan_idx]
    mAR_scan = results["mAR_per_scan"][scan_idx]

    quality_global = quality_from_map(mAP_scan)

    quality_per_class = {}
    raw_per_class = {}
    valid_class = {}

    for cat in categories:
        ap = results["ap_per_scan"][cat][scan_idx]
        gt_count = results["gt_count_per_scan"][cat][scan_idx]

        raw_per_class[cat] = f"{ap:.2f}"

        if gt_count == 0:
            quality_per_class[cat] = "N/A"
            valid_class[cat] = False
        else:
            quality_per_class[cat] = quality_from_map(ap)
            valid_class[cat] = True

    valid_scores = {cat: quality_per_class[cat] for cat in categories if valid_class[cat]}
    best_class = max(valid_scores, key=valid_scores.get)
    worst_class = min(valid_scores, key=valid_scores.get)

    return {
        "scan": scan_name,
        "mAP": f"{mAP_scan:.2f}",
        "mAR": f"{mAR_scan:.2f}",
        "quality_global": quality_global,
        "quality_per_class": quality_per_class,
        "raw_per_class": raw_per_class,
        "best_class": best_class,
        "worst_class": worst_class,
    }