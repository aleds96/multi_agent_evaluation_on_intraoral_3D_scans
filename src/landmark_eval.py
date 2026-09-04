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
    #Costruisci GT e predizioni
    gt_all = build_gt_all(gt_root, categories)
    pred_all_map = build_pred_all_map(pred_csv, categories)

    all_scans = set()
    for cat in categories:
        all_scans.update(gt_all[cat].keys())
        all_scans.update(pred_all_map[cat].keys())
    all_scans = sorted(list(all_scans))

    #AP/AR per scan e categoria
    ap_per_scan = {cat: [] for cat in categories}
    ar_per_scan = {cat: [] for cat in categories}

    #Liste globali (media su categorie)
    mAP_per_scan = []
    mAR_per_scan = []

    for scan_name in all_scans:

        #Filtra GT e predizioni per questo scan
        gt_scan = {cat: {scan_name: gt_all[cat].get(scan_name, [])} for cat in categories}
        pred_scan = {cat: {scan_name: pred_all_map[cat].get(scan_name, [])} for cat in categories}

        #Categorie valide (quelle che hanno almeno un GT)
        valid_categories = [cat for cat in categories if len(gt_scan[cat][scan_name]) > 0]

        #Se nessuna categoria ha GT allora ignora lo scan
        if len(valid_categories) == 0:
            continue

        #Calcola AP/AR SOLO sulle categorie valide
        metrics = score_(
            {cat: gt_scan[cat] for cat in valid_categories},
            {cat: pred_scan[cat] for cat in valid_categories}
        )

        #Salva AP/AR per categoria
        for cat in valid_categories:
            ap_per_scan[cat].append(metrics["AP"][cat])
            ar_per_scan[cat].append(metrics["AR"][cat])

        #Calcola mAP/mAR globali per questo scan
        mAP_scan = np.mean([metrics["AP"][cat] for cat in valid_categories])
        mAR_scan = np.mean([metrics["AR"][cat] for cat in valid_categories])

        mAP_per_scan.append(mAP_scan)
        mAR_per_scan.append(mAR_scan)

    return {
        "scans": all_scans,
        "ap_per_scan": ap_per_scan,
        "ar_per_scan": ar_per_scan,
        "mAP_per_scan": mAP_per_scan,
        "mAR_per_scan": mAR_per_scan
    }
