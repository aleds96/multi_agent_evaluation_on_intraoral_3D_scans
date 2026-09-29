import json
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from pathlib import Path
import random
from src.utils_fun import build_perturbation_label
from src.mesh_fun import (
    load_segmentation,load_pred_landmarks,load_mesh,
    color_mesh_by_groups,create_landmark_spheres,
    load_gt_landmarks,save_screenshot
)
from src.mesh_fun import load_gt_landmarks, load_pred_landmarks,compute_group_class_counts_for_scan
from src.var_constants import TOOTH_GROUP_PALETTE,TOOTH_TO_GROUP,CATEGORIES,LANDMARK_PALETTE
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
def extract_pred_scan(
    pred_all_map,
    scan_name,
):
    coords_pred = []
    classes_pred = []

    for cls, scans in pred_all_map.items():

        for coord_score in scans.get(scan_name, []):

            coord = coord_score[0]

            coords_pred.append(
                np.asarray(coord)
            )

            classes_pred.append(cls)

    return coords_pred, classes_pred
def extract_gt_scan(
    gt_all,
    scan_name,
):
    coords_gt = []
    classes_gt = []

    for cls, scans in gt_all.items():
        for coord in scans.get(scan_name, []):
            coords_gt.append(
                np.asarray(coord)
            )
            classes_gt.append(cls)
    return coords_gt, classes_gt
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

def evaluate_all_scans(
    gt_root,
    pred_csv,
    categories,
):
    gt_all = build_gt_all( gt_root, categories,)

    pred_all_map = build_pred_all_map( pred_csv, categories,)
    all_scans = set()
    for cat in categories:
        all_scans.update( gt_all[cat].keys() )
        all_scans.update(pred_all_map[cat].keys()   )

    all_scans = sorted(all_scans)
    results = []
    
    for scan_name in all_scans:
        coords_gt, classes_gt = (
            extract_gt_scan( gt_all,scan_name))
        coords_pred, classes_pred = (
            extract_pred_scan( pred_all_map,scan_name))
        try:
            metrics = (
                evaluate_single_scan_from_landmarks(
                    scan_name=scan_name,
                    coords_pred=coords_pred,
                    classes_pred=classes_pred,
                    coords_gt=coords_gt,
                    classes_gt=classes_gt,
                    categories=categories)
            )
        except ValueError:
            continue
        results.append(metrics)
    return results
#Valuta una singola scansione partendo direttamente
#da landmark GT e landmark predetti
def evaluate_single_scan_from_landmarks(
    scan_name,
    coords_pred,
    classes_pred,
    coords_gt,
    classes_gt,
    categories,
):
    gt_scan = {cat: {scan_name: []}for cat in categories}
    pred_scan = {cat: {scan_name: []}for cat in categories }
    pred_count_per_class = {cat: 0 for cat in categories}

    #GT
    for coord, cls in zip(coords_gt,classes_gt,
    ):
        if cls not in gt_scan:
            continue
        gt_scan[cls][scan_name].append(coord)
    #Predizioni
    for coord, cls in zip(coords_pred,classes_pred):
        if cls not in pred_scan:
            continue
        pred_scan[cls][scan_name].append([coord, 1.0])
        pred_count_per_class[cls] += 1
    valid_categories = [cat for cat in categories if len(gt_scan[cat][scan_name]) > 0]
    if len(valid_categories) == 0:
        raise ValueError(f"Nessuna categoria valida per scan {scan_name}")
    metrics = score_({cat: gt_scan[cat] for cat in valid_categories},
                     {    cat: pred_scan[cat] for cat in valid_categories})
    ap_per_class = {}
    ar_per_class = {}
    gt_count_per_class = {}
    for cat in categories:
        if cat in valid_categories:
            ap_per_class[cat] = ( metrics["AP"][cat])
            ar_per_class[cat] = (  metrics["AR"][cat])
            gt_count_per_class[cat] = len(gt_scan[cat][scan_name] )
        else:
            ap_per_class[cat] = 0.0
            ar_per_class[cat] = 0.0
            gt_count_per_class[cat] = 0

    mAP_scan = np.mean([
        metrics["AP"][cat]
        for cat in valid_categories
    ])

    mAR_scan = np.mean([
        metrics["AR"][cat]
        for cat in valid_categories
    ])
    return {
        "scan": scan_name,
        "ap_per_class": ap_per_class,
        "ar_per_class": ar_per_class,
        "gt_count_per_class": gt_count_per_class,
        "pred_count_per_class":pred_count_per_class,
        "mAP": float(mAP_scan),
        "mAR": float(mAR_scan),
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

    quantile_mAP = compute_quantile_table([scan_["mAP"] for scan_ in results])

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
    scans = [ scan['scan'] for scan in results]
    mAP = [scan['mAP'] for scan in results]

    def closest_m_to(value, m):
        ordered = sorted(zip(scans, mAP), key=lambda x: abs(x[1] - value))
        return [s for s, _ in ordered[:m]]

    primary_examples = {}
    extra_examples = {}

    for label, q in quantiles.items():
        q_value = quantile_table.loc[quantile_table["quantile"] == q, "value"].item()
        pool = closest_m_to(q_value, m)

        primary_examples[label] = pool[:k]
        if float(q)>=0.5:
            extra_examples[label] = pool[k:]
    return primary_examples, extra_examples
def compute_profiles_for_scans(results, scans, categories):
    
    #Restituisce una lista di profili per le scans fornite.
    return [
        quality_profile_for_scan(results, categories, scan_name=scan)
        for scan in scans
    ]
def quality_profile_for_scan( results, categories, scan_name):
    scan_metrics = next((item for item in results if item["scan"] == scan_name ), None,)
    if scan_metrics is None:
        raise ValueError( f"Scan {scan_name} non trovata nei risultati.")

    mAP_scan = scan_metrics["mAP"]
    mAR_scan = scan_metrics["mAR"]

    quality_global = quality_from_map( mAP_scan)
    quality_per_class = {}
    raw_per_class = {}
    valid_class = {}

    for cat in categories:
        ap = scan_metrics["ap_per_class"][cat]
        gt_count = scan_metrics[  "gt_count_per_class" ][cat]
        raw_per_class[cat] = f"{ap:.2f}"
        if gt_count == 0:
            quality_per_class[cat] = "N/A"
            valid_class[cat] = False
        else:
            quality_per_class[cat] = (      quality_from_map(ap) )
            valid_class[cat] = True
    valid_scores = { cat: quality_per_class[cat] for cat in categories if valid_class[cat]}
    best_class = max( valid_scores,  key=valid_scores.get)

    worst_class = min( valid_scores,   key=valid_scores.get)

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
#Restituisce un dizionario {classe -> count} per le 6 classi di landmark.

def compute_class_counts_for_scan(PRED_CSV: Path, scan_name: str, CATEGORIES: list):
   
    coords_pred, classes_pred = load_pred_landmarks(PRED_CSV, scan_name)
    counts = {cat: 0 for cat in CATEGORIES}

    for cls in classes_pred:
        if cls in counts:
            counts[cls] += 1

    return counts

def build_input_dataset(results, exclude_scans, SCANS, GT_ROOT, PRED_CSV,SCREENSHOT_ROOT,CATEGORIES):
    
    input_scans = [s for s in results if s['scan'] not in exclude_scans]
    dataset = []
    for scan in input_scans:
        scan_name=scan['scan']
        #print('### scan to check=>',scan_name)
        profile = quality_profile_for_scan(results, CATEGORIES, scan_name)
        #print('###### profile==>',profile)
        coords_pred, classes_pred = load_pred_landmarks(PRED_CSV, scan_name)
        #print('### available keys==>',scan.keys()) 
        pred_count_per_class = scan['pred_count_per_class']
        gt_count_per_class = scan['gt_count_per_class']
        #count_per_class = compute_class_counts_for_scan(PRED_CSV, scan, CATEGORIES)
        mesh_path = SCANS / f"{scan_name}.obj"
        seg_path  = SCANS / f"{scan_name}_seg.json"

        count_per_group_class, count_per_group = compute_group_class_counts_for_scan(
            mesh_path,
            seg_path,
            PRED_CSV,
            scan_name,
            CATEGORIES,
            TOOTH_TO_GROUP
        )

        dataset.append({
            "scan": scan_name,
            "profile": profile,
            "mesh": str(SCANS / f"{scan_name}.obj"),
            "gt": str(GT_ROOT / f"{scan_name}__kpt.json"),
            "seg": str(SCANS / f"{scan_name}_seg.json"),
            "image_pred": str(SCREENSHOT_ROOT / f"{scan_name}/predicted.png"),
            "pred_coords": coords_pred,
            "pred_classes": classes_pred,
            "gt_count_per_class":gt_count_per_class,
            "pred_count_per_class": pred_count_per_class,
            "pred_count_per_group_class": count_per_group_class,
            "pred_count_per_group": count_per_group,
        })

    return dataset
#Rimuove una percentuale di landmark
def apply_missing_landmark_perturbation(
    coords,
    classes,
    fraction,
    #specifica opzionalmente un sotto-insieme di classi da considerare. Es. Mesial
    target_landmark_classes=[],
    seed=42,
):
    rng = random.Random(seed)
    if len(coords) == 0:
        return coords.copy(), classes.copy()
    if len(target_landmark_classes)==0:
        candidate_idx = list(range(len(coords)))
    else:
        candidate_idx = [i for i, cls in enumerate(classes) if cls in target_landmark_classes]
    if len(candidate_idx) == 0:
        return ( coords.copy(),classes.copy() )
    n_remove = max( 1, int(len(candidate_idx) * fraction))

    remove_idx = set(rng.sample(  candidate_idx, min(    n_remove,len(candidate_idx))) )
    keep_idx = [i for i in range(len(coords)) if i not in remove_idx]
    coords_new = coords[keep_idx]
    classes_new = [classes[i] for i in keep_idx]
    return (
        coords_new,
        classes_new,
    )
#fai il flip della classe di un landmark
def apply_class_flip_perturbation(
    coords,
    classes,
    fraction,
    target_landmark_classes=[],
    seed=42,
):
    MESH_CLASSES = list(set(classes))
    rng = random.Random(seed)
    if len(classes) == 0:
        return coords.copy(), classes.copy()

    #GLOBAL FLIP==> considera tutti i punti
    if len(target_landmark_classes) == 0:
        candidate_idx = list(range(len(classes)))
        #print('### global flip with len(candidate)==>',len(candidate_idx))
    #LOCALIZED FLIP=> solo specifica classe target
    else:
        candidate_idx = [ i for i, cls in enumerate(classes) if cls in target_landmark_classes]

    if len(candidate_idx) == 0:
        return (coords.copy(),classes.copy())
    n_flip = max(1,int(len(candidate_idx) * fraction))
    #print('## num flip==>',n_flip)
    flip_idx = rng.sample(candidate_idx,min(n_flip, len(candidate_idx))
                          )
    classes_new = classes.copy()
    for idx in flip_idx:
        current_class = classes_new[idx]
        candidate_classes = [ c for c in MESH_CLASSES if c != current_class]
        if len(candidate_classes) == 0:
            continue
        new_class = rng.choice(candidate_classes)
        classes_new[idx] = new_class

    return (coords.copy(),classes_new)
def apply_perturbation(perturbation_type,coords_pred,classes_pred,perturbation_params):
    if perturbation_type == "missing_landmarks":
        return apply_missing_landmark_perturbation(
            coords_pred,
            classes_pred,
            fraction=perturbation_params["fraction"],
            target_landmark_classes=perturbation_params["target_landmark_classes"],
        )
    elif perturbation_type == "class_flip":
            return apply_class_flip_perturbation(
                coords_pred,
                classes_pred,
                fraction=perturbation_params["fraction"],
                target_landmark_classes=perturbation_params["target_landmark_classes"],
            )
    return coords_pred,classes_pred
def build_synthetic_oracle_sample(
    scan_name,
    mesh_path,
    gt_path,
    seg_path,
    pred_landmark_path,
    screenshot_dir,
    perturbation_type,
    perturbation_params,
    pertubation_metadata,
    prefix=''
):

    pert_scan_name = f"{scan_name}_{perturbation_type}_{build_perturbation_label(perturbation_params)}"
    tooth_seg_labels = load_segmentation(seg_path)
    mesh = load_mesh(mesh_path)
    mesh = color_mesh_by_groups(mesh, tooth_seg_labels, TOOTH_TO_GROUP, TOOTH_GROUP_PALETTE) 

    coords_pred, classes_pred = load_pred_landmarks(pred_landmark_path, scan_name)
    #print('## len class pred',len(classes_pred),'###shape coords pred==>',coords_pred.shape)
    #print('### pred landmark path ==>')
    coords_gt, classes_gt = load_gt_landmarks(gt_path)

    #applica pertubazione richista
    coords_after_pert,classes_after_pert=apply_perturbation(perturbation_type,coords_pred,classes_pred,perturbation_params)
    #print('pert scan name==>',pert_scan_name)
    #print("original:", len(classes_pred))
    #print("noisy:", len(classes_after_pert))
    #valuta scan e ottieni profilo qualità
    scan_evaluation_pred=evaluate_single_scan_from_landmarks(scan_name=scan_name,
                                coords_pred=coords_after_pert,
                                classes_pred=classes_after_pert,
                                coords_gt=coords_gt,
                                classes_gt=classes_gt,categories=CATEGORIES)
    #print('####### scan eval pred==>\n',scan_evaluation_pred)
    scan_quality_profile_pred=quality_profile_for_scan([scan_evaluation_pred],categories=CATEGORIES,scan_name=scan_name)
    #crea screenshot 
    spheres_pred_after_pert = create_landmark_spheres(coords_after_pert, classes_after_pert, LANDMARK_PALETTE)
    image_pred_after_pert_path = screenshot_dir / scan_name /f"{pert_scan_name}.png"
    if image_pred_after_pert_path.exists()==False:
        save_screenshot(mesh, spheres_pred_after_pert, image_pred_after_pert_path, zoom=0.65)


    obs_features ={
                f'{prefix}_scan_conf_name': pert_scan_name,
                f'{prefix}_count_per_class':scan_evaluation_pred.get('pred_count_per_class'), 
                f'{prefix}_quality_score':scan_quality_profile_pred.get('quality_global'),
                f'{prefix}_gmap': scan_evaluation_pred.get('mAP'),
                f'{prefix}_quality_per_class': scan_quality_profile_pred.get('quality_per_class'), 
                f'{prefix}_metadata': pertubation_metadata.get('description',''),
                f'{prefix}_image_path': image_pred_after_pert_path
            }
    return obs_features
#qui potrei aggiungere anche un max di campioni che hanno specifica qualità (es. quality=3 o quality=4) prima di passare
#ad un'altra configurazione
def build_synthetic_oracle_dataset(scan_names, SCANS, GT_ROOT, PRED_CSV,SCREENSHOT_DIR,pertubation_configs):
    dataset = []
    for scan_name in scan_names: 
        mesh_path = SCANS / f"{scan_name}.obj"
        gt_path = GT_ROOT / f"{scan_name}__kpt.json"
        seg_path  =SCANS / f"{scan_name}_seg.json"

        #memorizziamo le informazioni basilari sullinput sul quale viene applicata la pertubazione.
        #in modo da fornire all'agente contesto maggiore sul dato originale e non solo sulla versione pertubata
        pred_features_before_pert=build_synthetic_oracle_sample(scan_name,
                                    mesh_path,gt_path,seg_path,
                                    PRED_CSV,SCREENSHOT_DIR,
                                    perturbation_type='',
                                    perturbation_params={},
                                    pertubation_metadata={},
                                    prefix='origin_pred')
        #print('##### PRED INFO BEFORE PERTUBATION==>\n',pred_features_before_pert)

        #applica tutte le possibili pertubazioni per generare k dati sintetici
        for pert_cfg in pertubation_configs: 
            pert_features=build_synthetic_oracle_sample(scan_name,
                                                mesh_path,gt_path,seg_path,
                                                PRED_CSV,SCREENSHOT_DIR,
                                                perturbation_type=pert_cfg.get('type'),
                                                perturbation_params=pert_cfg.get('params'),
                                                pertubation_metadata=pert_cfg.get('metadata'),
                                                prefix='pred')
            
            dataset.append({
            "scan": scan_name,
            **pred_features_before_pert, 
            **pert_features,
            "mesh_path": mesh_path,
            "gt_path": gt_path,
            "seg_path": seg_path,
        })
        return dataset

def build_missing_landmark_configs(
        global_config={'frac':[0.05,0.1]},
        local_config={'cls':["Mesial",
        "Distal",
        "InnerPoint",
        "OuterPoint"], 'frac':[0.3,0.6]}):
    configs = []

    for frac in global_config.get('frac'):
        configs.append({
            "type": "missing_landmarks",
            "params": {
                "target_landmark_classes":[],
                "fraction": frac},
            "metadata": {
                "description":
                    f"rimosso {int(frac*100)}% landmarks globalmente" }
        })

    #configurazioni rispetto ad una classe target
    for cls in local_config.get('cls'):
        for frac in local_config.get('frac'):
            configs.append({
                "type": "missing_landmarks",
                "params": {
                    "fraction": frac, 
                    "target_landmark_classes": [] if cls == 'any class' else  [cls] },
                "metadata": {
                    "description":
                        f"rimosso {int(frac*100)}% {cls}" }
            })
    return configs
def build_class_flip_configs(
    global_config={
        'frac': [0.05]
    },
    local_config={
        'cls': [
            "Mesial",
            "Distal",
            "InnerPoint",
            "OuterPoint",
        ],
        'frac': [0.20, 0.30, 0.50] 
    }):
    configs = []

    #GLOBAL FLIP
    for frac in global_config.get('frac', []):
        configs.append({
            "type": "class_flip",
            "params": {
                "target_landmark_classes": [],
                "fraction": frac,
            },
            "metadata": {
                "description":
                    f"flip classe {int(frac * 100)}% landmarks globalmente"
            },
        })

    #flip locale
    for cls in local_config.get('cls', []):
        for frac in local_config.get('frac', []):
            configs.append({
                "type": "class_flip",
                "params": {
                    "fraction": frac,
                    "target_landmark_classes":
                        [] if cls == 'any class' else [cls],
                },
                "metadata": {
                    "description":f"flip {int(frac * 100)}% dei landmarks con classe {cls}"}})
    return configs
def build_landmark_shift_configs(
    global_config={
        "frac": [0.20,0.40],
        "shift_mm": [0.5,1.0, 2.0, 3.0],
    },
    local_config={
        "cls": [
            "Mesial",
            "OuterPoint",
        ],
        "frac": [0.20, 0.30, 0.50],
        "shift_mm": [1.0, 2.0, 3.0],
    }
):

    configs = []
    #GLOBAL SHIFT
    for frac in global_config.get("frac", []):

        for shift_mm in global_config.get("shift_mm", []):
            configs.append({
                "type": "landmark_shift",
                "params": {
                    "fraction": frac,
                    "shift_mm": shift_mm,
                    "target_landmark_classes": [],
                },
                "metadata": {
                    "description":
                        f"spostato {int(frac * 100)}% landmarks globalmente di {shift_mm}mm"
                },
            })
    #LOCALIZED SHIFT
    for cls in local_config.get("cls", []):
        for frac in local_config.get("frac", []):
            for shift_mm in local_config.get("shift_mm", []):
                configs.append({
                    "type": "landmark_shift",
                    "params": {
                        "fraction": frac,
                        "shift_mm": shift_mm,
                        "target_landmark_classes":
                            [] if cls == "any class" else [cls],
                    },
                    "metadata": {
                        "description":
                            f"spostato {int(frac * 100)}% landmarks {cls} di {shift_mm}mm"
                    },
                })

    return configs