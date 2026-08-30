import open3d as o3d
import numpy as np
import pandas as pd
import json
from pathlib import Path

from src.mesh_fun import load_mesh, create_landmark_spheres,save_screenshot

def load_gt_landmarks(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)
    coords = np.array([obj["coord"] for obj in data["objects"]])
    classes = [obj["class"] for obj in data["objects"]]
    return coords, classes

def load_pred_landmarks(csv_path: Path, scan_name: str):
    df = pd.read_csv(csv_path, header=None,
                     names=["scan", "x", "y", "z", "class", "conf"])
    df = df[df["scan"] == scan_name]
    coords = df[["x", "y", "z"]].values
    classes = df["class"].tolist()
    return coords, classes

def visualize_pred_and_gt(mesh_path, pred_csv, gt_json, scan_name, out_dir,landmark_palette):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Carica mesh
    mesh = load_mesh(mesh_path)

    # Carica predizioni
    pred_coords, pred_classes = load_pred_landmarks(pred_csv, scan_name)
    spheres_pred = create_landmark_spheres(pred_coords, pred_classes, landmark_palette)

    # Carica GT
    gt_coords, gt_classes = load_gt_landmarks(gt_json)
    spheres_gt = create_landmark_spheres(gt_coords, gt_classes, landmark_palette)

    # Screenshot predizioni
    save_screenshot(mesh, spheres_pred, out_dir / "predicted.png")

    # Screenshot GT
    save_screenshot(mesh, spheres_gt, out_dir / "ground_truth.png")

    # Screenshot overlay
    save_screenshot(mesh, spheres_pred + spheres_gt, out_dir / "overlay.png")

