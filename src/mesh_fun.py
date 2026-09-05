import json
import numpy as np
import open3d as o3d
import pymeshlab
import sys
from pathlib import Path
import os
def load_unsorted_mesh(mesh_path: Path):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    return mesh
def load_mesh(mesh_path: Path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))

    vertices = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix()

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    mesh.compute_vertex_normals()
    return mesh
def load_segmentation(seg_path: Path):
    with open(seg_path, "r") as f:
        data = json.load(f)
    return np.array(data["labels"])
def load_landmarks(kpt_path: Path):
    with open(kpt_path, "r") as f:
        data = json.load(f)
    coords = np.array([obj["coord"] for obj in data["objects"]])
    classes = [obj["class"] for obj in data["objects"]]
    return coords, classes
def color_mesh_by_labels(mesh, labels, FDI_palette):
    colors = np.array([FDI_palette.get(int(l), [255,255,255]) for l in labels]) / 255.0
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh
def color_mesh_by_groups(mesh, labels, tooth_to_group, group_palette):
    #mesh = mesh.clone()

    colors = []
    for lbl in labels:
        group = tooth_to_group.get(int(lbl), "gingiva")
        color = group_palette[group]
        colors.append(color)

    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))
    return mesh
def create_landmark_spheres(coords, classes, landmark_palette, radius=0.45):
    spheres = []
    for coord, cls in zip(coords, classes):

        # geometria più liscia
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
        sphere.translate(coord)

        # colore base (invariato)
        color = np.array(landmark_palette.get(cls, [1.0, 1.0, 1.0]))

        # leggero scurimento → effetto bordo percettivo
        color = color * 0.9

        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()

        spheres.append(sphere)

    return spheres

def visualize_landmarks(mesh_path: Path, coords, classes,landmark_palette):
    mesh = load_mesh(mesh_path)
    spheres = create_landmark_spheres(coords, classes, landmark_palette)
    o3d.visualization.draw_geometries([mesh] + spheres)

def apply_model_config_view(vis, width, height):
    vc = vis.get_view_control()

    front  = [0.0, 0.0, 1.0]
    up     = [-0.095852611932817064, 0.99539553785701518, 0.0]
    lookat = [1.2521453313041158, 1.8914098504878176, -101.39684191000001]
    zoom   = 0.75

   
    corrected_zoom = zoom 

    vc.set_front(front)
    vc.set_up(up)
    vc.set_lookat(lookat)
    vc.set_zoom(corrected_zoom)


def save_screenshot(mesh, spheres, out_path, view="front", width=1600, height=1600):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=width, height=height)  

    vis.add_geometry(mesh)
    for s in spheres:
        vis.add_geometry(s)

    vis.poll_events()
    vis.update_renderer()

    #set_view(vis, view)
    apply_model_config_view(vis,width=width, height=height)

    # Second poll/update is REQUIRED for camera update
    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(str(out_path))
    vis.destroy_window()

    print(f"Screenshot salvato in: {out_path}")

def visualize(mesh, spheres, view="front", width=1200, height=1200):
    """
    Visualizza mesh + landmark in una finestra Open3D interattiva,
    impostando la vista desiderata.

    Parametri:
    - mesh: TriangleMesh Open3D
    - spheres: lista di landmark (TriangleMesh)
    - view: "front", "top", "bottom", "left", "right"
    - width, height: dimensioni della finestra
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=True)

    vis.add_geometry(mesh)
    for s in spheres:
        vis.add_geometry(s)

    vis.poll_events()
    vis.update_renderer()
    #apply_model_config_view(vis)
    vis.poll_events()
    vis.update_renderer()   
    #set_default_open3d_view(vis, mesh)
    # Imposta la vista usando la tua funzione
    #set_view(vis, view)
    #apply_tooth_config_view(vis)
    vis.run()
    vis.destroy_window()

def check_alignment(mesh, labels):
    if len(mesh.vertices) != len(labels):
        print("Mesh e JSON NON sono allineati!")
        print(f"Vertices mesh: {len(mesh.vertices)}")
        print(f"Labels JSON:  {len(labels)}")
        return False
    print("Mesh e JSON allineati.")
    return True

