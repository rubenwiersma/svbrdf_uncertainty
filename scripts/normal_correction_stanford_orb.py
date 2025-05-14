import os
import trimesh
import numpy as np
import shutil

# Get the current working directory
current_directory = os.getcwd()
# Append a relative directory to the current working directory and resolve the aboslute path
# TODO: Change this to the path where you downloaded the Stanford ORB dataset
root_folder = os.path.abspath(os.path.join(current_directory, '..', 'data', 'Stanford ORB', 'ground_truth'))
folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

for folder in folders:
    backup_folder = os.path.join(root_folder, folder, 'mesh_blender/mesh_original')
    if not os.path.exists(backup_folder):
        os.mkdir(backup_folder)
        mesh_path = os.path.join(root_folder, folder, 'mesh_blender/mesh.obj')
        mesh = trimesh.load_mesh(mesh_path)
        mesh.vertex_normals = mesh.vertex_normals @ np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        shutil.move(mesh_path, backup_folder)
        with open(mesh_path, 'w', encoding='utf8') as f:
            f.write(trimesh.exchange.obj.export_obj(mesh, include_normals=False))