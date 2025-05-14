import os
from pathlib import Path
import sys

import gin
import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

from mitsuba import ScalarTransform4f as T
import numpy as np
from pyexr import read
from tqdm import tqdm

from svbrdf_uncertainty.datasets.stanford_orb_dataset import correct_axes_stanford_orb
from svbrdf_uncertainty.datasets import StanfordORBDataset


current_path = Path(__file__).parent.absolute()

def evaluate(dataset, root_folder, method='sh', experiment_name=None):
    if isinstance(root_folder, str):
        root_folder = Path(root_folder)

    # Get last results
    result_folder = root_folder / method
    if experiment_name is None:
        experiment_name = [f for f in sorted(os.listdir(result_folder)) if os.path.isdir(os.path.join(result_folder, f))][-1]

    # Create material for Mitsuba
    material = {
        'type': 'principled',
        'eta': 1.5,
        'metallic': {
            'type': 'bitmap',
            'bitmap': mi.util.convert_to_bitmap(read(str(result_folder / experiment_name / 'metallic.exr')))
        },
        'base_color': {
            'type': 'bitmap',
            'bitmap': mi.util.convert_to_bitmap(read(str(result_folder / experiment_name / 'base_color.exr')))
        },
        'roughness': {
            'type': 'bitmap',
            'bitmap': mi.util.convert_to_bitmap(read(str(result_folder / experiment_name / 'roughness.exr')))
        }
    }

    for frame, sensor in zip(dataset.frames_data, dataset.sensors):
        # Create scene with novel envmap
        frame_id = frame['file_path'].split('/')[-1]
        envmap_path = str(dataset.root_path / 'ground_truth' / frame['scene_name'] / 'env_map' / (frame_id + '.exr'))
        envmap_transform = T(correct_axes_stanford_orb(np.array(frame['transform_matrix'])))
        scene = dataset.get_scene(material=material, envmap_path=envmap_path, envmap_transform=envmap_transform)

        # Render scene and save in corresponding folder
        out_folder = (root_folder / 'relighting' / method / experiment_name / frame['scene_name'])
        out_folder.mkdir(parents=True, exist_ok=True)
        mi.util.write_bitmap(str(out_folder / (frame_id + '.exr')), mi.render(scene, sensor=sensor, spp=128))


if __name__ == '__main__':
    method = sys.argv[1] if len(sys.argv) > 1 else 'sh'
    assert method in ['sh', 'mitsuba']

    dataset_root_folder = '/local/home/rwiersma/Dev/Adobe/data/Stanford ORB'
    result_root_folder = current_path / 'out/stanford_orb'

    scenes = [f for f in os.listdir(result_root_folder) if os.path.isdir(os.path.join(result_root_folder, f))]
    for scene in tqdm(scenes):
        dataset = StanfordORBDataset(dataset_root_folder, scene, split='novel', scale_factor=0.25)
        evaluate(dataset, result_root_folder / scene, method)
