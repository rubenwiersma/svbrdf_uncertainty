import importlib
import json
import os
import sys
from pathlib import Path

import gin
import numpy as np
from tqdm import tqdm

from evaluate_relighting import evaluate
from svbrdf_uncertainty.datasets import StanfordORBDataset

root_path = Path(__file__).parent.parent.absolute()
dataset_root_folder = (root_path / '../data/Stanford ORB').resolve()
default_config_folder = root_path / 'experiments/configs/mitsuba_stanford_orb'

sys.path.append(str(root_path))
from opt_mi import optimize_material

sys.path.append(str((root_path / 'third_party/Stanford-ORB').resolve()))
from orb.utils.ppp import list_of_dicts__to__dict_of_lists
from orb.utils.test import compute_metrics_image_similarity
from orb.constant import get_scenes_from_id
from orb.constant import PROJ_ROOT
from orb.constant import get_scenes_from_id


@gin.configurable
def run_benchmark(experiment_name, texture_res=512, epoch_count=10, learning_rate=0.01, early_stopping=0.01, grad_spp=32,
                  smoothness_weight=1e-3, global_sharing_weight=0, smoothness_k=-1, smoothness_kernel_width=0.5,
                  init_from=None):
    ground_truth_folder = dataset_root_folder / 'ground_truth'
    scenes = [f for f in os.listdir(ground_truth_folder) if os.path.isdir(os.path.join(ground_truth_folder, f))]
    keys = ['shape.bsdf.base_color.data', 'shape.bsdf.roughness.data', 'shape.bsdf.metallic.data']

    print(f'Mitsuba: Starting {experiment_name}')

    mi_timings = []
    for scene in tqdm(scenes):
        # Training
        # ---
        init_textures_path = None
        if init_from is not None:
            init_textures_path = root_path / 'out/stanford_orb' / scene / init_from
        dataset = StanfordORBDataset(str(dataset_root_folder), scene, scale_factor=0.25)
        timing = optimize_material(
            label='stanford_orb/' + scene, dataset=dataset, keys=keys, texture_res=texture_res, experiment_name=experiment_name,
            epoch_count=epoch_count, learning_rate=learning_rate, early_stopping=early_stopping,
            grad_spp=grad_spp, smoothness_weight=smoothness_weight, global_sharing_weight=global_sharing_weight, smoothness_k=smoothness_k, smoothness_kernel_width=smoothness_kernel_width,
            init_textures_path=init_textures_path,
            vis_bake_textures=False)
        mi_timings.append(timing)

        # Evaluation
        # ---
        dataset_eval = StanfordORBDataset(str(dataset_root_folder), scene, split='novel', scale_factor=0.25)
        result_folder = root_path / 'out/stanford_orb' / scene
        evaluate(dataset_eval, result_folder, 'mitsuba', experiment_name=experiment_name)

    print('Evaluating...')

    # Run Stanford ORB evaluation scripts
    stanford_orb_pipeline = getattr(importlib.import_module(f'orb.pipelines.mitsuba', package=None), 'Pipeline')()
    stanford_orb_scenes = get_scenes_from_id('full')
    stanford_orb_output_path = root_path / f'out/stanford_orb/mi_{experiment_name}.json'

    ret_new_light = dict()
    for scene in tqdm(stanford_orb_scenes):
        results = stanford_orb_pipeline.test_new_light(scene, experiment_name=experiment_name)
        ret_new_light[scene] = compute_metrics_image_similarity(results, scale_invariant=True)

    scores = {'light_all': ret_new_light}
    ret_new_light = list_of_dicts__to__dict_of_lists(list(ret_new_light.values()))
    ret_new_light = {k: (np.mean(v), np.std(v)) for k, v in ret_new_light.items()}
    ret_new_light['scene_count'] = len(scores['light_all'])
    ret_new_light['time'] = (np.mean(mi_timings), np.std(mi_timings))

    scores_stats = {'light': ret_new_light}
    os.makedirs(os.path.dirname(stanford_orb_output_path), exist_ok=True)
    with open(stanford_orb_output_path, 'w') as f:
        json.dump({'scores_stats': scores_stats,
                   'scores': scores}, f, indent=4)

    print('Done!')


if __name__ == '__main__':
    config_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else default_config_folder
    assert config_folder.exists() and config_folder.is_dir()

    experiments = []
    for root, dirs, files in os.walk(config_folder):
        if 'skip' in files:
            print(f'Skipping folder {root}')
        else:
            for file in files:
                if file[-3:] == 'gin':
                    experiments.append(os.path.relpath(os.path.join(root, file), config_folder))

    for experiment in experiments:
        gin.clear_config()
        gin.parse_config_file(config_folder / experiment)
        run_benchmark(experiment_name=experiment[:-4])