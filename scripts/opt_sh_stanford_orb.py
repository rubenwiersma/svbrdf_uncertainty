import importlib
import json
import os
import sys
from pathlib import Path
import time

import gin
import numpy as np
from tqdm import tqdm

from evaluate_relighting import evaluate
from svbrdf_uncertainty.datasets import StanfordORBDataset

root_path = Path(__file__).parent.parent.absolute()
dataset_root_folder = (root_path / '../data/Stanford ORB').resolve()
default_config_folder = (root_path / 'experiments/configs/sh_stanford_orb').resolve()

sys.path.append(str(root_path))
from opt_sh import optimize_material

sys.path.append(str((root_path / 'third_party/Stanford-ORB').resolve()))
from orb.utils.ppp import list_of_dicts__to__dict_of_lists
from orb.utils.test import compute_metrics_image_similarity
from orb.constant import get_scenes_from_id
from orb.constant import PROJ_ROOT
from orb.constant import get_scenes_from_id


@gin.configurable
def run_benchmark(experiment_name, texture_res=512,
                  lmax=5, sh_regularizer=1e-3, sh_regularizer_func='exp', sample_margin=1.0, sample_weight_exponent=1,
                  fresnel_enabled=True, shadowing_enabled=True, masking_enabled=True, update_masking_every=50,
                  loss_domain='angular', num_iterations=100, learning_rate=0.02, init_global=False, early_stopping=0,
                  smoothness_weight=1e-3, global_sharing_weight=0, smoothness_k=20, smoothness_kernel_width=0.4,
                  loss_entropy_weight=0, loss_entropy_exponent=1,
                  entropy_sigma=1e-5, entropy_resolution=8):
    ground_truth_folder = dataset_root_folder / 'ground_truth'
    scenes = [f for f in os.listdir(ground_truth_folder) if os.path.isdir(os.path.join(ground_truth_folder, f))]

    print(f'Ours: Starting {experiment_name}')

    sh_timings = []
    for scene in tqdm(scenes):
        # Training
        # ---
        dataset = StanfordORBDataset(str(dataset_root_folder), scene, scale_factor=0.25)
        timing = optimize_material(
            label='stanford_orb/' + scene, dataset=dataset, texture_res=texture_res, experiment_name=experiment_name,
            lmax=lmax, sh_regularizer=sh_regularizer, sh_regularizer_func=sh_regularizer_func, sample_margin=sample_margin, sample_weight_exponent=sample_weight_exponent,
            fresnel_enabled=fresnel_enabled, shadowing_enabled=shadowing_enabled, masking_enabled=masking_enabled, update_masking_every=update_masking_every,
            loss_domain=loss_domain, num_iterations=num_iterations, learning_rate=learning_rate, init_global=init_global, early_stopping=early_stopping,
            smoothness_weight=smoothness_weight, global_sharing_weight=global_sharing_weight, smoothness_k=smoothness_k, smoothness_kernel_width=smoothness_kernel_width,
            loss_entropy_weight=loss_entropy_weight, loss_entropy_exponent=loss_entropy_exponent,
            entropy_sigma=entropy_sigma, entropy_resolution=entropy_resolution,
            vis_bake_textures=False)
        sh_timings.append(timing)

        # Evaluation
        # ---
        dataset_eval = StanfordORBDataset(str(dataset_root_folder), scene, split='novel', scale_factor=0.25)
        result_folder = root_path / 'out/stanford_orb' / scene
        evaluate(dataset_eval, result_folder, 'sh', experiment_name=experiment_name)

    print('Evaluating...')

    # Run Stanford ORB evaluation scripts
    stanford_orb_pipeline = getattr(importlib.import_module(f'orb.pipelines.sh', package=None), 'Pipeline')()
    stanford_orb_scenes = get_scenes_from_id('full')
    stanford_orb_output_path = root_path / f'out/stanford_orb/sh_{experiment_name}.json'

    ret_new_light = dict()
    for scene in tqdm(stanford_orb_scenes):
        results = stanford_orb_pipeline.test_new_light(scene, experiment_name=experiment_name)
        ret_new_light[scene] = compute_metrics_image_similarity(results, scale_invariant=True)

    scores = {'light_all': ret_new_light}
    ret_new_light = list_of_dicts__to__dict_of_lists(list(ret_new_light.values()))
    ret_new_light = {k: (np.mean(v), np.std(v)) for k, v in ret_new_light.items()}
    ret_new_light['scene_count'] = len(scores['light_all'])
    ret_new_light['time'] = (np.mean(sh_timings), np.std(sh_timings))

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

    print(f'Found {len(experiments)} config files in {str(config_folder)}')

    for experiment in experiments:
        gin.clear_config()
        gin.parse_config_file(config_folder / experiment)
        run_benchmark(experiment_name=experiment[:-4])
