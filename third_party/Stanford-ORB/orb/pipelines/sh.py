from .base import BasePipeline
import json
from orb.constant import PROJ_ROOT
import glob
import os
from pathlib import Path


import logging


root_path = Path(__file__).parent.parent.absolute()
SCENE_DATA_DIR = (root_path / '../../../../data/Stanford ORB').resolve()
OUTPUT_DATA_DIR = (root_path / '../../../out/stanford_orb').resolve()
logger = logging.getLogger(__name__)


class Pipeline(BasePipeline):
    def test_new_light(self, scene: str, experiment_name: str = None, overwrite: bool = False):
        with open(os.path.join(SCENE_DATA_DIR, f'blender_HDR/{scene}/transforms_novel.json'), 'r', encoding='utf8') as f:
            transforms_data = json.load(f)
        frames_data = transforms_data['frames']

        target_paths = []
        output_paths = []
        for frame in frames_data:
            target_paths.append(os.path.join(SCENE_DATA_DIR, "blender_HDR", frame['scene_name'], frame['file_path'] + '.exr'))
            result_folder = os.path.join(OUTPUT_DATA_DIR, f'{scene}/relighting/sh')
            if experiment_name is None:
                experiment_name = [f for f in sorted(os.listdir(result_folder)) if os.path.isdir(os.path.join(result_folder, f))][-1]
            output_paths.append(os.path.join(result_folder, experiment_name, frame['scene_name'], frame['file_path'].split('/')[-1] + '.exr'))
        for target_path in target_paths:
            assert os.path.exists(target_path), target_path

        ret = [{'output_image': output_paths[ind], 'target_image': target_paths[ind]}
               for ind in range(len(output_paths))]
        return ret
