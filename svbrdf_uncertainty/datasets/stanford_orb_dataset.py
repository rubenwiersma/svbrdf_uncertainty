from pathlib import Path
import json

import cv2
import drjit as dr
import gin
import mitsuba as mi
from mitsuba import ScalarTransform4f as T
import numpy as np
import torch


@gin.configurable
class StanfordORBDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, scene, split='train', scale_factor=1.0, texture_resolution=512, envmap_idx=0, hdr=True):
        # Setup paths to input folders
        self.root_path = Path(root_path)
        self.scene_label = scene
        self.input_folder = 'blender_HDR' if hdr else 'blender_LDR'
        self.ground_truth_folder = self.root_path / 'ground_truth' / scene
        self.data_path = self.root_path / self.input_folder / scene

        # Read metadata on views
        with open(self.data_path / f'transforms_{split}.json', 'r', encoding='utf8') as f:
            transforms_data = json.load(f)
        self.fov_x = transforms_data['camera_angle_x'] / np.pi * 180 if split == 'train' else None
        self.frames_data = transforms_data['frames']

        # Get envmap and corresponding transform
        with open(self.data_path / f'transforms_test.json', 'r', encoding='utf8') as f:
            envmap_info = json.load(f)['frames'][envmap_idx]
        self.envmap_path = str(self.ground_truth_folder / 'env_map' / (envmap_info['file_path'].split('/')[-1] + '.exr'))
        self.envmap_transform = T(correct_axes_stanford_orb(np.array(envmap_info['transform_matrix'])))

        # Get mesh path
        meshes_in_data_path = list((self.ground_truth_folder / 'mesh_blender').glob('*.obj'))
        assert len(meshes_in_data_path) == 1
        self.mesh = str(self.data_path / meshes_in_data_path[0])

        self.scale_factor = scale_factor
        self.texture_resolution = texture_resolution
        self.hdr = hdr
        self.file_extension = '.exr' if hdr else '.png'

        f = self.frames_data[0]
        frame0_path = self.data_path / (f['file_path'] + self.file_extension)
        if split == 'novel':
            frame0_path = self.root_path / self.input_folder / f['scene_name'] / (f['file_path'] + self.file_extension)

        frame0 = mi.TensorXf(mi.Bitmap(str(frame0_path))).numpy()
        self.w = int(frame0.shape[1] * scale_factor)
        self.h = int(frame0.shape[0] * scale_factor)

        self._sensors = self.get_sensors()
        self._scene = self.get_scene()

        self.cache_path = self.data_path / 'cache.pt'
        if not self.cache_path.exists():
            print('Cached dataset not found, creating it now. This may take a while.')
            self.frames = self.load_frames(self.cache_path).to('cuda')
        else:
            self.frames = torch.load(self.cache_path, weights_only=False).to('cuda')

    def __len__(self):
        return len(self.frames_data)

    def __getitem__(self, index):
        return self.frames[index]

    @property
    def sensors(self):
        return self._sensors

    @property
    def scene(self):
        return self._scene

    @property
    def relighting_scenes(self):
        if self._relighting_scenes is None:
            self._relighting_scenes = self.get_relighting_scenes()
        return self._relighting_scenes


    def load_frames(self, cache_path):
        n_frames = len(self.frames_data)
        frames = torch.zeros(n_frames, self.h, self.w, 3)
        for i in range(n_frames):
            frame_data = self.frames_data[i]
            frame_path = self.data_path / (frame_data['file_path'] + self.file_extension)
            # Load and resize frame with RGB convention
            frame_opencv = mi.TensorXf(mi.Bitmap(str(frame_path))).numpy()
            if not self.hdr:
                frame_opencv = frame_opencv / 255.0
            frame_opencv = cv2.resize(frame_opencv, (self.w, self.h))
            frames[i] = torch.from_numpy(frame_opencv)
        torch.save(frames, cache_path)
        return frames


    def get_scene(self, mesh=None, material=None, envmap_path=None, envmap_transform=None):
        envmap_path = self.envmap_path if envmap_path is None else envmap_path
        envmap_transform = self.envmap_transform if envmap_transform is None else envmap_transform
        envmap = mi.load_dict({'type': 'envmap', 'filename': envmap_path, 'to_world': envmap_transform})

        material = mi.load_dict(
            {
                'type': 'principled',
                'base_color': {
                    'type': 'bitmap',
                    'bitmap': mi.Bitmap(dr.full(mi.TensorXf, 0.5, shape=(self.texture_resolution, self.texture_resolution, 3))),
                    'filter_type': 'nearest',
                },
                'roughness': {
                    'type': 'bitmap',
                    'bitmap': mi.Bitmap(dr.full(mi.TensorXf, 0.5, shape=(self.texture_resolution, self.texture_resolution))),
                    'filter_type': 'nearest',
                },
                'metallic': {
                    'type': 'bitmap',
                    'bitmap': mi.Bitmap(dr.full(mi.TensorXf, 0.0, shape=(self.texture_resolution, self.texture_resolution))),
                    'filter_type': 'nearest',
                }
            }
        ) if material is None else material

        mesh = self.mesh if mesh is None else mesh
        if isinstance(mesh, str):
            shape = mi.load_dict({'type': Path(mesh).suffix[1:], 'filename': mesh, 'material': material})
        else:
            mesh['material'] = material
            shape = mi.load_dict(mesh)
        scene = mi.load_dict({'type': 'scene', 'integrator': {'type': 'path', 'max_depth': -1}, 'envmap': envmap, 'sensor': self.sensors[0], 'shape': shape})

        return scene


    def get_sensors(self):
        sensors = []
        for f in self.frames_data:
            to_world = correct_axes_stanford_orb(np.array(f['transform_matrix']))
            fov = f['camera_angle_x'] / np.pi * 180 if self.fov_x is None else self.fov_x

            sensors.append(
                mi.load_dict(
                    {
                        'type': 'perspective',
                        'fov': fov,
                        'to_world': T(to_world),
                        'film': {
                            'type': 'hdrfilm',
                            'width': self.w,
                            'height': self.h,
                            'pixel_format': 'rgb',
                        },
                    }
                )
            )

        return sensors


def correct_axes_stanford_orb(mat):
    mat[:3, :3] = mat[:3, :3] @ np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    return mat
