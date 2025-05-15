from pathlib import Path
import shutil

import drjit as dr
import gin
import mitsuba as mi
import numpy as np
import trimesh
import torch

from svbrdf_uncertainty.util import sample_sphere, sample_hemisphere
from svbrdf_uncertainty.util.io import read_texture


@gin.configurable
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, mesh_path, base_color_path, roughness_path, metallic_path, envmap_path, sensor_count=100, resolution=512, spp=256, radius=1.0, full_sphere=False, hdr=True, rerender_frames=False, views_subfolder=None):
        """Synthetic dataset for inverse rendering.
        Renders a scene with a given mesh and textures,
        and samples sensors from a (hemi)sphere around the mesh.
        The dataset is stored in the given data path, and can be loaded from disk.

        Args:
            data_path (str): Path to the dataset.
            mesh_path (str): Path to mesh in the scene or Mitsuba shape dictionary.
            base_color_path (str): Path to base color texture.
            roughness_path (str): Path to roughness texture.
            metallic_path (str): Path to metallic texture.
            envmap_path (str): Path to environment map.
            sensor_count (int): Number of sensors to use.
            resolution (int): Resolution of the sensors (currently only supports square).
            spp (int): Samples per pixel for rendering.
            radius (float): Radius of the hemisphere to sample the sensors from.
            full_sphere (bool, optional): Whether to sample the sensors from a full sphere
                or a hemisphere. Defaults to False.
            hdr (bool, optional): Whether to render the dataset in HDR or LDR. Defaults to True.
            rerender_frames (bool, optional): Whether to rerender the frames if they already exist.
                Defaults to False.
            views_subfolder (str, optional): Subfolder to store the views in. Defaults to None.
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.views_path = self.data_path / 'views'
        if views_subfolder is not None:
            self.views_path = self.views_path / views_subfolder
        self.mesh = str(self.data_path / mesh_path)
        self.envmap_path = envmap_path
        self.sensor_count = sensor_count
        self.resolution = resolution
        self.spp = spp
        self.radius = radius
        self.full_sphere = full_sphere
        self.hdr = hdr

        # Configure sensors
        self._sensors = self.get_sensors()

        # Create synthetic scene
        self.base_color = read_texture(self.data_path / base_color_path)
        self.roughness = read_texture(self.data_path / roughness_path)
        if self.roughness.ndim > 2:
            self.roughness = self.roughness[..., 0]
        self.metallic = read_texture(self.data_path / metallic_path)
        if self.metallic.ndim > 2:
            self.metallic = self.metallic[..., 0]
        material = self.get_material()
        self._scene = self.get_scene(self.mesh, material, envmap_path)

        if rerender_frames:
            shutil.rmtree(self.views_path)

        # Render dataset if it does not exist
        if not self.views_path.exists():
            self.render_dataset()

        # Load dataset in memory
        self.frames = torch.stack([self.load_frame(i) for i in range(sensor_count)], dim=0)


    def __len__(self):
        return self.sensor_count

    def __getitem__(self, index):
        return self.frames[index]

    @property
    def sensors(self):
        return self._sensors

    @property
    def scene(self):
        return self._scene

    def load_frame(self, index):
        """Load view from disk given an index of the view."""
        frame = mi.TensorXf(mi.Bitmap(str(self.views_path / f'ref_{index:03d}.exr')))
        if not self.hdr:
            frame = dr.clamp(frame, 0.0, 1.0)
        return frame.torch()

    def get_scene(self, mesh, material, envmap_path, load_parallel=False):
        if self.hdr:
            envmap = mi.load_dict({'type': 'envmap', 'filename': envmap_path}, parallel=load_parallel)
        else:
            envmap_clamped = mi.util.convert_to_bitmap(dr.clamp(mi.TensorXf(mi.Bitmap(envmap_path)), 0.0, 1.0))
            envmap = mi.load_dict({'type': 'envmap', 'bitmap': envmap_clamped}, parallel=load_parallel)
        if isinstance(mesh, str):
            shape = mi.load_dict({'type': Path(mesh).suffix[1:], 'filename': mesh, 'material': material}, parallel=load_parallel)
        else:
            mesh['material'] = material
            shape = mi.load_dict(mesh, parallel=load_parallel)
        scene = mi.load_dict({'type': 'scene', 'integrator': {'type': 'path', 'max_depth': -1}, 'envmap': envmap, 'sensor': self.sensors[0], 'shape': shape}, parallel=load_parallel)

        return scene

    def get_sensors(self, load_parallel=False):
        """Create a list of sensors sampled from a hemisphere."""
        look_at_target = trimesh.load(self.mesh).centroid

        sensors = []
        sampling_fun = sample_sphere if self.full_sphere else sample_hemisphere
        sensor_origins = sampling_fun(self.sensor_count, self.radius, method="fibonacci")
        for sensor_origin in sensor_origins:
            sensors.append(
                mi.load_dict(
                    {
                        'type': 'perspective',
                        'fov_axis': 'smaller',
                        'fov': 45,
                        'to_world': mi.ScalarTransform4f.look_at(target=look_at_target, origin=sensor_origin + look_at_target, up=[0.0, 1.0, 0.0]),
                        'film': {'type': 'hdrfilm', 'width': self.resolution, 'height': self.resolution, 'filter': {'type': 'gaussian'}, 'pixel_format': 'rgb'},
                    }, parallel=load_parallel
                )
            )

        return sensors

    def render_dataset(self):
        """Render the dataset (#sensor_count views of the scene) and store it in data_path."""
        self.views_path.mkdir(parents=True, exist_ok=True)

        # Create and store synthetic training data in out path
        for i in range(self.sensor_count):
            ref_image = mi.render(self.scene, sensor=self.sensors[i], spp=self.spp)
            mi.util.write_bitmap(str(self.views_path / f'ref_{i:03d}.exr'), ref_image)

    def get_material(self, load_parallel=False):
        return mi.load_dict({
            'type': 'principled',
            'eta': 1.5,
            'metallic': {
                'type': 'bitmap',
                'bitmap': mi.util.convert_to_bitmap(self.metallic)
            },
            'base_color': {
                'type': 'bitmap',
                'bitmap': mi.util.convert_to_bitmap(self.base_color)
            },
            'roughness': {
                'type': 'bitmap',
                'bitmap': mi.util.convert_to_bitmap(self.roughness)
            }
        }, parallel=load_parallel)
