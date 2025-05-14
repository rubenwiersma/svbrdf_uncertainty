import nvdiffrast.torch as nvdr
import torch
import numpy as np
import drjit as dr
import mitsuba as mi


def read_mesh(mesh, load_parallel=False):
    """Reads positions, faces and uv coordinates from a Mitsuba mesh.
    """
    if isinstance(mesh, str):
        assert mesh[-3:] == 'obj' or mesh[-3:] == 'ply'
        mesh = mi.load_dict({
            'type': mesh[-3:],
            'filename': mesh,
        }, parallel=load_parallel)
    else:
        mesh = mi.load_dict(mesh, parallel=load_parallel)

    # Read out mesh positions, faces and uv coordinates
    mesh_params = mi.traverse(mesh)
    pos = mesh_params['vertex_positions'].torch().reshape(-1, 3)
    faces = dr.cuda.ad.Int(mesh_params['faces']).torch().reshape(-1, 3)
    uv = mesh_params['vertex_texcoords'].torch().reshape(-1, 2)
    normals = mesh_params['vertex_normals'].torch().reshape(-1, 3)
    return pos, faces, uv, normals


def transform_pos(mtx, pos):
    """Apply a transformation matrix to a set of positions.
    """
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1], device=pos.device)], axis=1)
    posw = torch.matmul(posw, mtx.t())
    return posw


def get_mvp(sensor, return_camera_pos=False):
    """Compute the Model-View-Projection matrix (MVP) from a Mitsuba sensor.
    """
    sensor_params = mi.traverse(sensor)
    proj = mi.Transform4f.perspective(
        fov=sensor_params['x_fov'],
        near=sensor_params['near_clip'],
        far=sensor_params['far_clip']
    ).matrix.torch().squeeze()
    camera_to_world = sensor_params['to_world'].matrix.torch().squeeze()
    world_to_camera = torch.linalg.inv(camera_to_world)
    mvp = proj @ world_to_camera

    # Flip x, y axis to match OpenGL convention
    mvp[:2] = -1 * mvp[:2]
    if return_camera_pos:
        return mvp, camera_to_world[:3, 3]
    return mvp


def get_z_buffer(pos, faces, mvp, size, gl_context):
    """Compute the z-buffer from a mesh and sensor.
    """
    # create context for renderer
    pos_clip = transform_pos(mvp, pos)
    rast_out, rast_out_db = nvdr.rasterize(gl_context, pos_clip[None], faces, resolution=size)
    return rast_out[0, ..., 2]


def render_nvdiffrast(gl_context, pos_clip, pos_idx, uv, uv_idx, tex, size, max_mip_level=3):
    """Render a textured mesh.

    Args:
        gl_context (nvdiffrast context): nvdiffrast.RasterizeCudaContext
            or nvdiffrast.RasterizeGLContext.
        pos_clip_space (torch.Tensor): Mesh vertices in clip space.
        pos_idx (torch.Tensor): Mesh faces.
        uv (torch.Tensor): Mesh uv coordinates.
        uv_idx (torch.Tensor): Mesh uv indices.
        tex (torch.Tensor): Texture.
        size (list): Resolution of the output image as a list per axis.
        max_mip_level (optional, int): Max mip level of the texture.
            Defaults to 3.
    """
    rast_out, rast_out_db = nvdr.rasterize(gl_context, pos_clip, pos_idx, resolution=size)

    texc, texd = nvdr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
    color = nvdr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)

    color = color * torch.clamp(rast_out[..., -1:], 0, 1)
    return color