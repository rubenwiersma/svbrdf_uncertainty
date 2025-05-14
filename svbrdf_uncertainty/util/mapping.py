import mitsuba as mi
import torch
import nvdiffrast.torch as nvdr

from .geometry import build_tangent_basis
from .sample_sphere import sample_sphere
from .nvdiffrast import read_mesh, transform_pos, get_z_buffer, render_nvdiffrast, get_mvp


def map_world_to_local(dataset, mvp, camera_pos, envmap, mesh, envmap_transform=None, texture_res=64):
    """Map radiances from image space and world space to
    local coordinates for each texel.

    Args:
        dataset (torch.utils.data.Dataset): A dataset containing images per view.
        envmap (torch.Tensor): The environment map of shape [H, W, 3].
        mesh (string or mi.Mesh): The mesh to get the local geometry from.
        texture_res (int): Resolution of the resulting square texture.
    """
    if envmap.shape[-1] > 3:
        envmap = envmap[..., :3]

    # Read out Mitsuba parameters before going to nvdiffrast
    mesh_pos, mesh_faces, mesh_uv, mesh_normal = read_mesh(mesh)
    position, normal, s_basis, t_basis = get_geometry_texture(mesh_pos, mesh_faces, mesh_uv, mesh_normal, texture_res)

    # Compute incoming and outgoing radiance for each texel
    outgoing_radiance = torch.zeros((texture_res, texture_res, len(camera_pos), 3), device=position.device)
    incoming_radiance = torch.zeros((texture_res, texture_res, len(camera_pos), 3), device=position.device)
    theta_local = torch.zeros((texture_res, texture_res, len(camera_pos)), device=position.device)
    phi_local = torch.zeros((texture_res, texture_res, len(camera_pos)), device=position.device)

    # create context for renderer
    gl_context = nvdr.RasterizeCudaContext()
    for i in range(len(camera_pos)):
        mvp_sensor, camera_pos_sensor = mvp[i], camera_pos[i]

        # Get radiance texture for a given mesh and view and compute angles
        outgoing_radiance[..., i, :] = project_image_to_texture(dataset[i], mesh_pos, mesh_faces, mvp_sensor, position, gl_context)

        view_vector = camera_pos_sensor[None, None] - position
        view_vector = view_vector / torch.linalg.norm(view_vector, dim=-1, keepdim=True).clamp(1e-5)

        # Reflection = 2 * dot(n, v) * n - v
        reflection_vector = 2 * torch.bmm(view_vector.view(-1, 1, 3), normal.view(-1, 3, 1)).squeeze(-1) * normal.view(-1, 3) - view_vector.view(-1, 3)

        # Convert to spherical coordinates
        theta_local[..., i] = torch.acos(torch.bmm(view_vector.view(-1, 1, 3), normal.view(-1, 3, 1)).squeeze()).view(texture_res, texture_res)
        phi_local[..., i] = torch.atan2(torch.bmm(view_vector.view(-1, 1, 3), t_basis.view(-1, 3, 1)).squeeze(),
                            torch.bmm(view_vector.view(-1, 1, 3), s_basis.view(-1, 3, 1)).squeeze()).view(texture_res, texture_res)

        # Sample values from environment map for incoming light directions
        incoming_radiance[..., i, :] = evaluate_envmap(reflection_vector, envmap).view(texture_res, texture_res, 3)

    return incoming_radiance, outgoing_radiance, theta_local, phi_local, normal, t_basis, s_basis, position


def sample_envmap(n_samples, envmap, normal, t_basis, s_basis, envmap_transform=None, method='fibonacci'):
    # Get virtual samples from light source to perform SH fitting
    incoming_samples = torch.from_numpy(sample_sphere(n_samples, method=method)).float().to(envmap.device)
    incoming_radiance = evaluate_envmap(incoming_samples, envmap, envmap_transform).view(1, 1, -1, envmap.shape[-1])
    # Convert to spherical coordinates
    theta_incoming = torch.acos(torch.matmul(incoming_samples.view(1, -1, 1, 3), normal.view(-1, 1, 3, 1)).squeeze()).view(*normal.shape[:2], -1)
    phi_incoming = (torch.atan2(torch.matmul(incoming_samples.view(1, -1, 1, 3), t_basis.view(-1, 1, 3, 1)).squeeze(),
                                torch.matmul(incoming_samples.view(1, -1, 1, 3), s_basis.view(-1, 1, 3, 1)).squeeze()).view(*normal.shape[:2], -1)
                    + torch.pi) # Rotate by 180 degrees
    phi_incoming = torch.where(phi_incoming > torch.pi, phi_incoming - 2 * torch.pi, phi_incoming)
    return incoming_radiance, theta_incoming, phi_incoming


def evaluate_envmap(directions, envmap, envmap_transform=None):
    """Evaluate emittance from an envmap in the given directions.
    Transforms rays to the envmap space if an envmap_transform is given.
    """
    if envmap_transform is not None:
        directions = directions @ envmap_transform[:3, :3]

    # Follows the envmap convention from Mitsuba
    #     0,0 -> u
    #    v -------------------- <- +Y
    #      |                  |
    #-Z -> |   +X   +Z   -X   | <- -Z
    #      |                  |
    #      -------------------- <- -Y
    #      ^-------wraps------^
    # Angle from +Y axis toward XZ plane, from 0 to pi
    theta_in_envmap = torch.acos(directions[..., 1].clip(-1, 1))
    # Angle from +Z toward -X, from -pi (-Z), along -1/2pi (+X), 0 (+Z), 1/2pi (-X), to pi (-Z)
    phi_in_envmap = torch.atan2(-directions[..., 0], directions[..., 2])
    # Map to pixel coordinate
    theta_pixel = (theta_in_envmap / torch.pi) * (envmap.shape[0] - 1)                  # [texture_res, texture_res]
    # Phi coordinate needs to be shifted by pi, so that -pi corresponds to u coordinate 0
    phi_pixel = ((phi_in_envmap + torch.pi) / (2 * torch.pi)) * (envmap.shape[1] - 1)   # [texture_res, texture_res]
    pos = torch.stack([theta_pixel, phi_pixel], dim=-1)                                 # [texture_res, texture_res, 2]
    # Lookup values in envmap using bilinear interpolation
    return bilinear_interpolate(pos, envmap)


def render_mirror(mesh, envmap_path, sensor, envmap_threshold=False):
    envmap = mi.Bitmap(envmap_path)
    if envmap_threshold:
        envmap = mi.Bitmap(mi.TensorXf((mi.TensorXf(envmap).torch().mean(dim=-1) >= 1.0) * 1.0))

    if isinstance(mesh, str):
        shape = mi.load_dict({
            'type': 'obj',
            'filename': mesh,
            'material': { 'type': 'conductor' }
        })
    else:
        if 'material' in mesh:
            del mesh['material']
        mesh['material'] = { 'type': 'conductor' }
        shape = mi.load_dict(mesh)

    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': -1
        },
        'envmap': {
            'type': 'envmap',
            'bitmap': envmap
        },
        'sensor': sensor,
        'shape': shape
    })

    return mi.render(scene, sensor=sensor, spp=10)


def render_aovs(mesh, sensor, aovs, spp=16):
    """Use Mitsuba 3 to render aovs
    """
    if isinstance(mesh, str):
        shape = mi.load_dict({
            'type': 'obj',
            'filename': mesh,
        })
    else:
        if 'material' in mesh:
            del mesh['material']
        shape = mi.load_dict(mesh)

    scene_aovs = mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'aov',
            'aovs': aovs
        },
        'sensor': sensor,
        'shape': shape
    })

    # Get uv_coords in image space
    aov_render = mi.render(scene_aovs, spp=spp)
    return aov_render


def project_image_to_texture(image, pos, faces, mvp, pos_texture, gl_context):
    # Project 3D texture positions to view space
    pos_image_space = transform_pos(mvp, pos_texture.reshape(-1, 3)).reshape(pos_texture.shape[:2] + (4,))

    # Get coordinates in pixel grid
    xy = ((pos_image_space[..., :2] / pos_image_space[..., 3:]) / 2 + 0.5).clip(0, 1)
    xy[..., 0] = xy[..., 0] * (image.shape[0] - 1)
    xy[..., 1] = xy[..., 1] * (image.shape[1] - 1)
    xy = torch.flip(xy, dims=(-1,))

    # Interpolate image and z-buffer to texels
    texture_z_buffer = bilinear_interpolate(xy, torch.cat([image, get_z_buffer(pos, faces, mvp, image.shape[:2], gl_context)[..., None]], dim=-1))

    # Test visibility by checking z-buffer with z-value
    z = pos_image_space[..., 2:3] / pos_image_space[..., 3:]
    visibility = (texture_z_buffer[..., 3:] - z).abs() < 1e-5
    texture = visibility * texture_z_buffer[..., :3]

    return texture


def render_texture(texture, pos, faces, uv, sensor):
    """Render a texture mesh using rasterization without shading.

    Args:
        texture (torch.Tensor): Texture.
        mesh (str or dict): Mitsuba mesh or path to a mesh.
        sensor (dict): Mitsuba sensor.
    """
    mvp = get_mvp(sensor)
    pos_clip = transform_pos(mvp, pos)
    gl_context = nvdr.RasterizeCudaContext()
    size = mi.traverse(sensor)['film.size']
    image = render_nvdiffrast(gl_context, pos_clip[None], faces, uv, faces, texture, size, max_mip_level=9)
    return image[0]


def render_mask(pos, faces, sensor):
    """Render a mask of a mesh from the given sensor.
    """
    mvp = get_mvp(sensor)
    pos_clip = transform_pos(mvp, pos)
    gl_context = nvdr.RasterizeCudaContext()
    size = mi.traverse(sensor)['film.size']
    rast_out, _ = nvdr.rasterize(gl_context, pos_clip[None], faces, resolution=size)
    return rast_out[0, ..., -1] > 0


def get_geometry_texture(pos, faces, uv, normals, tex_res=512):
    """Returns the positions, normals and tangent basis
    in the texture space of the mesh.
    """
    # Wrap around edges for triangles that are outside of the texture
    uv_offsets = torch.stack(torch.meshgrid(*[torch.arange(-2, 5, 2, device=uv.device)]*2, indexing='ij'), axis=-1).reshape(-1, 1, 2)
    uv_pos = (uv * 2 - 1)[None] + uv_offsets
    uv_pos = torch.cat([uv_pos, torch.zeros_like(uv_pos)], dim=-1)
    uv_pos[..., 3] = 1

    # Rasterize geometric data to triangles and interpolate
    gl_context = nvdr.RasterizeCudaContext()
    rast_out, rast_out_db = nvdr.rasterize(gl_context, uv_pos, faces, [tex_res]*2)
    texc, texd = nvdr.interpolate(torch.cat([pos[None], normals[None]], axis=-1), rast_out, faces)

    # Read out interpolated data and compute basis
    position_normal = texc.sum(dim=0)
    position = position_normal[..., :3]
    position = grow_local_mean(position, 5) # Grow local mean to fill holes on boundary edges
    normal = position_normal[..., 3:]
    normal = grow_local_mean(normal, 5)
    normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True).clamp(1e-5)
    s_basis, t_basis = [basis.reshape(normal.shape) for basis in build_tangent_basis(normal.reshape(-1, 3))]

    return position, normal, s_basis, t_basis


def bilinear_interpolate(pos, grid_values):
    """Bilinear interpolation of grid values at positions pos.
    """
    # Get bilinear interpolation weights
    # w00 |--- w01
    # |---x--- |
    # |   |    |
    # w10 ---- w11
    w00 = (1 - pos[..., 0] % 1) * (1 - pos[..., 1] % 1)
    w01 = (1 - pos[..., 0] % 1) * (pos[..., 1] % 1)
    w10 = (pos[..., 0] % 1) * (1 - pos[..., 1] % 1)
    w11 = (pos[..., 0] % 1) * (pos[..., 1] % 1)

    interp_values = w00[..., None] * grid_values[pos[..., 0].floor().long(), pos[..., 1].floor().long()] + \
                    w01[..., None] * grid_values[pos[..., 0].floor().long(), pos[..., 1].ceil().long()] + \
                    w10[..., None] * grid_values[pos[..., 0].ceil().long(), pos[..., 1].floor().long()] + \
                    w11[..., None] * grid_values[pos[..., 0].ceil().long(), pos[..., 1].ceil().long()]
    return interp_values


def grow_local_mean(image, kernel_size=5):
    """Extend boundary pixels by local mean."""
    H, W, C = image.shape
    image_unfold = torch.nn.functional.unfold(image.permute(2, 0, 1)[None], kernel_size=kernel_size, padding=(kernel_size - 1) // 2).view(C, -1, H, W)
    image_mean = (image_unfold.sum(dim=1) / (image_unfold.abs().sum(dim=0) > 0).sum(dim=0).clip(1e-10)[None]).permute(1, 2, 0)
    mask = (image.abs().sum(dim=-1) == 0)[..., None].repeat(1, 1, C)
    image[mask] = image_mean[mask]
    return image