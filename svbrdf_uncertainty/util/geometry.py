import mitsuba as mi
import drjit as dr
import torch


def normalize_mesh(mesh):
    """Given a Mitsuba Mesh object,
    normalize the mesh into a unit sphere centered around the origin.
    """
    mesh_params = mi.traverse(mesh)
    pos = dr.unravel(mi.Point3d, mesh_params['vertex_positions']).torch()
    pos = pos - torch.mean(pos, dim=0)
    pos = pos / torch.max(torch.linalg.norm(pos, dim=1))
    mesh_params['vertex_positions'] = dr.ravel(mi.TensorXf(pos))
    mesh_params.update()
    return mesh


def build_tangent_basis(normal, EPS=1e-6):
    """Constructs an orthonormal tangent basis, given a normal vector.

    Args:
        normal (Tensor): an [N, 3] tensor with normals per point.
    """

    # Pick an arbitrary basis vector that does not align too much with the normal
    testvec = normal.new_tensor([[0, 1, 0]]).expand(normal.size(0), 3)
    testvec_alt = normal.new_tensor([[1, 0, 0]]).expand(normal.size(0), 3)
    testvec = torch.where(torch.bmm(normal.unsqueeze(1), testvec.unsqueeze(-1)).squeeze(-1).abs() > 0.9, testvec_alt, testvec)

    # Derive x basis using cross product and normalize
    x_basis = torch.linalg.cross(testvec, normal)
    x_basis = x_basis / torch.linalg.norm(x_basis, dim=-1, keepdim=True).clamp(EPS)

    # Derive y basis using cross product and normalize
    y_basis = torch.linalg.cross(normal, x_basis)
    y_basis = y_basis / torch.linalg.norm(y_basis, dim=-1, keepdim=True).clamp(EPS)
    return x_basis, y_basis
