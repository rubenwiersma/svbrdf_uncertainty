from __future__ import annotations as __annotations__

import drjit as dr
import mitsuba as mi
import torch

class _RenderRayOp(dr.CustomOp):
    """
    This class is an implementation detail of the render() function. It
    realizes a CustomOp that provides evaluation, and forward/reverse-mode
    differentiation callbacks that will be invoked as needed (e.g. when a
    rendering operation is encountered by an AD graph traversal).
    """

    def __init__(self) -> None:
        super().__init__()
        self.variant = mi.variant()

    def eval(self, scene, ray, sensor, params, integrator, seed, spp):
        self.scene = scene
        self.ray = ray
        self.sensor = sensor
        self.params = params
        self.integrator = integrator
        self.seed = seed
        self.spp = spp

        with dr.suspend_grad():
            return self.integrator.render(
                scene=self.scene,
                ray=self.ray,
                sensor=sensor,
                seed=seed[0],
                spp=spp[0],
                develop=True,
                evaluate=False
            )

    def forward(self):
        mi.set_variant(self.variant)
        if not isinstance(self.params, mi.SceneParameters):
            raise Exception('An instance of mi.SceneParameter containing the '
                            'scene parameter to be differentiated should be '
                            'provided to mi.render() if forward derivatives are '
                            'desired!')
        self.set_grad_out(
            self.integrator.render_forward(self.scene, self.params, self.ray,
                                           self.sensor, self.seed[1], self.spp[1]))

    def backward(self):
        mi.set_variant(self.variant)
        if not isinstance(self.params, mi.SceneParameters):
            raise Exception('An instance of mi.SceneParameter containing the '
                            'scene parameter to be differentiated should be '
                            'provided to mi.render() if backward derivatives are '
                            'desired!')
        self.integrator.render_backward(self.scene, self.params, self.grad_out(), self.ray,
                                        self.sensor, self.seed[1], self.spp[1])

    def name(self):
        return "RenderOp"

def render_ray(scene: mi.Scene,
            params: Any = None,
            ray: mi.RayDifferential3f = None,
            sensor: Union[int, mi.Sensor] = 0,
            integrator: mi.Integrator = None,
            seed: int = 0,
            seed_grad: int = 0,
            spp: int = 0,
            spp_grad: int = 0) -> mi.TensorXf:
    """
    This function mimics the Mitsuba render function, but allows to specify
    custom rays to be used for rendering.
    """

    if params is not None and not isinstance(params, mi.SceneParameters):
        raise Exception('params should be an instance of mi.SceneParameter!')

    assert isinstance(scene, mi.Scene)

    if integrator is None:
        integrator = scene.integrator()

    if integrator is None:
        raise Exception('No integrator specified! Add an integrator in the scene '
                        'description or provide an integrator directly as argument.')

    if isinstance(sensor, int):
        if len(scene.sensors()) == 0:
            raise Exception('No sensor specified! Add a sensor in the scene '
                            'description or provide a sensor directly as argument.')
        sensor = scene.sensors()[sensor]

    assert isinstance(integrator, mi.Integrator)
    assert isinstance(sensor, mi.Sensor)

    if spp_grad == 0:
        spp_grad = spp

    if seed_grad == 0:
        # Compute a seed that de-correlates the primal and differential phase
        seed_grad = mi.sample_tea_32(seed, 1)[0]
    elif seed_grad == seed:
        raise Exception('The primal and differential seed should be different '
                        'to ensure unbiased gradient computation!')

    return dr.custom(_RenderRayOp, scene, ray, sensor, params, integrator,
                     (seed, seed_grad), (spp, spp_grad))

def sample_rays_multiple_sensors(integrator, scene, sensors, seed):
    """Sample a ray per pixel, per sensor and concatenate all the rays.
    Outputs the origin and direction of the rays, as well as the wavelengths.
    """
    with dr.suspend_grad():
        o, d = [], []
        wavelengths = None
        for i, sensor in enumerate(sensors):
            # We only use one primary ray per pixel and sample it multiple times
            sampler, spp = integrator.prepare(sensor=sensor, seed=seed, spp=1)
            ray, _, _ = integrator.sample_rays(scene, sensor, sampler)
            if wavelengths is None:
                wavelengths = ray.wavelengths
            o.append(ray.o.torch().cpu())
            d.append(ray.d.torch().cpu())
        o = torch.cat(o, dim=0)
        d = torch.cat(d, dim=0)
        return o, d, wavelengths

def integrate_ray_samples(L, spp):
    """Integrate #spp samples for one pixel by taking the average."""
    n_out = dr.width(L) // spp
    scatter_idx = dr.repeat(dr.arange(mi.UInt, n_out), spp)
    L_integrated = dr.zeros(type(L), n_out)
    dr.scatter_reduce(dr.ReduceOp.Add, L_integrated, L, scatter_idx)
    return L_integrated * (1 / spp)