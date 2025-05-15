from __future__ import annotations as __annotations__

import gc

import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


class CustomRayIntegrator(mi.ad.integrators.common.RBIntegrator):
    """Custom ray integrator for Mitsuba 3.
    This class allows to pass a custom batch of rays to the integrator,
    which we use to give a random set of rays sampled from all sensors per iteration,
    rather than the rays corresponding to a single sensor."""
    def __init__(self, props = mi.Properties()):
        super().__init__(props)
        self.integrator = props.get('integrator')

    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               ray: mi.RayDifferential3f = None,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:

        if not develop:
            raise Exception("develop=True must be specified when "
                            "invoking AD integrators")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.integrator.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aov_names()
            )

            # Generate a set of rays starting at the sensor
            if ray is None:
                ray, _, _ = self.integrator.sample_rays(scene, sensor, sampler)

            # Launch the Monte Carlo sampling process in primal mode
            L, valid, state = self.integrator.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                reparam=None,
                active=mi.Bool(True)
            )

            # Explicitly delete any remaining unused variables
            del sampler, valid
            gc.collect()

        return L

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       ray: mi.RayDifferential3f = None,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:
        raise NotImplementedError("Forward mode not implemented yet")

    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        ray: mi.RayDifferential3f = None,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:
        """
        Customized version of Mitsuba's PRB backward pass that
        supports passing rays to be rendered.
        See documentation of Mitsuba for more details.
        """

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        aovs = self.integrator.aov_names()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.integrator.prepare(sensor, seed, spp, aovs)

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            if ray is None:
                ray, _, _ = self.integrator.sample_rays(scene, sensor, sampler)

            δL = grad_in

            # Launch the Monte Carlo sampling process in primal mode (1)
            L, valid, state_out = self.integrator.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler.clone(),
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                reparam=None,
                active=mi.Bool(True)
            )

            # Launch Monte Carlo sampling in backward AD mode (2)
            L_2, valid_2, state_out_2 = self.integrator.sample(
                mode=dr.ADMode.Backward,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=δL,
                state_in=state_out,
                reparam=None,
                active=mi.Bool(True)
            )

            # We don't need any of the outputs here
            del L_2, valid_2, state_out, state_out_2, δL, sampler

            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()