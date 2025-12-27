from __future__ import annotations

import numpy as np

from ..model import PointMass
from ..sim import Simulation, SymplecticEulerIntegrator
from .base import ScenarioUIDefaults
from .registry import scenario_registry


class ComSandboxScenario:
    scenario_id = "com_sandbox_discrete"
    name = "Centre of Mass Sandbox (Discrete)"

    def create_simulation(self) -> Simulation:
        entities = [
            PointMass(entity_id="A", mass=3.0, position=np.array([-2.0, 1.0]), velocity=np.array([0.3, 0.0])),
            PointMass(entity_id="B", mass=5.0, position=np.array([1.5, -0.5]), velocity=np.array([-0.2, 0.1])),
            PointMass(entity_id="C", mass=2.0, position=np.array([0.0, 2.5]), velocity=np.array([0.0, -0.15])),
        ]
        return Simulation(entities=entities, dt=0.05, integrator=SymplecticEulerIntegrator())

    def ui_defaults(self) -> ScenarioUIDefaults:
        return ScenarioUIDefaults(view_range=(-5.0, 5.0, -5.0, 5.0))


scenario_registry.register(ComSandboxScenario())
