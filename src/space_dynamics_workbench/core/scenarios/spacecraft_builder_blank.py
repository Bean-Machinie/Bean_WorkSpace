from __future__ import annotations

import numpy as np

from ..model import RigidBody, RigidBodyComponent
from ..sim import Simulation, SymplecticEulerIntegrator
from .base import ScenarioUIDefaults
from .registry import scenario_registry


class SpacecraftBuilderBlankScenario:
    scenario_id = "spacecraft_builder_blank"
    name = "Spacecraft Builder (Blank)"

    def create_simulation(self) -> Simulation:
        components = [RigidBodyComponent(component_id="C1", mass=1.0, position_body=np.zeros(3))]
        rigid_body = RigidBody(
            entity_id="SC-Builder",
            components=components,
            com_position=np.zeros(3, dtype=float),
            com_velocity=np.zeros(3, dtype=float),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            omega_world=np.zeros(3, dtype=float),
        )
        return Simulation(entities=[rigid_body], dt=0.05, integrator=SymplecticEulerIntegrator())

    def ui_defaults(self) -> ScenarioUIDefaults:
        return ScenarioUIDefaults(view_range=(-6.0, 6.0, -5.0, 5.0))


scenario_registry.register(SpacecraftBuilderBlankScenario())
