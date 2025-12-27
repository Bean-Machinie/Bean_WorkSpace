from __future__ import annotations

import numpy as np

from ..model import RigidBody, RigidBodyComponent
from ..sim import Simulation, SymplecticEulerIntegrator
from .base import ScenarioUIDefaults
from .registry import scenario_registry


class RigidBodySandboxScenario:
    scenario_id = "rigid_body_sandbox_kinematic"
    name = "Rigid Spacecraft Sandbox (Kinematic)"

    def create_simulation(self) -> Simulation:
        raw_components = [
            ("Bus", 3.0, np.array([1.4, 0.4, 0.0])),
            ("Tank", 2.2, np.array([-1.1, -0.3, 0.0])),
            ("Antenna", 1.1, np.array([0.3, 1.3, 0.0])),
            ("Panel-L", 0.8, np.array([-0.7, 0.9, 0.0])),
            ("Panel-R", 0.9, np.array([0.6, -0.8, 0.0])),
        ]
        masses = np.array([mass for _, mass, _ in raw_components], dtype=float)
        positions = np.stack([pos for _, _, pos in raw_components])
        com_body = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
        components = [
            RigidBodyComponent(
                component_id=comp_id,
                mass=mass,
                position_body=pos - com_body,
            )
            for comp_id, mass, pos in raw_components
        ]
        angle = np.deg2rad(25.0)
        orientation = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)], dtype=float)
        rigid_body = RigidBody(
            entity_id="SC-1",
            components=components,
            com_position=np.array([-3.0, 1.0, 0.0], dtype=float),
            com_velocity=np.array([0.35, -0.05, 0.0], dtype=float),
            orientation=orientation,
            omega_world=np.array([0.0, 0.0, 0.6], dtype=float),
        )
        return Simulation(entities=[rigid_body], dt=0.05, integrator=SymplecticEulerIntegrator())

    def ui_defaults(self) -> ScenarioUIDefaults:
        return ScenarioUIDefaults(view_range=(-8.0, 8.0, -6.0, 6.0))


scenario_registry.register(RigidBodySandboxScenario())
