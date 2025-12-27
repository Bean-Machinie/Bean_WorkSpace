from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..model import PointMass, RigidBody, SimEntity, Vector


class Integrator(Protocol):
    def step(self, sim: "Simulation") -> None:
        ...


@dataclass
class SymplecticEulerIntegrator:
    """Semi-implicit Euler integrator suitable for interactive work."""

    def step(self, sim: "Simulation") -> None:
        for entity in sim.entities:
            if isinstance(entity, PointMass):
                acceleration = sim.acceleration_for(entity)
                entity.velocity = entity.velocity + acceleration * sim.dt
                entity.position = entity.position + entity.velocity * sim.dt
            elif isinstance(entity, RigidBody):
                entity.step_kinematic(sim.dt)


@dataclass
class Simulation:
    entities: list[SimEntity]
    dt: float
    integrator: Integrator
    time: float = 0.0

    def step(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        self.integrator.step(self)
        self.time += self.dt

    def acceleration_for(self, entity: PointMass) -> Vector:
        _ = entity
        # TODO: placeholder acceleration hook until force models are implemented.
        return entity.position * 0.0
