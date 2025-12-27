from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..sim import Simulation


@dataclass(frozen=True)
class ScenarioUIDefaults:
    view_range: tuple[float, float, float, float] | None = None


class Scenario(Protocol):
    scenario_id: str
    name: str

    def create_simulation(self) -> Simulation:
        ...

    def ui_defaults(self) -> ScenarioUIDefaults | None:
        ...
