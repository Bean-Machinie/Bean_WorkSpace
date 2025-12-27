from __future__ import annotations

from typing import Dict, List

from .base import Scenario


class ScenarioRegistry:
    def __init__(self) -> None:
        self._scenarios: Dict[str, Scenario] = {}

    def register(self, scenario: Scenario) -> None:
        if scenario.scenario_id in self._scenarios:
            raise ValueError(f"Duplicate scenario id: {scenario.scenario_id}")
        self._scenarios[scenario.scenario_id] = scenario

    def get(self, scenario_id: str) -> Scenario:
        return self._scenarios[scenario_id]

    def all(self) -> List[Scenario]:
        return list(self._scenarios.values())


scenario_registry = ScenarioRegistry()
