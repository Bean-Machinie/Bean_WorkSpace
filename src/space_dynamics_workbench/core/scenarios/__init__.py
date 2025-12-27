from .base import Scenario, ScenarioUIDefaults
from .registry import ScenarioRegistry, scenario_registry


def load_builtin_scenarios() -> None:
    # Import side effects to register built-in scenarios.
    from . import com_sandbox  # noqa: F401
    from . import rigid_body_sandbox  # noqa: F401
    from . import spacecraft_builder_blank  # noqa: F401


__all__ = [
    "Scenario",
    "ScenarioUIDefaults",
    "ScenarioRegistry",
    "scenario_registry",
    "load_builtin_scenarios",
]
