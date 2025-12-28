import numpy as np

from space_dynamics_workbench.core.io.scenario_io import (
    deserialize_scenario_definition,
    serialize_scenario_definition,
)
from space_dynamics_workbench.core.model import MeshMetadata
from space_dynamics_workbench.core.scenario_definition import (
    RigidBodyComponentDefinition,
    RigidBodyDefinition,
    RigidBodyStateDefinition,
    ScenarioDefinition,
    SimulationDefinition,
    simulation_from_definition,
)


def test_scenario_roundtrip() -> None:
    components = [
        RigidBodyComponentDefinition("C1", 1.0, np.array([0.0, 0.0, 0.0])),
        RigidBodyComponentDefinition("C2", 2.0, np.array([1.0, 0.0, 0.0])),
    ]
    state = RigidBodyStateDefinition(
        com_position=np.array([1.0, 2.0, 3.0]),
        com_velocity=np.array([0.1, 0.2, 0.3]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_world=np.array([0.01, 0.02, 0.03]),
    )
    mesh = MeshMetadata(
        path="assets/models/test.glb",
        path_is_absolute=False,
        scale=np.array([1.0, 1.0, 1.0]),
        offset_body=np.array([0.1, 0.2, 0.3]),
        rotation_body=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    definition = ScenarioDefinition(
        name="Test Scenario",
        simulation=SimulationDefinition(dt=0.1, integrator="symplectic_euler"),
        entities=[RigidBodyDefinition(entity_id="SC-1", components=components, state=state, mesh=mesh)],
    )

    payload = serialize_scenario_definition(definition)
    loaded = deserialize_scenario_definition(payload)

    assert loaded.name == definition.name
    assert loaded.simulation.dt == definition.simulation.dt
    assert loaded.simulation.integrator == definition.simulation.integrator
    assert len(loaded.entities) == 1
    loaded_entity = loaded.entities[0]
    assert loaded_entity.entity_id == "SC-1"
    assert len(loaded_entity.components) == 2
    assert np.allclose(loaded_entity.state.com_position, state.com_position)
    assert np.allclose(loaded_entity.state.com_velocity, state.com_velocity)
    assert np.allclose(loaded_entity.state.orientation, state.orientation)
    assert np.allclose(loaded_entity.state.omega_world, state.omega_world)
    assert loaded_entity.mesh is not None
    assert loaded_entity.mesh.path == mesh.path
    assert loaded_entity.mesh.path_is_absolute == mesh.path_is_absolute
    assert np.allclose(loaded_entity.mesh.offset_body, mesh.offset_body)


def test_simulation_instantiation() -> None:
    definition = ScenarioDefinition(
        name="Instantiate",
        simulation=SimulationDefinition(dt=0.05, integrator="symplectic_euler"),
        entities=[
            RigidBodyDefinition(
                entity_id="SC-1",
                components=[RigidBodyComponentDefinition("C1", 1.0, np.zeros(3))],
                state=RigidBodyStateDefinition(),
                mesh=None,
            )
        ],
    )
    sim = simulation_from_definition(definition)
    assert sim.dt == 0.05
    assert len(sim.entities) == 1
    entity = sim.entities[0]
    assert entity.entity_id == "SC-1"
    assert entity.components[0].component_id == "C1"
