# Scenario File Format

Scenario files are JSON and describe the initial state of a simulation. They are stored with a
`schema_version` to allow forward compatibility.

## Top-level schema (schema_version: 1)
- `schema_version` (int): currently `1`
- `name` (string): scenario display name
- `created_at` / `modified_at` (string, optional): ISO-8601 UTC timestamps
- `simulation` (object)
  - `dt` (number): fixed timestep in seconds
  - `integrator` (string): integrator id (currently `symplectic_euler`)
- `entities` (array): rigid-body and optional point-mass definitions

## Rigid body entity
Each entry in `entities` is a rigid body with:
- `type`: `"rigid_body"` (optional, defaults to rigid body if omitted)
- `entity_id`: string identifier (e.g., `SC-1`)
- `state`:
  - `com_position`: `[x, y, z]`
  - `com_velocity`: `[vx, vy, vz]`
  - `orientation`: `[w, x, y, z]` quaternion
  - `omega_world`: `[wx, wy, wz]`
- `components`: array of mass points in the body frame
  - `component_id`: string (e.g., `C1`)
  - `mass`: number (kg)
  - `position_body`: `[x, y, z]`
- `mesh` (optional): visual-only mesh metadata
  - `path`: string (relative paths preferred, e.g., `assets/models/ship.glb`)
  - `path_is_absolute`: bool (true for external paths)
  - `scale`: `[sx, sy, sz]`
  - `offset_body`: `[x, y, z]`
- `rotation_body`: `[w, x, y, z]` quaternion

## Point mass entity (optional)
- `type`: `"point_mass"`
- `entity_id`: string identifier
- `mass`: number (kg)
- `position`: `[x, y]` or `[x, y, z]` depending on usage
- `velocity`: `[vx, vy]` or `[vx, vy, vz]`

## Notes
- Components are defined in the body frame and are recentered so the body origin matches the CoM.
- Meshes are visual only; physics uses the mass components list.
- Relative mesh paths under `assets/` are recommended for portability.
