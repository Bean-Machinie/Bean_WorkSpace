# Space Dynamics Workbench

A physics-first workspace for spacecraft dynamics visualizations. The UI only displays and edits state; the physics core owns all state evolution.

Guiding principles
- Physics-first: simulation state is owned by the core, UI only observes and edits.
- Separation of concerns: core has no Qt/pyqtgraph dependencies.
- Deterministic and testable: fixed-step integration, core unit tests.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

Run the app
```bash
python -m space_dynamics_workbench.app.main
```

## Visualizing frames and vectors
- Use the "Reference Frame" dropdown to switch between World (O) and Center of Mass (C).
- Toggle r_OP, r_OC, and r_CP overlays to visualize the textbook vectors.
- In CoM frame, points are rendered relative to C (C is at the origin; O is shown as a marker).

## Interaction controls
- Pan the camera: hold Space and drag with LMB.
- Move a point: hold Ctrl and drag with LMB near a point.

Run tests
```bash
pytest
```

## Launching via double-click (Windows)
Prerequisites: the `.venv` exists and dependencies are installed (see Setup).

- Double-click `run_app.bat` to launch the app.
- If it reports `.venv` missing, run:
  `py -m venv .venv`
  `.\.venv\Scripts\Activate.ps1`
  `pip install -e .[dev]`
- Optional PowerShell launcher:
  `powershell -ExecutionPolicy Bypass -File .\run_app.ps1`

Note: the launcher keeps the console open only when an error occurs, so you can read messages.

## Adding a new scenario
1) Create a new module in `src/space_dynamics_workbench/core/scenarios/` implementing the Scenario protocol.
2) Register it with the scenario registry.
3) Expose UI defaults (view range) if needed.
4) Load built-in scenarios in the app by importing your module in `core/scenarios/__init__.py`.

## Units and coordinates
- Units are SI by convention (meters, kilograms, seconds).
- 2D vectors are stored as N-D arrays to allow 3D expansion later.
