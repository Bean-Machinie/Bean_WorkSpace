from __future__ import annotations

from typing import Iterable

import numpy as np
from PySide6 import QtWidgets

from ...core.model import PointMass
from ...core.physics import (
    center_of_mass,
    center_of_velocity,
    invariant_position_sum,
    invariant_velocity_sum,
    total_mass,
)


class InvariantsPanel(QtWidgets.QGroupBox):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Invariants / Truth Meter", parent)
        layout = QtWidgets.QFormLayout(self)

        self._total_mass = QtWidgets.QLabel("-")
        self._com = QtWidgets.QLabel("-")
        self._cov = QtWidgets.QLabel("-")
        self._pos_invariant = QtWidgets.QLabel("-")
        self._vel_invariant = QtWidgets.QLabel("-")

        layout.addRow("Total Mass", self._total_mass)
        layout.addRow("r_C", self._com)
        layout.addRow("v_C", self._cov)
        layout.addRow("||sum(m r_CP)||", self._pos_invariant)
        layout.addRow("||sum(m v_CP)||", self._vel_invariant)

    def update_values(self, entities: Iterable[PointMass]) -> None:
        entities_list = list(entities)
        if not entities_list:
            self._total_mass.setText("-")
            self._com.setText("-")
            self._cov.setText("-")
            self._pos_invariant.setText("-")
            self._vel_invariant.setText("-")
            return

        total = total_mass(entities_list)
        com = center_of_mass(entities_list)
        cov = center_of_velocity(entities_list)
        pos_inv = np.linalg.norm(invariant_position_sum(entities_list))
        vel_inv = np.linalg.norm(invariant_velocity_sum(entities_list))

        self._total_mass.setText(f"{total:.3f}")
        self._com.setText(f"[{com[0]:.3f}, {com[1]:.3f}]")
        self._cov.setText(f"[{cov[0]:.3f}, {cov[1]:.3f}]")
        self._pos_invariant.setText(f"{pos_inv:.6f}")
        self._vel_invariant.setText(f"{vel_inv:.6f}")
