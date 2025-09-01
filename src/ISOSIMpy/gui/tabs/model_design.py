from functools import partial

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtWidgets import (
    QSizePolicy as SP,
)


class ModelDesignTab(QWidget):
    selection_changed = pyqtSignal()

    def __init__(self, state, registry, parent=None):
        super().__init__(parent)
        self.state = state
        self.registry = registry
        # rows of (combobox, fraction editor)
        self.rows = []

        # ensure attributes exist on state
        if not hasattr(self.state, "unit_fractions"):
            self.state.unit_fractions = {}
        if not hasattr(self.state, "steady_state_input"):
            self.state.steady_state_input = 0.0
        if not hasattr(self.state, "steady_state_enabled"):
            self.state.steady_state_enabled = False
        if not hasattr(self.state, "n_warmup_half_lives"):
            self.state.n_warmup_half_lives = 0.0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        # Title
        title = QLabel("Select up to 4 units and set fractions:")
        title_font = QFont(title.font())
        title_font.setBold(True)
        title.setFont(title_font)
        outer.addWidget(title)

        # Main grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        outer.addLayout(grid)

        ### Unit section headers
        hdr_unit = QLabel("Unit")
        hdr_frac = QLabel("Fraction")
        hdr_unit.setStyleSheet("font-weight: 600;")
        hdr_frac.setStyleSheet("font-weight: 600;")
        grid.addWidget(hdr_unit, 0, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(hdr_frac, 0, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 0)

        # Float validator
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setDecimals(6)

        # Width probe
        probe = QLineEdit()
        fm = probe.fontMetrics()
        frac_width = fm.horizontalAdvance(" -0.0000 ") + 18

        ### Unit rows (up to 4 selections)
        row = 1
        unit_names = list(self.registry.keys())
        placeholder = "— Select —"
        for i in range(4):
            combo = QComboBox(self)
            combo.addItem(placeholder, userData=None)
            for nm in unit_names:
                combo.addItem(nm, userData=nm)

            fx = QLineEdit(self)
            fx.setText("0.0000")
            fx.setAlignment(Qt.AlignRight)
            fx.setValidator(validator)
            fx.setMaximumWidth(frac_width)
            fx.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            fx.setEnabled(False)

            grid.addWidget(combo, row, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
            grid.addWidget(fx, row, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)

            combo.currentIndexChanged.connect(
                partial(self._on_combo_changed, combo=combo, frac_edit=fx)
            )
            fx.textChanged.connect(self._update)

            self.rows.append({"combo": combo, "frac": fx})
            row += 1

        # spacer between sections
        grid.addItem(QSpacerItem(0, 10, SP.Minimum, SP.Minimum), row, 0)
        row += 1

        ### Steady-state section headers
        hdr_ss = QLabel("Steady-State Input")
        hdr_val = QLabel("Value")
        hdr_ss.setStyleSheet("font-weight: 600;")
        hdr_val.setStyleSheet("font-weight: 600;")
        grid.addWidget(hdr_ss, row, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(hdr_val, row, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        row += 1

        # Steady-state controls
        self.ss_checkbox = QCheckBox("", self)
        self.ss_checkbox.setChecked(bool(self.state.steady_state_enabled))

        self.ss_value = QLineEdit(self)
        self.ss_value.setText(f"{float(self.state.steady_state_input):.4f}")
        self.ss_value.setAlignment(Qt.AlignRight)
        self.ss_value.setValidator(validator)
        self.ss_value.setMaximumWidth(frac_width)
        self.ss_value.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ss_value.setEnabled(self.ss_checkbox.isChecked())

        grid.addWidget(self.ss_checkbox, row, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(self.ss_value, row, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        self.ss_checkbox.toggled.connect(self._on_ss_toggle)
        self.ss_value.textChanged.connect(self._update)
        row += 1

        # spacer
        grid.addItem(QSpacerItem(0, 10, SP.Minimum, SP.Minimum), row, 0)
        row += 1

        ### Warmup half lives header
        hdr_warm = QLabel("Warmup half lives")
        hdr_warm.setStyleSheet("font-weight: 600;")
        grid.addWidget(hdr_warm, row, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        row += 1

        # Warmup field (no checkbox, always enabled)
        self.warmup_value = QLineEdit(self)
        self.warmup_value.setText(f"{int(self.state.n_warmup_half_lives)}")
        self.warmup_value.setAlignment(Qt.AlignRight)
        self.warmup_value.setValidator(validator)
        self.warmup_value.setMaximumWidth(frac_width)
        self.warmup_value.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        grid.addWidget(self.warmup_value, row, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        self.warmup_value.textChanged.connect(self._update)

        # Try to restore previous design if present
        if hasattr(self.state, "design_units") and self.state.design_units:
            for (name, frac), r in zip(self.state.design_units, self.rows):
                # set combo selection
                idx = r["combo"].findData(name)
                r["combo"].setCurrentIndex(idx if idx != -1 else 0)
                # set fraction and enable
                r["frac"].setText(f"{float(frac):.4f}")
                r["frac"].setEnabled(idx != -1 and idx != 0)

        self._update()

    def _on_combo_changed(self, index: int, combo: QComboBox, frac_edit: QLineEdit):
        # Enable fraction only when a real unit is selected
        selected = combo.currentData()
        frac_edit.setEnabled(selected is not None)
        self._update()

    def _on_ss_toggle(self, checked: bool):
        self.ss_value.setEnabled(checked)
        self.state.steady_state_enabled = bool(checked)
        self._update()

    def _update(self):
        # Gather selected units (up to 4) and their fractions
        design_units = []  # list of (unit_name, fraction)
        for r in self.rows:
            name = r["combo"].currentData()
            if name is None:
                continue
            try:
                frac = float(r["frac"].text()) if r["frac"].text() else 0.0
            except ValueError:
                frac = 0.0
            design_units.append((name, frac))

        # Persist basic list
        self.state.design_units = design_units

        # Build per-instance descriptors with unique prefixes (e.g., pm1, pm2, ...)
        counts: dict[str, int] = {}
        instances = []
        for name, frac in design_units:
            counts[name] = counts.get(name, 0) + 1
            cls = self.registry[name]
            base = getattr(cls, "PREFIX", name.lower())
            inst_prefix = f"{base}{counts[name]}"
            instances.append({"name": name, "prefix": inst_prefix, "fraction": float(frac)})
        self.state.design_instances = instances

        # Maintain selected_units for any legacy consumers (unique types)
        seen = set()
        unique_units = []
        for name, _ in design_units:
            if name not in seen:
                seen.add(name)
                unique_units.append(name)
        self.state.selected_units = unique_units

        # Fractions per instance prefix for controller
        self.state.unit_fractions = {inst["prefix"]: inst["fraction"] for inst in instances}

        # Steady-state
        if self.ss_checkbox.isChecked():
            try:
                self.state.steady_state_input = (
                    float(self.ss_value.text()) if self.ss_value.text() else 0.0
                )
            except ValueError:
                pass

        # Warmup half lives
        try:
            self.state.n_warmup_half_lives = (
                int(self.warmup_value.text()) if self.warmup_value.text() else 0.0
            )
        except ValueError:
            pass

        self.selection_changed.emit()
