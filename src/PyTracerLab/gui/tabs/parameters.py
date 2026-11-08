"""Tab that renders and persists unit parameter editors."""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ...model.model import FRACTION
from .widgets import ParameterEditor

# Non-blocking hint when a binary mixture may be non-unique (D17)
CORRELATION_HINT = (
    "The mixing fraction and the mean travel times of both units are free. These "
    "parameters are often strongly correlated, so that different combinations fit the "
    "observations equally well. Unless several tracers with contrasting input histories "
    "or a long time series are available, consider fixing the mean travel time of one "
    "unit (cf. Jurgens et al., 2012)."
)


class ParametersTab(QWidget):
    """Tab that renders parameter editors for each selected unit instance.

    For a binary mixture (exactly two units), an additional row holds the
    mixing fraction of the first unit (bounds and "Fixed"; the value is set in
    the Model tab).

    Signals
    -------
    ready
        Emitted after the grid is built or refreshed.
    """

    # Define signals that this tab can emit
    ready = pyqtSignal()

    def __init__(self, state, registry, parent=None):
        super().__init__(parent)
        self.state = state
        self.registry = registry

        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(6)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        lay.addLayout(self.grid)

        self.hint = QLabel(CORRELATION_HINT)
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #8a5a00;")
        self.hint.setVisible(False)
        lay.addWidget(self.hint)
        lay.addStretch(1)

        self.editors = []
        self.fraction_editor = None
        self.refresh()

    def _clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def refresh(self):
        """Rebuild the parameter grid from current design instances."""
        # keep edits that were not yet committed (O.25)
        self.commit()
        self._clear_grid()
        self.editors.clear()
        self.fraction_editor = None

        instances = list(getattr(self.state, "design_instances", []) or [])
        if not instances:
            hint = QLabel(
                "No units selected. Select units in the Model Design tab to edit their parameters."
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #666;")
            self.grid.addWidget(hint, 0, 0, 1, 5, alignment=Qt.AlignLeft | Qt.AlignTop)
            self._update_hint()
            self.ready.emit()
            return

        ### Column headers
        headers = ["Unit / Name", "Lower Bound", "Initial Value", "Upper Bound", "Fixed"]
        for col, text in enumerate(headers):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: 600;")
            self.grid.addWidget(lbl, 0, col, alignment=Qt.AlignHCenter | Qt.AlignBottom)

        # Ensure consistent sizing
        self.grid.setColumnStretch(0, 0)  # label
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnStretch(2, 1)
        self.grid.setColumnStretch(3, 1)
        self.grid.setColumnStretch(4, 0)  # checkbox

        row = 1
        # Count instances per unit type to display ordinal labels
        counts: dict[str, int] = {}
        labels: list[str] = []
        for inst in instances:
            # Get properties of unit instances (name, prefix, etc.)
            unit_name = inst["name"]
            prefix = inst["prefix"]
            counts[unit_name] = counts.get(unit_name, 0) + 1
            ordinal = counts[unit_name]
            labels.append(f"{unit_name} ({ordinal})")

            cls = self.registry[unit_name]
            self.state.params.setdefault(prefix, {})

            for meta in getattr(cls, "PARAMS", []):
                key = meta["key"]
                initial = self.state.params[prefix].get(key)

                # Row label includes instance number for clarity
                pname = meta.get("label", key)

                # Attach units if possible
                # We use the literal parameter keys from the PyTracerLab units to
                # match the parameters
                if key == "mtt":
                    pname += " [years]"
                elif key == "eta":
                    pname += " [-]"
                elif key == "exp_part":
                    pname += " [volume]"
                elif key == "piston_part":
                    pname += " [volume]"
                elif key == "DP":
                    pname += " [-]"
                elif key == "alpha":
                    pname += " [-]"
                elif key in ("above", "screen", "below"):
                    pname += " [length]"

                row_label = QLabel(f"{unit_name} ({ordinal}) - {pname}")
                row_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                self.grid.addWidget(row_label, row, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)

                # Editor bound to the instance prefix
                ed = ParameterEditor(prefix, meta, initial)
                self._place(ed, row)
                if key == "mtt":
                    ed.fixed.toggled.connect(self._update_hint)
                row += 1

        if len(instances) == 2:
            row = self._add_fraction_row(instances[0], labels, row)

        self._update_hint()
        self.ready.emit()

    def _place(self, ed: ParameterEditor, row: int) -> None:
        """Place the fields of an editor in strict grid columns."""
        self.grid.addWidget(ed.lb, row, 1)
        self.grid.addWidget(ed.val, row, 2)
        self.grid.addWidget(ed.ub, row, 3)
        self.grid.addWidget(ed.fixed, row, 4, alignment=Qt.AlignCenter)
        self.editors.append(ed)

    def _add_fraction_row(self, first: dict, labels: list, row: int) -> int:
        """Row for the mixing fraction of the first unit of a binary mixture."""
        prefix = first["prefix"]
        frac = float(first.get("fraction", 0.0))
        stored = self.state.params.setdefault(prefix, {}).get(FRACTION)
        # bounds and "Fixed" are kept; the value always comes from the Model tab
        initial = dict(stored) if stored else {"lb": 0.0, "ub": 1.0, "fixed": True}
        initial["val"] = frac
        meta = {"key": FRACTION, "label": "mixing fraction", "bounds": (0.0, 1.0), "default": frac}

        row_label = QLabel(f"{labels[0]} - mixing fraction [-]")
        row_label.setToolTip(f"{labels[1]} contributes 1 minus this fraction.")
        self.grid.addWidget(row_label, row, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        ed = ParameterEditor(prefix, meta, initial)
        ed.val.setReadOnly(True)
        ed.val.setEnabled(False)
        ed.val.setToolTip("Set in the Model tab.")
        ed.fixed.toggled.connect(self._update_hint)
        self._place(ed, row)
        self.fraction_editor = ed
        return row + 1

    def _update_hint(self, *_):
        """Show the correlation hint if the fraction and both mean travel times are free."""
        show = False
        if self.fraction_editor is not None and not self.fraction_editor.fixed.isChecked():
            mtt_free = [ed for ed in self.editors if ed.key == "mtt" and not ed.fixed.isChecked()]
            show = len(mtt_free) >= 2
        self.hint.setVisible(show)

    def commit(self):
        """Persist edited parameter values/bounds/fixed flags into state.

        Editors with an incomplete number (e.g. a lone ``"-"`` while typing)
        keep their previously stored values.
        """
        for ed in self.editors:
            try:
                rec = ed.to_dict()
            except ValueError:
                continue
            self.state.params.setdefault(ed.prefix, {})
            self.state.params[ed.prefix][ed.key] = rec
