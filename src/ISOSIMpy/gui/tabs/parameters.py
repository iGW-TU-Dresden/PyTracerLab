from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .widgets import ParameterEditor


class ParametersTab(QWidget):
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

        self.editors = []
        self.refresh()

    def _clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def refresh(self):
        self._clear_grid()
        self.editors.clear()

        instances = list(getattr(self.state, "design_instances", []) or [])
        if not instances:
            hint = QLabel(
                "No units selected. Select units in the Model Design tab to edit their parameters."
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #666;")
            self.grid.addWidget(hint, 0, 0, 1, 5, alignment=Qt.AlignLeft | Qt.AlignTop)
            self.ready.emit()
            return

        ### Column headers
        headers = ["Unit / Name", "Lower Bound", "Value", "Upper Bound", "Fixed"]
        for col, text in enumerate(headers):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: 600;")
            self.grid.addWidget(lbl, 0, col, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        # Ensure consistent sizing
        self.grid.setColumnStretch(0, 0)  # label
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnStretch(2, 1)
        self.grid.setColumnStretch(3, 1)
        self.grid.setColumnStretch(4, 0)  # checkbox

        row = 1
        # Count instances per unit type to display ordinal labels
        counts: dict[str, int] = {}
        for inst in instances:
            unit_name = inst["name"]
            prefix = inst["prefix"]
            counts[unit_name] = counts.get(unit_name, 0) + 1
            ordinal = counts[unit_name]

            cls = self.registry[unit_name]
            self.state.params.setdefault(prefix, {})

            for meta in getattr(cls, "PARAMS", []):
                key = meta["key"]
                initial = self.state.params[prefix].get(key)

                # Row label includes instance number for clarity
                pname = meta.get("label", key)
                row_label = QLabel(f"{unit_name} ({ordinal}) - {pname}")
                row_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                self.grid.addWidget(row_label, row, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)

                # Editor bound to the instance prefix
                ed = ParameterEditor(prefix, meta, initial)

                # Place fields in strict columns
                self.grid.addWidget(ed.lb, row, 1)
                self.grid.addWidget(ed.val, row, 2)
                self.grid.addWidget(ed.ub, row, 3)
                self.grid.addWidget(ed.fixed, row, 4, alignment=Qt.AlignCenter)

                self.editors.append(ed)
                row += 1

        self.ready.emit()

    def commit(self):
        for ed in self.editors:
            self.state.params.setdefault(ed.prefix, {})
            self.state.params[ed.prefix][ed.key] = ed.to_dict()
