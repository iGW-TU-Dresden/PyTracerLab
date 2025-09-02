"""Reusable GUI widgets used across tabs (parameter editors)."""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QCheckBox, QGridLayout, QLabel, QLineEdit, QSizePolicy, QWidget


class ParameterEditor(QWidget):
    """Composite editor for a single parameter.

    Renders four controls for lower bound, value, upper bound, and a fixed
    checkbox. The line edits are accessible via attributes ``lb``, ``val``,
    and ``ub`` so that parent layouts can place them in a grid.

    Parameters
    ----------
    prefix : str
        Instance prefix used to namespace parameters (e.g., ``"epm1"``).
    meta : dict
        Parameter metadata with keys ``"key"``, ``"label"``, ``"bounds"``,
        and ``"default"``.
    initial : dict, optional
        Optional initial record with keys ``"val"``, ``"lb"``, ``"ub"``, and
        ``"fixed"``.

    Notes
    -----
    The widget is self-contained but can also be embedded into an external
    grid via its exposed sub-widgets.
    """

    def __init__(self, prefix: str, meta: dict, initial: dict | None = None, parent=None):
        super().__init__(parent)
        self.prefix = prefix
        self.key = meta["key"]
        lb, ub = meta["bounds"]
        init = initial or {"val": meta["default"], "lb": lb, "ub": ub, "fixed": False}

        # Validator: float, right-aligned entries
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setDecimals(12)  # generous; UI can show fewer

        # Consistent width hint so all rows look aligned
        # (Let the external grid handle the final column width; we set a reasonable min.)
        probe = QLineEdit()
        fm = probe.fontMetrics()
        minw = fm.horizontalAdvance(" -12345.123456 ") + 12

        grid = QGridLayout(self)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(0)
        grid.setContentsMargins(0, 0, 0, 0)

        # Inline label (kept for backward-compat; external grids can ignore this widget)
        title = QLabel(f"{prefix.upper()} — {meta.get('label', self.key)}")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(title, 0, 0, alignment=Qt.AlignLeft)

        # Editors
        self.lb = QLineEdit(str(init["lb"]))
        self.lb.setAlignment(Qt.AlignRight)
        self.lb.setValidator(validator)
        self.lb.setMinimumWidth(minw)
        self.lb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.val = QLineEdit(str(init["val"]))
        self.val.setAlignment(Qt.AlignRight)
        self.val.setValidator(validator)
        self.val.setMinimumWidth(minw)
        self.val.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.ub = QLineEdit(str(init["ub"]))
        self.ub.setAlignment(Qt.AlignRight)
        self.ub.setValidator(validator)
        self.ub.setMinimumWidth(minw)
        self.ub.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.fixed = QCheckBox("Fixed")
        self.fixed.setChecked(bool(init.get("fixed", False)))

        # Default internal layout (still works if someone adds ParameterEditor as a single widget)
        # External table can ignore this and use .lb/.val/.ub directly.
        grid.addWidget(self.lb, 0, 1)
        grid.addWidget(self.val, 0, 2)
        grid.addWidget(self.ub, 0, 3)
        grid.addWidget(self.fixed, 0, 4)

    def to_dict(self):
        """Return the current editor state as a serializable dict."""

        # Convert safely; fall back to current text->float conversion
        def _f(edit: QLineEdit):
            txt = edit.text().strip()
            return float(txt) if txt else 0.0

        return {
            "val": _f(self.val),
            "lb": _f(self.lb),
            "ub": _f(self.ub),
            "fixed": self.fixed.isChecked(),
        }
