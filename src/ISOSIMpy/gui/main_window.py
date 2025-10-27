"""Main window for the ISOSIMpy GUI, assembling all tabs."""

from PyQt5.QtWidgets import QMessageBox, QTabWidget, QVBoxLayout, QWidget

from ..model.registry import UNIT_REGISTRY
from .controller import Controller
from .state import AppState
from .tabs.file_input import FileInputTab
from .tabs.model_design import ModelDesignTab
from .tabs.parameters import ParametersTab
from .tabs.simulation import SimulationTab
from .tabs.tracer_tracer import TracerTracerTab


class MainWindow(QWidget):
    """Top-level application window hosting the main tabs."""

    def __init__(self):
        """Create the main window and wire up tabs and controller."""
        super().__init__()
        # Initialize the window
        self.setWindowTitle("ISOSIMpy")
        self.resize(600, 800)

        self.state = AppState()
        self.ctrl = Controller(self.state)

        tabs = QTabWidget()
        t1 = FileInputTab(self.state)
        t2 = ModelDesignTab(self.state, UNIT_REGISTRY)
        t3 = ParametersTab(self.state, UNIT_REGISTRY)
        t4 = SimulationTab(self.state)
        t5 = TracerTracerTab(self.state, UNIT_REGISTRY)

        tabs.addTab(t1, "[1] Input")
        tabs.addTab(t2, "[2] Model")
        tabs.addTab(t3, "[3] Parameters")
        tabs.addTab(t4, "[4] Simulation")
        tabs.addTab(t5, "[5] Tracer-Tracer")

        lay = QVBoxLayout(self)
        lay.addWidget(tabs)

        # wiring
        t1.changed.connect(t2.refresh_tracer_inputs)
        t1.changed.connect(t3.refresh)
        t1.changed.connect(t5.reset_results)
        t1.changed.connect(t5.refresh)
        t2.selection_changed.connect(t3.refresh)
        t2.selection_changed.connect(t5.refresh)
        t2.selection_changed.connect(t5.reset_results)
        t4.simulate_requested.connect(lambda: (t3.commit(), self.ctrl.simulate()))
        t4.calibrate_requested.connect(lambda: (t3.commit(), self.ctrl.calibrate()))
        t4.report_requested.connect(lambda fname: (t3.commit(), self.ctrl.write_report(fname)))
        t5.sweep_requested.connect(
            lambda start, stop, count, key: (
                t3.commit(),
                self.ctrl.run_tracer_tracer(start, stop, count, key),
            )
        )

        self.ctrl.simulated.connect(t4.show_results)
        self.ctrl.calibrated.connect(t4.show_results)
        self.ctrl.tracer_tracer_ready.connect(t5.handle_tracer_tracer_ready)
        self.ctrl.status.connect(t4.show_status)

        def _show_error(msg):
            t5.sweep_failed()
            QMessageBox.critical(self, "Error", msg)

        self.ctrl.error.connect(_show_error)
