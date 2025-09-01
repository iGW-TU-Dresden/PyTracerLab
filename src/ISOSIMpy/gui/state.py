from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Store the current state of the app in a centralized fashion

ArrayLike = np.ndarray


@dataclass
class AppState:
    is_monthly: bool = True
    tracer: str = "Tritium"
    input_series: Optional[Tuple[ArrayLike, ArrayLike]] = None
    target_series: Optional[Tuple[ArrayLike, ArrayLike]] = None
    selected_units: List[str] = field(default_factory=list)  # unique registry keys for params tab
    # detailed design: list of (unit_name, fraction) allowing duplicates
    design_units: List[Tuple[str, float]] = field(default_factory=list)
    # aggregated by prefix for controller/model: {"epm": 0.5, ...}
    unit_fractions: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    # params[prefix][key] = {"val":..., "lb":..., "ub":..., "fixed":0/1}
    steady_state_input: float = 0.0
    n_warmup_half_lives: int = 2
    last_simulation: Optional[ArrayLike] = None
    last_times: Optional[ArrayLike] = None
