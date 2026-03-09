from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

project = "PyTracerLab"
author = "Max G. Rudolph"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
]

# notebook execution (safe default for CI)
nb_execution_mode = "off"  # "cache" or "auto" for execution enabled
nb_execution_timeout = 300

# mock heavy deps during doc builds
autodoc_mock_imports = ["PyQt5", "numpy", "scipy", "matplotlib"]

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
# Use fully qualified paths for type-hint cross references to avoid
# ambiguous short names like "Model" and "Unit".
autodoc_typehints_format = "fully-qualified"
autodoc_type_aliases = {
    "Model": "PyTracerLab.model.model.Model",
    "Unit": "PyTracerLab.model.units.Unit",
}

napoleon_google_docstring = False  # True if using Google style
napoleon_numpy_docstring = True  # True if using NumPy style
# Render class attribute docs from "Attributes" sections as plain ivar fields
# (prevents duplicate attribute targets alongside autodoc's member listing)
napoleon_use_ivar = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "attrs_inline",
    "attrs_block",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/.ipynb_checkpoints"]

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_title = "PyTracerLab Documentation"
html_short_title = "PyTracerLab"
html_show_sphinx = True
html_show_copyright = True

# html_theme_options = {
#     "use_edit_page_button": True,
#     "header_links_before_dropdown": 6,
#     "icon_links": [
#         {
#             "name": "GitHub",
#             "url": "https://github.com/iGW-TU-Dresden/PyTracerLab",
#             "icon": "fab fa-github-square",
#             "type": "fontawesome",
#         }
#     ],
# }

# use None for inventories (Sphinx 8+)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# do not document re-exported names (prevents duplicate objects)
autosummary_imported_members = False

# Disambiguate short type names in NumPy-style docstrings
napoleon_type_aliases = {
    "Unit": "PyTracerLab.model.units.Unit",
    "Model": "PyTracerLab.model.model.Model",
    "Solver": "PyTracerLab.model.solver.Solver",
}

# Skip package-level re-exports in ``PyTracerLab.model`` to avoid duplicate
# Python domain targets (e.g., both ``PyTracerLab.model.Model`` and
# ``PyTracerLab.model.model.Model``).
_MODEL_PACKAGE_REEXPORTS = {"Model", "Unit", "EPMUnit", "EMUnit", "PMUnit", "DMUnit"}


def _skip_model_reexports(app, what, name, obj, skip, options):
    env = getattr(app, "env", None)
    temp_data = getattr(env, "temp_data", {}) if env is not None else {}
    current_module = temp_data.get("autodoc:module")
    if what == "module" and current_module == "PyTracerLab.model":
        if name in _MODEL_PACKAGE_REEXPORTS:
            return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", _skip_model_reexports)
