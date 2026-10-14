# Sphinx configuration for the quflow documentation.
import os
import re
import sys

sys.path.insert(0, os.path.abspath(".."))

with open(os.path.join("..", "quflow", "__init__.py"), encoding="utf-8") as f:
    release = re.search(r"__version__\s*=\s*['\"]([^'\"]*)['\"]", f.read()).group(1)
version = release

project = "quflow"
author = "Klas Modin"
copyright = "Klas Modin"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
]

autosummary_generate = True
autodoc_mock_imports = ["cupy", "cartopy", "pyssht"]
numpydoc_show_class_members = False
numpydoc_xref_param_type = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"quflow {release}"
