# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime

import legateboost

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "legate-boost"
copyright = f"2023-{datetime.today().year}, NVIDIA"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["examples"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#

# The short X.Y version.
version = legateboost.__version__

# The full version, including alpha/beta/rc tags.
release = version

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static", "_static/examples"]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "footer_start": ["copyright"],
    # https://github.com/pydata/pydata-sphinx-theme/issues/1220
    "icon_links": [],
    "navbar_align": "left",
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "primary_sidebar_end": ["indices.html"],
    "secondary_sidebar_items": ["page-toc"],
    "show_nav_level": 2,
    "show_toc_level": 2,
}

# -- Options for extensions --------------------------------------------------

autosummary_generate = True

# ensure links to third-party docs work
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
