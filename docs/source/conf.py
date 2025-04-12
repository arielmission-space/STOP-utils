# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import date

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, os.path.abspath("../../"))  # Add project root to path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "stop-utils"
copyright = f"2025-{date.today().year:d}, arielmission-space"

import importlib.metadata as metadata

version = metadata.version(project)
author = metadata.metadata(project)["Author"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Include documentation from docstrings
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.githubpages",  # Support for GitHub Pages
    "sphinx_autodoc_typehints",  # Automatically document typehints
    "myst_parser",  # Support for Markdown files
    "sphinxcontrib.mermaid",  # Support for Mermaid diagrams
]

# Configure mermaid-js output
mermaid_init_js = """
    mermaid.initialize({
        startOnLoad: true,
        theme: 'default',
        securityLevel: 'loose',
    });
"""

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress specific warnings
suppress_warnings = [
    "myst.xref_missing",  # Ignore missing cross-reference for LICENSE in README
    "pygments",  # Ignore Pygments lexer warnings (like for 'mermaid')
]

# -- Options for MyST Parser -------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/configuration.html
myst_enable_extensions = [
    "colon_fence",  # Enable ::: fenced code blocks
    "dollarmath",  # Enable inline and block math with $ and $$
    "amsmath",  # Enable advanced math
    "deflist",  # Enable definition lists
    "html_image",  # Enable HTML image tags
]

# Configure MyST-Parser to handle code blocks
myst_fence_as_directive = ["mermaid"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 2,  # Show two levels in the sidebar
    "collapse_navigation": False,  # Don't collapse sidebar sections
    "sticky_navigation": True,
    "titles_only": False,  # Explicitly keep default behavior (toctree titles can differ from page titles)
}

# -- Options for autodoc -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

autodoc_member_order = "bysource"  # Order members by source code order
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Options for Napoleon ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for intersphinx -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "photutils": ("https://photutils.readthedocs.io/en/stable/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "paos": ("https://paos.readthedocs.io/en/latest/", None),
    "typer": ("https://typer.tiangolo.com/", None),
    "rich": ("https://rich.readthedocs.io/en/latest/", None),
    "loguru": ("https://loguru.readthedocs.io/en/stable/", None),
}

# -- Options for sphinx-autodoc-typehints ------------------------------------
# https://github.com/tox-dev/sphinx-autodoc-typehints#configuration

always_document_param_types = True
typehints_fully_qualified = False
