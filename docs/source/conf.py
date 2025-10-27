# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Autodoc configuration ---------------------------------------------------
# Autogenerate documentation from the comments of the code
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "surface-sim"
copyright = "2024, Marc Serra Peralta"
author = "Marc Serra Peralta"
release = "0.10.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # autogenerate documentation
    "sphinx.ext.autosummary",  # autogenerate documentation rst files
    "sphinx.ext.viewcode",  # adds link to view source code
    "numpydoc",  # use numpy docstring format
    "matplotlib.sphinxext.plot_directive",  # generate plots from code
    "sphinx_design",  # for grids
]

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True
autodoc_typehints = "none"
numpydoc_show_class_members = False
autosummary_ignore_module_all = False  # create summary for elements in __all__
autosummary_imported_members = False
add_module_names = False
autodoc_inherit_docstrings = True

# avoid source, png, pdf links after every figure
plot_html_show_source_link = False
plot_html_show_formats = False

# to be able to use ".. code-block::"
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_css_files = ["options.css"]
html_theme_options = {
    "logo": {
        "text": "surface-sim",
        "image_dark": "_static/logo_dark.svg",
    },
    "show_prev_next": False,
    "github_url": "https://github.com/MarcSerraPeralta/surface-sim",
}
