# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'craft-optical-followup'
copyright = '2023, Lachlan Marnoch'
author = 'Lachlan Marnoch'
release = '0.9.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_immaterial",
    "sphinx_immaterial.apidoc.python.apigen"
]

python_apigen_modules = {
    "craftutils.wrap": "api/wrap.",
    "craftutils.astrometry": "api/astrometry/",
    "craftutils.observation": "api/observation/",
    "craftutils.observation.field": "api/observation/field/",
}

python_apigen_default_groups = [
    ("class:.*", "Classes"),
    ("data:.*", "Variables"),
    ("function:.*", "Functions"),
    ("method:.*", "Methods"),
    ("classmethod:.*", "Class methods"),
    ("property:.*", "Properties"),
    ("function:craftutils.astrometry.*", "Astrometry")
]

# Create hyperlinks to other documentation
intersphinx_mapping = {
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_immaterial'
html_static_path = ['_static']
