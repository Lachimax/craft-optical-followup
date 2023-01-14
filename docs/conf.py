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
    "craftutils.fits_files": "api/fits_files/",
    "craftutils.params": "api/params/",
    "craftutils.photometry": "api/photometry/",
    "craftutils.plotting": "api/plotting/",
    "craftutils.retrieve": "api/retrieve/",
    "craftutils.sne": "api/sne/",
    "craftutils.stats": "api/stats/",
    "craftutils.utils": "api/utils/",
    "craftutils.observation": "api/observation/",
    "craftutils.observation.field": "api/observation/field/",
}

python_apigen_default_groups = [
    ("class:.*", "Classes"),
    ("data:.*", "Variables"),
    ("function:.*", "Functions"),
    ("classmethod:.*", "Class methods"),
    ("property:.*", "Properties"),
    ("module:craftutils.*", "Modules"),
    ("function:craftutils.astrometry.*", "Astrometry"),
    ("function:craftutils.fits_files.*", "Fits files"),
    ("function:craftutils.params.*", "Params"),
    ("function:craftutils.photometry.*", "Photometry"),
    ("function:craftutils.plotting.*", "Plotting"),
    ("function:craftutils.retrieve.*", "Retrieve"),
    ("function:craftutils.sne.*", "SNe"),
    ("function:craftutils.stats.*", "Stats"),
    ("function:craftutils.utils.*", "Utils")

]

# Create hyperlinks to other documentation
intersphinx_mapping = {
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "astroquery": ("https://astroquery.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
    # "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sncosmo": ("https://sncosmo.readthedocs.io/en/stable/", None)
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_immaterial'
html_static_path = ['_static']
