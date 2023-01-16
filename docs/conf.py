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
    "craftutils.wrap": "apigen/wrap.",
    "craftutils.astrometry": "apigen/astrometry/",
    "craftutils.fits_files": "apigen/fits_files/",
    "craftutils.params": "apigen/params/",
    "craftutils.photometry": "apigen/photometry/",
    "craftutils.plotting": "apigen/plotting/",
    "craftutils.retrieve": "apigen/retrieve/",
    "craftutils.sne": "apigen/sne/",
    "craftutils.stats": "apigen/stats/",
    "craftutils.utils": "apigen/utils/",
    "craftutils.observation": "apigen/observation/",
    "craftutils.observation.field": "apigen/observation/field/",
    "craftutils.observation.image": "apigen/observation/image/",
    "craftutils.observation.instrument": "apigen/observation/instrument/",
    "craftutils.observation.log": "apigen/observation/log/",
    "craftutils.observation.objects": "apigen/observation/objects/",
    "craftutils.observation.survey": "apigen/observation/survey/",
    "craftutils.wrap.astrometry_net": "apigen/wrap/astrometry_net/",
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
    ("function:craftutils.utils.*", "Utils"),
    ("function:craftutils.observation.*", "Observation"),
    ("class:craftutils.observation.field.*", "Field classes"),
    ("function:craftutils.observation.field.*", "Field functions"),
    ("class:craftutils.observation.image.*", "Image classes"),
    ("function:craftutils.observation.image.*", "Image functions"),
    ("class:craftutils.observation.instrument.*", "Instrument classes"),
    ("function:craftutils.observation.instrument.*", "Instrument functions"),
    ("class:craftutils.observation.log.*", "Log classes"),
    ("function:craftutils.observation.log.*", "Log functions"),
    ("class:craftutils.observation.objects.*", "Object classes"),
    ("function:craftutils.observation.objects.*", "Object functions"),
    ("class:craftutils.observation.survey.*", "Survey classes"),
    ("function:craftutils.observation.survey.*", "Survey functions"),
    ("function:craftutils.wrap.astrometry_net.*", "Astrometry.net functions"),
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

highlight_language = "python3"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_immaterial'
html_static_path = ['_static']
