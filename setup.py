import os

import setuptools

packages = setuptools.find_packages()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="craftutils",
    version="0.9.0",
    author="Lachlan Marnoch",
    #    short_description=long_description,
    long_description=long_description,
    url="https://github.com/Lachimax/craft-optical-followup",
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    install_requires=[
        "astroalign",
        "astropy",
        "astroquery",
        "ccdproc",
        "matplotlib",
        "photutils",
        "requests",
        "numpy",
        "PyYAML",
        "sep",
    ],
    license='Attribution-NonCommercial-ShareAlike 4.0 International',
    scripts=["bin/craft_optical_pipeline"]
)
