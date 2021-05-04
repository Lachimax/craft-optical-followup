import os

import setuptools
from shutil import copy

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
    python_requires='>=3.6',
    install_requires=[
        "astropy",
        "astroquery",
        "matplotlib",
        "reproject",
        "requests",
        "sncosmo",
        "numpy",
        "PyYAML"
    ],
    license='Attribution-NonCommercial-ShareAlike 4.0 International'
)

import craftutils.params as p

param_path = os.path.join(os.getcwd(), "param")
config_path = os.path.join(param_path, "config.yaml")
if not os.path.isfile(config_path):
    copy(os.path.join(param_path, "config_template.yaml"), config_path)

    print(f"A fresh config file has been created at '{config_path}'")
    print(
        "In this file, please set 'top_data_dir' to a valid path in which to store all data products of this package.")
    print("WARNING: This may require a large amount of space.")
    print("You may also like to specify an alternate param_dir")

    input("\nOnce you have done this, press any key to proceed.")

    p.add_config_param(params={"proj_dir": os.getcwd()})

import craftutils.utils as u

data_dir = p.config["top_data_dir"]
u.mkdir_check_nested(data_dir)
key_path = os.path.join(p.config["param_dir"], 'keys.json')
if not os.path.isfile(key_path):
    copy(os.path.join(param_path, "keys.json"), key_path)
