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
    license='Attribution-NonCommercial-ShareAlike 4.0 International'
)

if not os.path.isfile("param/config.yaml"):
    copy("param/config_template.yaml", "param/config.yaml")

    config_path = os.path.join(os.getcwd(), "param", "config.yaml")
    print(f"A fresh config file has been created at '{config_path}'")
    print(
        "In this file, please set 'top_data_dir' to a valid path in which to store all data products of this package.")
    print("WARNING: This may require a large amount of space.")

    input("\nOnce you have done this, press any key to proceed.")

    import craftutils.params as p

    p.add_config_param(params={"proj_dir": os.getcwd() + "/"})

import craftutils.utils as u

data_dir = p.config["top_data_dir"]
u.mkdir_check_nested(data_dir)
