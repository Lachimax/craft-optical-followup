import setuptools
from shutil import copy
from os import getcwd

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

# copy("param/config_template.yaml", "param/config.yaml")
#
# input("A fresh config.yaml file has been created in the param/ directory. Please ensure top_data_dir points to "
#       "the valid path in which to store all data products of this package."
#       "\nOnce you have done this, press any key to proceed.")
#
# import craftutils.params as p
#
# p.add_config_param(params={"proj_dir": getcwd() + "/"})
#
# import craftutils.utils as u
#
# data_dir = p.config["top_data_dir"]
# u.mkdir_check_nested(data_dir)
