Installation
============

Currently, the best way to install this package is to clone it from GitHub and then use
``pip`` to install it.

#. While not a requirement, it is generally safest and most convenient to operate the code from within a :doc:`conda` environment. 
``conda`` can be set up for your system following the instructions at https://docs.anaconda.com/miniconda/.

#. If you are creating a new ``conda`` environment to install ``craft-optical-followup`` in, I recommend using the ``environment.yaml``
 file included with ``craft-optical-followup``, which includes most of the requirements. ``pip`` will take care of the rest in the following steps. Like so:

    .. code-block:: bash

        cd craft-optical-followup
        conda env create -f environment.yaml

#. Clone from https://github.com/Lachimax/craft-optical-followup.git

#. Open a terminal in the directory in which you have cloned it.

#. Install:
    .. code-block:: bash

        pip install .

    Or, if you want to edit the code while installed, e.g. to contribute:

    .. code-block:: bash

        pip install -e .

#. A package-level configuration file should have been generated in the user's home directory ``~/.craftutils/config.yaml``. See :doc:`config` for a full description; the most important values here are ``data_dir`` and ``param_dir``.
    #. ``data_dir`` should be set to the directory in which all data will be reduced. It should be on a drive with plenty of space. If this is not set in the file, you will be asked to enter it the first time that you run ``craft_optical_followup``.
    #. ``param_dir`` should be set to the directory in which the input ``.yaml`` files will be written to and read from. By default, the repository directory ``craft-optical-followup/craftutils/param/`` will be used.
    #. These can also be set from within Python by invoking the functions :func:`craftutils.params.set_data_dir` and :func:`craftutils.params.set_param_dir`.


Other requirements
==================

Astrometry.net
--------------

``astrometry.net`` is used for solving the astrometry of images.
It should be installed via ``conda`` and NOT via ``pip``; there *is* a package called ``astrometry`` in the Python Package Index, and it does in fact implement part of the ``astrometry.net`` codebase, but it does not behave in the same way as the ``conda`` package and does not include the full functionality. 
This is why ``astrometry`` is excluded from ``requirements.txt`` but included in ``environment.yaml``.
So, if you've made the main installation via ``pip``, and have not used ``environment.yaml`` to create the environment, ``astrometry.net`` should be installed separately via:

.. code-block:: bash

    conda install -c conda-forge astrometry

If you are not using ``conda``, ``astrometry.net`` can also be installed following the instructions at http://astrometry.net/doc/build.html. This should work fine but is marginally less convenient.

Optional dependencies
---------------------

Some packages are not required for primary functionality but are necessary for certain features.

* The `FRB repository <https://github.com/FRBs/FRB>`_ is used by :class:`craftutils.observation.objects.FRB` for some modelling to do with FRB propagation. This package by itself has quite a tangled web of dependencies that can be a bit tricky to satisfy, so I leave it to the advanced user to decide if they need it. It is not used in any image processing.
* `ESOReflex <https://www.eso.org/sci/software/esoreflex/>`_ is still required by the FORS2 pipeline, for the initial reduction.
