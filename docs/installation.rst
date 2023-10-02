Installation
============

Prequisites
-----------

Some packages are not required for primary functionality but are necessary for certain features.

* `FRB <https://github.com/FRBs/FRB>`_ is used by :class:`craftutils.observation.objects.FRB` for some modelling to do with FRB propagation.

Installing
----------

Currently, the best way to install this package is to clone it from GitHub and then use
``pip`` to install it. I may add the package to the Python Package Index (PyPI) at some point in order to streamline installation, but haven't yet.

#. Clone from https://github.com/Lachimax/craft-optical-followup.git

#. Open a terminal in the directory in which you have cloned it.

#. You may wish to create a new virtual environment in which to complete the installation. This author recommends :doc:`conda`, which is included in any Anaconda installation.

#. Install:
    .. code-block:: bash

        $ pip install .

    Or, if you want to edit the code while installed, e.g. to contribute:

    .. code-block:: bash

        $ pip install -e .

#. A package-level configuration file should have been generated in the user's home directory ``~/.craftutils/config.yaml``. See :doc:`config` for a full description; the most important values here are ``data_dir`` and ``param_dir``.
    #. ``data_dir`` should be set to the directory in which all data will be reduced. It should be on a drive with plenty of space.
    #. ``param_dir`` should be set to the directory in which the input ``.yaml`` files will be written to and read from.
    #. These can also be set from within Python by invoking the functions :func:`craftutils.params.set_data_dir` and :func:`craftutils.params.set_param_dir`.

Other dependencies
------------------

Astrometry.net
^^^^^^^^^^^^^^

``astrometry.net`` is used for solving the astrometry of images.
It should be installed via ``conda`` and NOT via ``pip``; there *is* a package called ``astrometry`` in the Python Package Index, and it does in fact implement part of the ``astrometry.net`` codebase, but it does not behave in the same way as the ``conda`` package and does not include the full functionality. 
This is why ``astrometry`` is excluded from ``requirements.txt`` but included in ``environment.yaml``.
So, if you've made the main installation via ``pip``,  ``astrometry.net`` should be installed separately via:

.. code-block:: bash

    $ conda install astrometry

If you are not using `conda`, `astrometry.net` can also be installed following the instructions at http://astrometry.net/doc/build.html. This should work fine but is marginally less convenient.
