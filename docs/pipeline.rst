Using the pipeline script
=========================

**WARNING:** take care when running more than one instance of the imaging pipeline on the same system, as they can interfere with each other on certain stages;
for example, the astrometry-related stages.


When the ``craft-optical-followup`` package has been properly installed (see :doc:`installation`), the script
``craft_optical_pipeline`` will be placed in the ``bin`` directory for the current environment, and thus should be
callable from anywhere.

.. code-block:: bash

        $ craft_optical_pipeline

Simply invoking this from your terminal will conjure an interactive programme via which any of the supported pipelines
can be launched, and guiding you through the main options, such as which Field and Epoch you want to work on. However,
some of these can be skipped by including parameter flags:

.. code-block:: bash

        $ craft_optical_pipeline --epoch FRB20180924_FORS2_1 --field FRB20180924

A full listing of valid flags is given `below <Script parameters>`_.

So long as an Epoch has been generated previously, the ``--field`` argument is redundant; the pipeline remembers which
field an epoch belongs to.

New Epochs
__________

If, however, it has not, the window will request further input. For the most part these questions are self-explanatory,
and will ask you to specify the mode (``spectroscopy`` or ``imaging``; currently only ``imaging`` has any meaningful
functionality) and the instrument on which the Epoch was observed. If the field is also new, you will need to provide
some other information such as position on the sky.

A new ``.yaml`` will be generated in the main parameter directory ``<param_dir>/fields/<field>/<mode>/<instrument>/<epoch_name>.yaml``,
containing the information you've given. The config file for the Field it belongs to can also be found under
``<param_dir>/fields/<field>/<mode>/<instrument>/<epoch_name>.yaml``,

For more detailed control--for example, to skip or modify pipeline steps (as may be necessary for unusual fields, eg if the astrometry stage is having
trouble converging on a solution with the default settings)--the configuration YAML file for the Epoch or for its Field
can be edited.
For more on this file, see :doc:`config`.

Script parameters
-----------------

<description of flags etc.>

A description of these flags and parameters can be displayed by invoking:

.. code-block:: bash

        $ craft_optical_pipeline -h

