Output files
============

The classes :class:``craftutils.observation.field.Field``, :class:``craftutils.observation.epoch.Epoch``,
:class:``craftutils.observation.image.Image``
:class:``craftutils.observation.instrument.Instrument``, and their various descendant classes, write output ``.yaml`` files in their data
directories that contain various calculated values and paths to other output files generated in the course of processing.
The filenames generally match the pattern ``<name>_outputs.yaml``.

Each key found in one of these files will correspond to the name of a property of that object as initialised in its
``__init__()`` method.

Image objects are represented as path strings that point to their image file.