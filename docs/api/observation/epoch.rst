.. currentmodule:: craftutils.observation.epoch
.. py:module:: craftutils.observation.epoch

Epochs (craftutils.observation.epoch)
================================================

Contains classes representing epochs of observation,
as well as functions for managing them.
The :cls:`craftutils.observation.epoch.Epoch` class is the primary means by which a pipeline is actuated.

The attribute `Epoch.do` is a numeric list of pipeline stages, which the `Epoch.pipeline()` function will attempt to perform. The corresponding numbers can be obtained with `Epoch.enumerate_stages()`, and will differ for each subclass of `Epoch`.
However, stages listed in `Epoch.do` will still only be performed if they are `default` (ie, their entry in `Epoch.stages()` dictionary has `"default": False`) OR they are listed in the parameter YAML for that epoch, with entry `True` within the `do` dictionary.

That is, the `do` list is really a list of steps to *check* for this particular run, each of which will be performed only if it is specified by the epoch YAML or it is a default stage.

In normal useage, this attribute is set by the `craft_optical_pipeline` argument `--do`.

Functions
---------
.. python-apigen-group:: Epoch functions

Classes
-------
.. python-apigen-group:: Epoch classes
