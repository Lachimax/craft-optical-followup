Pipelines
=========

Stages can be added to a pipeline (e.g. ``Epoch``) by subclassing.

The main requirement, for compatibility with the ``pipeline()`` method, is that the method added to the ``stages``
dictionary should have an ``output_dir`` argument, and accept ``**kwargs`` (even if it doesn't actually use them).

I have generally used a short ``proc_...`` method that wraps around the important method (and does some keyword
processing, etc.), for the sake of a consistent signature and for quickly recognising which
methods are pipeline methods, but this is not mandatory.

If a pipeline method returns ``False`` after running, ``pipeline()`` will interpret this as a failure, although no behaviour is
currently defined for this case; the stage will simply not be registered in ``stages_complete``.