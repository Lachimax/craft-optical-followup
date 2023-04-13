Configuration files
===================

The behaviour of the pipeline for a particular epoch can be modified by way of its ``.yaml`` configuration file. One is
generated for every new Epoch and Field created via the ``craft_optical_pipeline`` script.

..

<Here put an example config file>

.. code-block:: yaml

    data_path: FRB20180924/imaging/vlt-fors2/2019-08-23-FRB20180924_4
    date: !astropy.time.Time {format: iso, in_subfmt: '*', jd1: 2458718.0, jd2: 0.5, out_subfmt: date,
      precision: 3, scale: utc}
    field: FRB20180924
    instrument: vlt-fors2
    name: FRB20180924_FORS2_3
    program_id: 0103.A-0101(A)
    target: FRB 180924 Host

The ``date`` field here looks somewhat alarming, but this is just the YAML representation of an :class:`astropy`

<Somewhere there should also be a complete config file with every default>