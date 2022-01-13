import craftutils.retrieve as r


def test_svo_filter_id():
    assert r.svo_filter_id(instrument_name="wise", filter_name="w1") == "WISE/WISE.W1"


def test_retrieve_svo_filter():
    result = r.retrieve_svo_filter(instrument="wise", filter_name="w1")
    print(result)
