import os

import craftutils.retrieve as r
from pkg_resources import resource_filename

data_dir = os.path.join(os.path.dirname(__file__), 'files')

def test_svo_filter_id():
    assert r.svo_filter_id(
        facility_name="WISE",
        instrument_name="WISE",
        filter_name="W1"
    ) == "WISE/WISE.W1"


def test_retrieve_svo_filter():
    result = r.retrieve_svo_filter(
        facility_name="WISE",
        instrument_name="WISE",
        filter_name="W1"
    )
    compare = open(os.path.join(
        data_dir,
        'wise_W1_SVOTable.xml')
    )
    assert result.decode() == compare.read()
