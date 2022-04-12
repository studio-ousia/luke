import bz2
import os
import tempfile

import pytest

from luke.utils.interwiki_db import InterwikiDB

WIKIDATA_FIXTURE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../fixtures/wikidata_20180423_sitelinks10.json"
)


@pytest.fixture
def db():
    with open(WIKIDATA_FIXTURE_FILE, "rb") as f:
        data = bz2.compress(f.read())
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(data)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        return InterwikiDB.build(temp_file.name)


def test_query(db):
    ret = dict([(v, k) for k, v in db.query("Spain", "en")])
    assert ret["ja"] == "スペイン"
    assert ret["zh"] == "西班牙"
