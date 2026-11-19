# Shared set-up for the tests. You don't need to change this file.
# Each test that asks for bookings gets its own fresh copy of the 3,000 bookings in bookings.csv, read from this
# exercise's folder, so one test can't change the table another test reads.
from pathlib import Path

import pandas as pd
import pytest

BOOKINGS = Path(__file__).parent.parent / "bookings.csv"


@pytest.fixture
def bookings():
    """The 3,000 bookings in bookings.csv, freshly read for one test."""
    return pd.read_csv(BOOKINGS)
