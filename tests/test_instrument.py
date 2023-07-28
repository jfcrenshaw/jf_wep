"""Test the Instrument class."""
import pytest

from jf_wep.instrument import Instrument
from jf_wep.utils import getPolicyDir


class TestBadParams:
    """Test that bad parameter values raise errors."""

    def test_bad_diameter(self):
        with pytest.raises(ValueError):
            Instrument(diameter="fake")
        with pytest.raises(ValueError):
            Instrument(diameter=-1)

    def test_bad_obscuration(self):
        with pytest.raises(ValueError):
            Instrument(obscuration="fake")
        with pytest.raises(ValueError):
            Instrument(obscuration=-1)
        with pytest.raises(ValueError):
            Instrument(obscuration=1.2)

    def test_bad_focalLength(self):
        with pytest.raises(ValueError):
            Instrument(focalLength="fake")
        with pytest.raises(ValueError):
            Instrument(focalLength=-1)

    def test_bad_defocalOffset(self):
        with pytest.raises(ValueError):
            Instrument(defocalOffset="fake")


@pytest.mark.parametrize(
    "configFile", list((getPolicyDir() / "instruments").glob("*"))
)
def test_instrument_from_paramFile(configFile):
    """Test that we can create by specifying a file path."""
    # First test using an absolute path
    Instrument(configFile=configFile)

    # Now test we can use a path relative to policy
    Instrument(configFile=str(configFile).partition("jf_wep/")[-1])
