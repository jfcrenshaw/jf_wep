"""Test the WfEstimator class."""
import pytest

from jf_wep.wfEstimator import WfEstimator


class TestBadParams:
    """Test that bad parameter values raise errors."""

    def test_bad_algo(self):
        """Test bad algo name."""
        with pytest.raises(ValueError):
            WfEstimator(algo="fake")

    def test_bad_algo_params(self):
        """Test bad algo_params."""
        with pytest.raises(TypeError):
            WfEstimator(algo_params="fake")

    def test_bad_units(self):
        """Test bad units."""
        with pytest.raises(ValueError):
            WfEstimator(units="fake")

    def test_bad_jmax(self):
        """Test bad jmax."""
        with pytest.raises(TypeError):
            WfEstimator(jmax=6.2)
        with pytest.raises(ValueError):
            WfEstimator(jmax=3)

    def test_bad_shapeMode(self):
        """Test bad shapeMode."""
        with pytest.raises(ValueError):
            WfEstimator(shapeMode="fake")
