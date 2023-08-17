"""Test the donutImage class."""
import numpy as np
import pytest

from jf_wep.donutStamp import DonutImage


class TestBadParams:
    """Test that bad parameter values raise errors."""

    good_values = {
        "image": np.zeros((2, 2)),
        "fieldAngle": (0, 0),
        "defocalType": "intra",
        "filterLabel": "r",
        "blendOffsets": None,
    }

    def test_bad_image(self):
        """Test bad image inputs."""
        with pytest.raises(TypeError):
            DonutImage(**(self.good_values | {"image": "fake"}))
        with pytest.raises(ValueError):
            DonutImage(**(self.good_values | {"image": np.zeros((2, 3))}))

    def test_bad_fieldAngle(self):
        """Test bad fieldAngle inputs."""
        with pytest.raises(ValueError):
            DonutImage(**(self.good_values | {"fieldAngle": "fake"}))
        with pytest.raises(ValueError):
            DonutImage(**(self.good_values | {"fieldAngle": (0, 0, 0)}))

    def test_bad_defocalType(self):
        """Test bad fieldAngle inputs."""
        with pytest.raises(TypeError):
            DonutImage(**(self.good_values | {"defocalType": None}))

    def test_bad_filterLabel(self):
        """Test bad filterLabel inputs."""
        with pytest.raises(TypeError):
            DonutImage(**(self.good_values | {"filterLabel": None}))

    def test_bad_blendOffsets(self):
        """Test bad blendOffsets."""
        with pytest.raises(ValueError):
            DonutImage(
                **(self.good_values | {"blendOffsets": np.zeros((2, 3, 3))})
            )
