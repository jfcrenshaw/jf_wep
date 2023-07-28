"""Test the classes in the TIE module."""
import pytest

from jf_wep.wfAlgorithms.tie import (
    TIEAlgorithm,
    TIECompensator,
)


def test_bad_config():
    """Test that bad config values for TIEAlgorithm config raise errors."""
    # Create a TIE Algorithm we can configure
    tieAlg = TIEAlgorithm()

    # Bad solver name
    with pytest.raises(ValueError):
        tieAlg.config(solver="fake")

    # Bad optical model
    with pytest.raises(ValueError):
        tieAlg.config(opticalModel="fake")
