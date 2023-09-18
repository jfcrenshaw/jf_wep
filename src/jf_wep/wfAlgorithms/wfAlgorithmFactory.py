from pathlib import Path
from typing import Union

from jf_wep.utils import WfAlgorithmName, loadConfig
from jf_wep.wfAlgorithms.tie import TIEAlgorithm
from jf_wep.wfAlgorithms.wfAlgorithm import WfAlgorithm


class WfAlgorithmFactory:
    """Factory for loading different wavefront estimation algorithms."""

    @staticmethod
    def createWfAlgorithm(
        algoName: Union[WfAlgorithmName, str],
        algoConfig: Union[Path, str, dict, WfAlgorithm, None] = None,
    ):
        """Return a configured WfAlgorithm.

        Parameters
        ----------
        algoName : WfAlgorithmName or str
            A WfAlgorithmName enum or the corresponding string, indicating
            which WfAlgorithm to use.

        """
        # Convert to enum
        algoName = WfAlgorithmName(algoName)

        # Return the configured algorithm
        if algoName == WfAlgorithmName.TIE:
            return loadConfig(algoConfig, TIEAlgorithm)
        else:
            raise ValueError(f"{algoName} not supported.")
