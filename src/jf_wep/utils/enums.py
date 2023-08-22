"""Enum classes."""
from enum import Enum


class DefocalType(Enum):
    Intra = "intra"
    Extra = "extra"


class FilterLabel(Enum):
    LSST_U = "u"
    LSST_G = "g"
    LSST_R = "r"
    LSST_I = "i"
    LSST_Z = "z"
    LSST_Y = "y"
    REF = "ref"


class WfAlgorithmName(Enum):
    TIE = "tie"
