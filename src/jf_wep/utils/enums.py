"""Enum classes."""
from enum import Enum


class PlaneType(Enum):
    Image = "image"
    Pupil = "pupil"


class DefocalType(Enum):
    Intra = "intra"
    Extra = "extra"


class BandLabel(Enum):
    LSST_U = "u"
    LSST_G = "g"
    LSST_R = "r"
    LSST_I = "i"
    LSST_Z = "z"
    LSST_Y = "y"
    REF = "ref"


class WfAlgorithmName(Enum):
    TIE = "tie"
