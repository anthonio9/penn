from .core import *
from .crepe import Crepe
from .deepf0 import Deepf0
from .fcnf0 import Fcnf0
from .poly_pitch_net import PolyPitchNet1, PolyPENNFCN, PolyPENNHFCN, PolyPENNRFCN, PolyPENNDFCN

import penn


def Model(name=penn.MODEL):
    """Create a model"""
    if name == 'crepe':
        return Crepe()
    if name == 'deepf0':
        return Deepf0()
    if name == 'fcnf0':
        return Fcnf0()
    if name == 'ppn':
        return PolyPitchNet1()
    if name == 'polypennfcn':
        return PolyPENNFCN()
    if name == 'polypennhfcn':
        return PolyPENNHFCN()
    if name == 'polypennrfcn':
        return PolyPENNRFCN()
    if name == 'polypenndfcn':
        return PolyPENNDFCN()


    raise ValueError(f'Model {name} is not defined')
