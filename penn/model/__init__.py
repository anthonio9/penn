from .core import *
from .crepe import Crepe
from .deepf0 import Deepf0
from .fcnf0 import Fcnf0
from .poly_pitch_net import PolyPitchNet1, PolyPitchNet2, PolyPitchNet3, PolyPitchNet4

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
    if name == 'ppn2':
        return PolyPitchNet2()
    if name == 'ppn3':
        return PolyPitchNet3()
    if name == 'ppn4':
        return PolyPitchNet4()
    raise ValueError(f'Model {name} is not defined')
