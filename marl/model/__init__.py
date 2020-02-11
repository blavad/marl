from .model import Model, make, available, register
from marl.model.nn import *
from marl.model.qvalue import *


register(
    id='QTable',
    entry_point='marl.model.qvalue:QTable',
)

register(
    id='MlpNet',
    entry_point='marl.model.nn.mlpnet:MlpNet',
)
