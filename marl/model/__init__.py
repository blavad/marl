from .model import Model, make, available, register
from marl.model.nn import *
from marl.model.qvalue import *


register(
    id='VTable',
    entry_point='marl.model.qvalue:VTable',
)

register(
    id='QTable',
    entry_point='marl.model.qvalue:QTable',
)

register(
    id='MultiQTable',
    entry_point='marl.model.qvalue:MultiQTable',
)

register(
    id='ActionProb',
    entry_point='marl.model.qvalue:ActionProb',
)


register(
    id='MlpNet',
    entry_point='marl.model.nn.mlpnet:MlpNet',
)

register(
    id='GumbelMlpNet',
    entry_point='marl.model.nn.mlpnet:GumbelMlpNet',
)


register(
    id='ContinuousCritic',
    entry_point='marl.model.nn.mlpnet:ContinuousCritic',
)
