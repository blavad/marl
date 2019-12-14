from marl.policy.policy import Policy, make, available, register
from .policies import *

register(
    id='QPolicy',
    entry_point='marl.policy.policies:QPolicy',

)

register(
    id='QTable',
    entry_point='marl.policy.policies:QPolicy',
    q_value="marl.model.qvalue:QTable"
)

register(
    id='StochasticPolicy',
    entry_point='marl.policy.policies:StochasticPolicy'
)

register(
    id='DeterministicPolicy',
    entry_point='marl.policy.policies:DeterministicPolicy'
)