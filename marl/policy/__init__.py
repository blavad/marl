from marl.policy.policy import Policy, make, available, register
from .policies import *

register(
    id='RandomPolicy',
    entry_point='marl.policy.policies:RandomPolicy',
)

register(
    id='QPolicy',
    entry_point='marl.policy.policies:QPolicy',
)

register(
    id='StochasticPolicy',
    entry_point='marl.policy.policies:StochasticPolicy'
)

register(
    id='DeterministicPolicy',
    entry_point='marl.policy.policies:DeterministicPolicy'
)