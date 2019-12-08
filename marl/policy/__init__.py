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
    id='PolicyApprox',
    entry_point='marl.policy.policies:PolicyApprox'
)
