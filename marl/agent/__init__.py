from .agent import Agent, TrainableAgent, register, make, available
from .q_agent import *
from .pg_agent import *
from ..marl import MARL

register(
    id='TrainalbleAgent',
    entry_point='marl.agent.agent:TrainableAgent'
)

register(
    id='QAgent',
    entry_point='marl.agent.q_agent:QAgent'
)

register(
    id='DQNAgent',
    entry_point='marl.agent.q_agent:DQNAgent'
)