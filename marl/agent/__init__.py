from .agent import Agent, TrainableAgent, MATrainable, register, make, available
from .q_agent import *
from .pg_agent import *
from .maac_agent import *
from ..marl import *

register(
    id='TrainalbleAgent',
    entry_point='marl.agent.agent:TrainableAgent'
)

register(
    id='QTableAgent',
    entry_point='marl.agent.q_agent:QTableAgent'
)

register(
    id='DQNAgent',
    entry_point='marl.agent.q_agent:DQNAgent'
)

register(
    id='MinimaxQAgent',
    entry_point='marl.agent.q_agent:MinimaxQAgent'
)