from .agent import Agent, TrainableAgent, MATrainable, register, make, available
from .q_agent import *
from .pg_agent import *
from .maac_agent import *
from ..marl import *

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

register(
    id='DeepACAgent',
    entry_point='marl.agent.pg_agent:DeepACAgent'
)

register(
    id='PHCAgent',
    entry_point='marl.agent.pg_agent:PHCAgent'
)

register(
    id='DDPGAgent',
    entry_point='marl.agent.pg_agent:DDPGAgent'
)


register(
    id='MADDPGAgent',
    entry_point='marl.agent.maac_agent:MADDPGAgent'
)
