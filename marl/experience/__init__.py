from .experience import Experience, make, available, register
from .replay_buffer import ReplayMemory, ExpReplayBuffer

register(
    id='ReplayMemory',
    entry_point='marl.experience.replay_buffer:ReplayMemory'
)

register(
    id='ReplayMemory-1',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=1
)

register(
    id='ReplayMemory-1000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=1000
)

register(
    id='ReplayMemory-10000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=10000
)

register(
    id='ReplayMemory-30000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=30000
)