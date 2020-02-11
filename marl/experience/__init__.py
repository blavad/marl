from .experience import Experience, make, available, register
from .replay_buffer import ReplayMemory

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
    id='ReplayMemory-100',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=100
)

register(
    id='ReplayMemory-500',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=500
)

register(
    id='ReplayMemory-1000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=1000
)

register(
    id='ReplayMemory-2000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=2000
)

register(
    id='ReplayMemory-5000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=5000
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

register(
    id='ReplayMemory-50000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=50000
)

register(
    id='ReplayMemory-100000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=100000
)