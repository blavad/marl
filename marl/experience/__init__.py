from .experience import Experience, make, available, register
from .replay_buffer import ReplayMemory, PrioritizedReplayMemory
from .replay_buffer import transition_tuple, transition_type

available_transition = transition_tuple.keys()

# Replay Memory

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

# RNN Replay Memory
register(
    id='RNNReplayMemory',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    transition_type='RNNTransition'
)

register(
    id='RNNReplayMemory-1',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=1,
    transition_type = 'RNNTransition'
),

register(
    id='RNNReplayMemory-100',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=100,
    transition_type = 'RNNTransition'
)

register(
    id='RNNReplayMemory-500',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=500,
    transition_type = 'RNNTransition'
)

register(
    id='RNNReplayMemory-1000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=1000,
    transition_type = 'RNNTransition'
)

register(
    id='RNNReplayMemory-2000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=2000,
    transition_type = 'RNNTransition'
)

register(
    id='RNNReplayMemory-5000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=5000,
    transition_type = 'RNNTransition'
)

register(
    id='RNNReplayMemory-10000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=10000,
    transition_type = 'RNNTransition'
)

register(
    id='RNNReplayMemory-30000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=30000,
    transition_type = 'RNNTransition'
)

register(
    id='RNNReplayMemory-50000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=50000,
    transition_type = 'RNNTransition'
)

register(
    id='RNNReplayMemory-100000',
    entry_point='marl.experience.replay_buffer:ReplayMemory',
    capacity=100000,
    transition_type = 'RNNTransition'
)




# Prioritized Exp Replay
register(
    id='PrioritizedReplayMemory',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    transition_type='FFTransition'
)

register(
    id='PrioritizedReplayMemory-1',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=1,
    transition_type='FFTransition'
)

register(
    id='PrioritizedReplayMemory-100',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=100,
    transition_type='FFTransition'
)

register(
    id='PrioritizedReplayMemory-500',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=500,
    transition_type='FFTransition'
)

register(
    id='PrioritizedReplayMemory-1000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=1000,
    transition_type='FFTransition'
)

register(
    id='PrioritizedReplayMemory-2000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=2000,
    transition_type='FFTransition'
)

register(
    id='PrioritizedReplayMemory-5000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=5000,
    transition_type='FFTransition'
)

register(
    id='PrioritizedReplayMemory-10000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=10000,
    transition_type='FFTransition'
)

register(
    id='PrioritizedReplayMemory-30000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=30000,
    transition_type='FFTransition'
)

register(
    id='PrioritizedReplayMemory-50000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=50000,
    transition_type='FFTransition'
)

register(
    id='PrioritizedReplayMemory-100000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=100000,
    transition_type='FFTransition'
)


# RNN Prioritized Exp Replay
register(
    id='RNNPrioritizedReplayMemory',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    transition_type='RNNTransition'
)

register(
    id='RNNPrioritizedReplayMemory-1',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=1,
    transition_type='RNNTransition'
)

register(
    id='RNNPrioritizedReplayMemory-100',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=100,
    transition_type='RNNTransition'
)

register(
    id='RNNPrioritizedReplayMemory-500',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=500,
    transition_type='RNNTransition'
)

register(
    id='RNNPrioritizedReplayMemory-1000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=1000,
    transition_type='RNNTransition'
)

register(
    id='RNNPrioritizedReplayMemory-2000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=2000,
    transition_type='RNNTransition'
)

register(
    id='RNNPrioritizedReplayMemory-5000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=5000,
    transition_type='RNNTransition'
)

register(
    id='RNNPrioritizedReplayMemory-10000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=10000,
    transition_type='RNNTransition'
)

register(
    id='RNNPrioritizedReplayMemory-30000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=30000,
    transition_type='RNNTransition'
)

register(
    id='RNNPrioritizedReplayMemory-50000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=50000,
    transition_type='RNNTransition'
)

register(
    id='RNNPrioritizedReplayMemory-100000',
    entry_point='marl.experience.replay_buffer:PrioritizedReplayMemory',
    capacity=100000,
    transition_type='RNNTransition'
)
