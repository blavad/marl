# MARL

MARL is a high-level multi-agent reinforcement learning API, written in Python.

## Installation
```bash
git clone https://github.com/blavad/marl.git
pip intall -e marl
```

## Examples

### Examples of training a single agent with DQN algorithm
```python
import gym

import marl
from marl.agent import DQNAgent
from marl.model.nn import MlpNet

env = gym.make("LunarLander-v2")

obs_s = env.observation_space
act_s = env.action_space

mlp_model = MlpNet(8,4, hidden_size=[64, 32])

dqn_agent = DQNAgent(mlp_model, obs_s, act_s, experience="ReplayMemory-5000", exploration="EpsGreedy", lr=0.001, name="DQN-LunarLander")

# Train the agent for 100 000 timesteps
dqn_agent.learn(env, nb_timesteps=100000)

# Test the agent for 100 episodes
dqn_agent.test(env, nb_episodes=100)
```





