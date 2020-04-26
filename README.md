# MARL

MARL is a high-level multi-agent reinforcement learning API, written in Python.

Project doc : <a href="https://blavad.github.io/marl/html/index.html"> [DOC]</a>

## Installation
```bash
git clone https://github.com/blavad/marl.git
cd marl
pip install -e .
```

## Implemented algorithms

### Single-agent algorithms

| **Q-learning**     | **DQN**             | **Actor-Critic**     | **DDPG**            |**TD3**            |
| ------------------ | ------------------- | -------------------- | ------------------- |------------------- | 
| :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark:   | :heavy_check_mark:  |:x:  |


### Multi-agent algorithms

| **minimaxQ**         | **PHC**       | **JAL** | **MAAC**             | **MADDPG**           | 
| -------------------- | -------------------- | ------- |  -------------------- |  ------------------- | 
|  :heavy_check_mark: | :heavy_check_mark:   |  :x:    |  :heavy_check_mark:   | :heavy_check_mark:   |

## Examples

### Train a single agent with DQN algorithm
```python
import marl
from marl.agent import DQNAgent
from marl.model.nn import MlpNet

import gym

env = gym.make("LunarLander-v2")

obs_s = env.observation_space
act_s = env.action_space

mlp_model = MlpNet(8,4, hidden_size=[64, 32])

dqn_agent = DQNAgent(mlp_model, obs_s, act_s, experience="ReplayMemory-5000", exploration="EpsGreedy", lr=0.001, name="DQN-LunarLander")

# Train the agent for 100 000 timesteps
dqn_agent.learn(env, nb_timesteps=100000)

# Test the agent for 10 episodes
dqn_agent.test(env, nb_episodes=10)
```

### Train two agents with Minimax-Q algorithm

```python
import marl
from marl import MARL
from marl.agent import MinimaxQAgent
from marl.exploration import EpsGreedy

from soccer import DiscreteSoccerEnv
# Environment available here "https://github.com/blavad/soccer"
env = DiscreteSoccerEnv(nb_pl_team1=1, nb_pl_team2=1)

obs_s = env.observation_space
act_s = env.action_space

# Custom exploration process
expl1 = EpsGreedy(eps_deb=1.,eps_fin=.3)
expl2 = EpsGreedy(eps_deb=1.,eps_fin=.3)

# Create two minimax-Q agents
q_agent1 = MinimaxQAgent(obs_s, act_s, act_s, exploration=expl1, gamma=0.9, lr=0.001, name="SoccerJ1")
q_agent2 = MinimaxQAgent(obs_s, act_s, act_s, exploration=expl2, gamma=0.9, lr=0.001, name="SoccerJ2")

# Create the trainable multi-agent system
mas = MARL(agents_list=[q_agent1, q_agent2])

# Assign MAS to each agent
q_agent1.set_mas(mas)
q_agent2.set_mas(mas)

# Train the agent for 100 000 timesteps
mas.learn(env, nb_timesteps=100000)

# Test the agents for 10 episodes
mas.test(env, nb_episodes=10, time_laps=0.5)
```



