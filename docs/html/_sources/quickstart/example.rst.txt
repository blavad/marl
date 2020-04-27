.. _example:

Examples
==================

Check available classes
------------------------

.. code-block:: python

  import marl

  # Check available agents
  print("\n| Agents\t\t", list(marl.agent.available()))

  # Check available agents
  print("\n| Policies\t\t", list(marl.policy.available()))

  # Check available agents
  print("\n| Models\t\t", list(marl.model.available()))

  # Check available exploration process
  print("\n| Expl. Processes\t", list(marl.exploration.available()))

  # Check available experience memory
  print("\n| Experience Memory\t", list(marl.experience.available()))



Single-agent example
--------------------

Example for training a single agent with DQN algorithm.

.. code-block:: python

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


Multi-agent example
-------------------

Example for training a system composed of several agents with minimax-Q algorithm.

.. warning:: Most of the multi-agent algorithms requires external knowledge.  It is necessary to specify to each of these agents their multi-agent system (MAS) by using ``ag.set_mas`` function.  

.. code-block:: python

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
  expl = EpsGreedy(eps_deb=1.,eps_fin=.3)

  # Create two minimax-Q agents
  q_agent1 = MinimaxQAgent(obs_s, act_s, act_s, exploration=expl, gamma=0.9, lr=0.001, name="SoccerJ1")
  q_agent2 = MinimaxQAgent(obs_s, act_s, act_s, exploration=expl, gamma=0.9, lr=0.001, name="SoccerJ2")

  # Create the trainable multi-agent system
  mas = MARL(agents_list=[q_agent1, q_agent2])

  # Assign MAS to each agent
  q_agent1.set_mas(mas)
  q_agent2.set_mas(mas)

  # Train the agent for 100 000 timesteps
  mas.learn(env, nb_timesteps=100000)

  # Test the agents for 10 episodes
  mas.test(env, nb_episodes=10, time_laps=0.5)  

Training two independant DQN agents
-----------------------------------

The environment ``HanabiMarlEnv`` is coming soon. 

.. code-block:: python

  import marl
  from marl.agent import DQNAgent

  from hanabi_coop.env import HanabiMarlEnv # coming soon

  config_hanabi = {   "players": 2,
                      "random_start_player": True,
                      "hand_size": 4,
                      "max_life_tokens": 3,
                      "max_information_tokens": 8,
                      "vectorized":[True,True]
                  }

  env = HanabiMarlEnv(config=config_hanabi)

  obs_s = env.observation_space
  act_s = env.action_space

  ag1 = DQNAgent("MlpNet", obs_s, act_s, name="Bob")
  ag2 = DQNAgent("MlpNet", obs_s, act_s, name="Jack")

  mas = marl.MARL([ag1,ag2])

  mas.learn(env, nb_timesteps=100000)