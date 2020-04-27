.. _environment:

Environment requirements
=========================

The environment is crucial in the learning procedure. A good trained agent requires an adequate environment. In order to fit the implementation of the package **marl**,
the environment must follow some simple rules.

OpenAI Gym based environment
----------------------------
MARL-API project is related to OpenAI Gym project https://gym.openai.com/ . 
To be in accordance with our implementation, the environment used must inherit or reimplement 
the following methods (specific to OpenAI Gym environments):

* ``reset()`` : Reset the environment to an intial state. This method is called when starting a new episode and return an observation.
* ``step(action)`` : Update the state of the environment given an action (possibly a joint action for multi-agent training). The output of this method consist in four elements (next observation(s), reward(s), boolean(s) indicating whether the episode is done or not, extra informations)
* ``render()`` : Display the environment (only used for testing with parameter ``display=True``)

Moreover, it is recommended that environments have two attributes:

* ``observation_space`` (gym.Spaces): Defines the observation space of the agent(s)
* ``action_space`` (gym.Spaces): Defines the action space of the agent(s)

At the time only ``Discrete`` and ``Box`` spaces are admitted.

Markov Games formalism 
------------------------
MARL-API project is based on the formalism of Markov games. 
Thus, in the multi-agent case, we consider that each agent perceive a specific reward and we do not consider explicit communication channel. 

.. warning:: Markov games formalism implies that the *next_observation*, the *reward* and the *is_done* returned by ``step`` function in the environment (see above) are of type ``list`` and are not single values.

In order to work with other formalisms such as **Dec-POMDP** or **Dec-POMDP-Com**, we need to adapt the environment to fit 
above requirements. For instance, transform **Dec-POMDP** formalism into **Markov Game** one consists in giving to each and every agents the common reward. 