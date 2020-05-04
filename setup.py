from setuptools import setup, find_packages

setup(name='marl',
      version='0.0.1',
      description='Multi-Agent Reinforcement Learning',
      url='https://github.com/blavad/marl',
      author='David Albert',
      author_email='david.albert@insa-rouen.fr',
      packages=['marl', 'marl.agent', 'marl.experience', 'marl.exploration', 'marl.policy', 'marl.model','marl.model.nn'],
      install_requires=['gym', 'numpy','torch', 'torchvision']
)
