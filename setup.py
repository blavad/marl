from setuptools import setup, find_packages

setup(name='marl',
      version='0.0.1',
      description='Multi-Agent Reinforcement Learning',
      url='https://github.com/blavad/marl',
      author='David Albert',
      author_email='david.albert@insa-rouen.fr',
      packages=find_packages(),
      install_requires=['gym', 'numpy','torch', 'torchvision']
)
