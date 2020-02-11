from marl.exploration.expl_process import ExplorationProcess, make, available, register
from marl.exploration.greedy import Greedy
from marl.exploration.eps_greedy import EpsGreedy
from marl.exploration.ou_noise import OUNoise

    
register(
    id='Greedy',
    entry_point='marl.exploration.greedy:Greedy'
)

register(
    id='EpsGreedy-cst01',
    entry_point='marl.exploration.eps_greedy:EpsGreedy',
    eps_deb=0.1,
    eps_fin=0.1
)

register(
    id='EpsGreedy-cst1',
    entry_point='marl.exploration.eps_greedy:EpsGreedy',
    eps_deb=1.,
    eps_fin=1.
)

register(
    id='EpsGreedy-cst05',
    entry_point='marl.exploration.eps_greedy:EpsGreedy',
    eps_deb=0.5,
    eps_fin=0.5
)

register(
    id='EpsGreedy-cst02',
    entry_point='marl.exploration.eps_greedy:EpsGreedy',
    eps_deb=0.2,
    eps_fin=0.2
)

register(
    id='EpsGreedy-cst002',
    entry_point='marl.exploration.eps_greedy:EpsGreedy',
    eps_deb=0.02,
    eps_fin=0.02
)

register(
    id='EpsGreedy-cst001',
    entry_point='marl.exploration.eps_greedy:EpsGreedy',
    eps_deb=0.01,
    eps_fin=0.01
)

register(
    id='EpsGreedy',
    entry_point='marl.exploration.eps_greedy:EpsGreedy',
)

register(
    id='EpsGreedy-lin',
    entry_point='marl.exploration.eps_greedy:EpsGreedy',   
)

register(
    id='OUNoise',
    entry_point='marl.exploration.ou_noise:OUNoise'
)