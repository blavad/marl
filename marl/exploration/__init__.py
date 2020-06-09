from marl.exploration.expl_process import ExplorationProcess, make, available, register
from marl.exploration.eps_greedy import EpsGreedy, Greedy, EpsExpert, Expert, EpsExpertEpsGreedy, HierarchicalEpsGreedy, EpsSoftmax, Softmax
from marl.exploration.ou_noise import OUNoise
from marl.exploration.expls import UCB1

    
register(
    id='Greedy',
    entry_point='marl.exploration.eps_greedy:Greedy'
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
    id='Softmax',
    entry_point='marl.exploration.eps_greedy:Softmax',
)

register(
    id='EpsSoftmax',
    entry_point='marl.exploration.eps_greedy:EpsSoftmax',
)

register(
    id='UCB1',
    entry_point='marl.exploration.expls:UCB1',
)

register(
    id='Expert',
    entry_point='marl.exploration.eps_greedy:Expert',
)

register(
    id='EpsExpert',
    entry_point='marl.exploration.eps_greedy:EpsExpert',
)

register(
    id='HierarchicalEpsGreedy',
    entry_point='marl.exploration.eps_greedy:HierarchicalEpsGreedy',
)

register(
    id='EpsExpertEpsGreedy',
    entry_point='marl.exploration.eps_greedy:EpsExpertEpsGreedy',
)

register(
    id='OUNoise',
    entry_point='marl.exploration.ou_noise:OUNoise'
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
