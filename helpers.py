import numpy as np

# tianshou code
from tianshou.policy import SACPolicy, BasePolicy
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.common import Net
from tianshou.data import VectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.utils import WandbLogger


import torch
class NumpyDeque(object):
    def __init__(self, shape:tuple, device='cpu') -> None:
        self.shape_arr = shape

        self.array = np.zeros((self.shape_arr), dtype=np.float32)

    def __len__(self):
        return self.shape_arr[1]
    
    def append(self, els):
        assert els.shape[0] == self.shape_arr[0] 

        self.array = np.roll(self.array, els.shape[1], axis=1)
        self.array[:,0:els.shape[1]] = els.astype(np.float32)

    def reset(self, vecs=None):
        if vecs is None:
            self.array = np.zeros((self.shape_arr), dtype=np.float32)
        elif isinstance(vecs,np.ndarray):
            self.array[vecs==1] = 0.
            
    def __call__(self):
        return self.array
    def __repr__(self):
        return str(self.array)
    def __array__(self, dtype=None):
        if dtype:
            return self.array.astype(dtype)
        return self.array
    @property
    def shape(self):
        return self.shape_arr


if __name__=='__main__':
    def create_policy(env):
        observation_space = env.observation_space.shape or env.observation_space.n
        action_space = env.action_space.shape or env.action_space.n
        # create the networks behind actors and critics
        net_a = Net(state_shape=observation_space,
                    hidden_sizes=[64,64], device='cpu')
        net_c1 = Net(state_shape=observation_space,action_shape=action_space,
                        hidden_sizes=[64,64],
                        concat=True,)
        net_c2 = Net(state_shape=observation_space,action_shape=action_space,
                        hidden_sizes=[64,64],
                        concat=True,)

        # create actors and critics
        actor = ActorProb(
            net_a,
            action_space,
            unbounded=True,
            conditioned_sigma=True,
        )
        critic1 = Critic(net_c1, device='cpu')
        critic2 = Critic(net_c2, device='cpu')

        # create the optimizers
        actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
        critic_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)

        # create the policy
        policy = SACPolicy(actor=actor, actor_optim=actor_optim, \
                            critic=critic1, critic_optim=critic_optim,\
                            critic2=critic2, critic2_optim=critic2_optim,\
                            action_space=env.action_space,\
                            observation_space=env.observation_space, \
                            action_scaling=True) # make sure actions are scaled properly
        return policy

    def forward_policy(policy, state):
        return policy(state)


    test_qeue = NumpyDeque((3,5))
    print(type(test_qeue.array))
    print(test_qeue)
    ones_1 = np.ones((3,1))
    twos_2 = np.ones((3,2))*2
    test_qeue.append(ones_1)
    print(test_qeue)
    test_qeue.append(twos_2)
    print(test_qeue)
    test_qeue.reset(np.array([0,1,0]))
    print(test_qeue)

