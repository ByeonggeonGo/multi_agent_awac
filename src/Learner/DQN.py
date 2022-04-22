import numpy as np
import torch
import torch.nn as nn
from src.utils.train_utils import soft_update

class DQN(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 qnet: nn.Module,
                 qnet_target: nn.Module,
                 lr: float,
                 gamma: float,
                 epsilon: float,):
        """
        :param state_dim: input state dimension
        :param action_dim: action dimension
        :param qnet: main q network
        :param qnet_target: target q network
        :param lr: learning rate
        :param gamma: discount factor of MDP
        :param epsilon: E-greedy factor
        """
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = qnet
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.register_buffer('epsilon', torch.ones(1,) * epsilon)

        # target network related
        qnet_target.load_state_dict(qnet.state_dict())
        self.qnet_target = qnet_target
        self.criteria = nn.SmoothL1Loss()

    def get_action(self, state):

        qs = self.qnet(state)

        prob = np.random.uniform(0.0, 1.0, 1)
        if torch.from_numpy(prob).float() <= self.epsilon:  # random
            action = np.random.choice(range(self.action_dim))
        else:  # greedy
            action = qs.argmax(dim=-1)
        return int(action)

    #DQN offline 학습도 보기위해 살짝 수정
    def update(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        # compute Q-Learning target with 'target network'
        with torch.no_grad():
            #next state의 reward 계산
            q_max, _ = self.qnet_target(ns).max(dim=-1, keepdims=True)
            q_target = r + self.gamma * q_max * (1 - done)

        # 여기서 qnet에 state를 입력해서 나온 action에 gather라는 메쏘드로 q_val을 구함
        # qnet을 통해 reward 추출
        q_val = self.qnet(s).gather(1, a)

        # SmoothL1Loss()계산
        # smoothl1loss?
        loss = self.criteria(q_val, q_target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        #target부분 수정
        tau = 5 * 1e-3
        for param_target, param in zip(self.qnet_target.parameters(), self.qnet.parameters()):
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
                
        

        return loss
