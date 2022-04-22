import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.utils.train_utils import soft_update

# 파이토치로 구현 
class AWAC(nn.Module):

    def __init__(self,
                 critic: nn.Module,  # Q(s,a)
                 critic_target: nn.Module,
                 actor: nn.Module,  # pi(a|s)
                 lam: float = 0.3,  # Lagrangian parameter
                 tau: float = 5 * 1e-3,
                 gamma: float = 0.9,
                 num_action_samples: int = 1,
                 critic_lr: float = 3 * 1e-4,
                 actor_lr: float = 3 * 1e-4,
                 use_adv: bool = False):
        super(AWAC, self).__init__()

        # 리워드 함수 estimator
        self.critic = critic
        self.critic_target = critic_target
        self.critic_target.load_state_dict(critic.state_dict())
        self.critic_opt = torch.optim.Adam(params=self.critic.parameters(), lr=critic_lr)

        # action 함수 estimator
        self.actor = actor
        self.actor_opt = torch.optim.Adam(params=self.actor.parameters(), lr=actor_lr)

        assert lam > 0, "Lagrangian parameter 'lam' requires to be strictly larger than 0.0"
        self.lam = lam
        self.tau = tau
        self.gamma = gamma
        self.num_action_samples = num_action_samples
        self.use_adv = use_adv

    # actor에 state를 넣으면 action 출력(action은 결국 reward함수의 input으로 볼 수 있음)
    def get_action(self, state, num_samples: int = 1):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        return dist.sample(sample_shape=[num_samples]).T

    # reward함수
    def update_critic(self, state, action, reward, next_states, dones):
        with torch.no_grad():
            qs = self.critic_target(next_states)  # [minibatch size x #.actions]
            sampled_as = self.get_action(next_states, self.num_action_samples)  # [ minibatch size x #. action samples]
            mean_qsa = qs.gather(1, sampled_as).mean(dim=-1, keepdims=True)  # [minibatch size x 1]
            q_target = reward + self.gamma * mean_qsa * (1 - dones)

        # 식 3에서 봤던 것처럼 mse형태로 로스함수 설정
        q_val = self.critic(state).gather(1, action)
        loss = F.mse_loss(q_val, q_target)

        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

        # 그래디언트 구해서 critic 업데이트
        # target network update
        soft_update(self.critic, self.critic_target, self.tau)

        return loss

    # action함수
    # action 자체를 적합하는 것이 아니라 결국 액션에 따른 예상 reward를 구해서 적합한다는 점이 DQN학습방식과 차이점
    def update_actor(self, state, action):
        # state를 넣으면 action 출력
        logits = self.actor(state)
        log_prob = Categorical(logits=logits).log_prob(action.squeeze()).view(-1, 1)

        with torch.no_grad():
            if self.use_adv:
                # critic을 통해 reward 계산 이 식에서는 최종 adv를 최적화함
                # 결국  critic의 input으로 actor의 output이 사용되는 방식
                qs = self.critic_target(state)  # [#. samples x # actions]
                action_probs = F.softmax(logits, dim=-1)
                vs = (qs * action_probs).sum(dim=-1, keepdims=True)
                qas = qs.gather(1, action)
                adv = qas - vs
            else:
                adv = self.critic_target(state).gather(1, action)

            weight_term = torch.exp(1.0 / self.lam * adv)

        # 식 10
        # 실제 이 값은 커질수록 좋기때문에 -1을 곱해서 처리함
        loss = (log_prob * weight_term).mean() * -1

        # 식 10에 따라 모델 업데이트
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        return loss
