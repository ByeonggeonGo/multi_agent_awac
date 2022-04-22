import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from typing import List


class Multi_AWAC():
    def __init__(
        self,
        agent_num: int,
        hidden_structure: List[int],
        input_shape: int,
        output_shape: int,
        act_function: str,
        optimizer: keras.optimizers.Adam(learning_rate=3 * 1e-4),
        gamma: float = 0.9,
        state_len = int,
        ):
        self.agent_num = agent_num
        self.hidden_structure = hidden_structure
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.act_function = act_function
        self.state_len = state_len

        self.actor_list = []
        self.critic_qnet_list = []
        self.critic_qnet_target_list = []

    def make_dense_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        
        for i, val in enumerate(self.hidden_structure):
            n_percep = val 
            if i == 0:
                x = keras.layers.Dense(n_percep, activation= self.act_function)(inputs)

            elif  i != 0 and i != len(self.hidden_structure)-1:
                x = keras.layers.Dense(n_percep, activation= self.act_function)(x)

            elif  i == len(self.hidden_structure)-1:
                x = keras.layers.Dense(n_percep,)(x)#activation= outlayer_act_function)(x)
        x = keras.layers.Dense(self.output_shape,)(x)

        dense_model = tf.keras.Model(inputs=inputs, outputs=x)
        return dense_model

    def build_structure(self):
        for i in range(self.agent_num):
            self.actor_list.append(self.make_dense_model())
            self.critic_qnet_list.append(self.make_dense_model())
            self.critic_qnet_target_list.append(self.make_dense_model())

    
    def train(self, transition_sample, tau, optimizer):
        state_len = self.state_len
        qnet_weight_list = []
        
        loss_list = []
        # joint loss의 gradient 계산
        for i in range(self.agent_num):
            room_data = transition_sample[i]
            with tf.GradientTape() as t:
                # Trainable variables are automatically tracked by GradientTape
                qnet = self.critic_qnet_list[i]
                qnet_target = self.critic_qnet_target_list[i]
                qnet_weight_list.append(qnet.trainable_variables)
                

                s = room_data[:state_len]
                ns = room_data[state_len:state_len*2]
                a = room_data[state_len*2]
                r = room_data[state_len*2+1]
                done = room_data[-1]
                loss = self.cal_loss(qnet, qnet_target,s,ns,a,r,done)
                loss_list.append(loss)
        joint_loss = np.mean(loss_list)

        grads= t.gradient(joint_loss, qnet_weight_list)
            
        # 그라디언트 적용하여 qnet 수정하고 각 레이어별로 qnet_target의 웨이트 소프트 업데이트
        for i in range(self.agent_num):
            new_target_weights = []
            #큐넷 업데이트
            optimizer.apply_gradients(zip(grads[i], self.critic_qnet_list[i].trainable_variables))
            #각 레이어별로 타겟네트워크 웨이트 소프트 업데이트
            for j,target_weights in enumerate(self.critic_qnet_target_list[i].trainable_weights):
                weights = self.critic_qnet_list[i].trainable_weights[j]
                updated_target_weights = target_weights*(1 - tau) + weights*tau
                new_target_weights.append(updated_target_weights)
                
            self.critic_qnet_target_list[i].set_weights(new_target_weights)

        # actor network 업데이트
    
    
    
    def cal_loss(self,qnet,qnet_target, state, next_states, action, reward, dones):
        with tf.GradientTape() as t:
            qs = qnet_target(next_states)  # [minibatch size x #.actions]
            sampled_as = self.get_action(next_states, self.num_action_samples)  # [ minibatch size x #. action samples]
            mean_qsa = qs.gather(1, sampled_as).mean(dim=-1, keepdims=True)  # [minibatch size x 1]
            q_target = reward + self.gamma * mean_qsa * (1 - dones)

        # 식 3에서 봤던 것처럼 mse형태로 로스함수 설정
        q_val = qnet(state).gather(1, action)
        loss = loss_fun(q_val, q_target)
        return loss

    def train_loop(self, transition_matrix, optimizer, loss_fun):
        # 트레인함수 적용하여 데이터셋에서 샘플수만큼 루프돌기, 이때 샘플은 전체 메모리에서 랜덤샘플링한 배치데이터
        tau = 0.1
        for i in range(len(transition_matrix)):
            iter_sample = transition_matrix[i]
            self.train(iter_sample,tau,optimizer)
