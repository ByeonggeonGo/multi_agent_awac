import tensorflow as tf
from tensorflow.python.client import device_lib
import os
from glob import glob
import pandas as pd
import numpy as np
import dask.dataframe as dd
from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp
from time import time

print(tf.__version__)
print(device_lib.list_local_devices())

# params of multi_AWAC
agent_num = 5
state_len = 2
hidden_structure= [64, 128, 256, 128, 64]
input_shape= state_len*agent_num
output_shape= 2
act_function= 'relu'
lam = 0.3
optimizer= keras.optimizers.Adam(learning_rate=3 * 1e-4)
gamma= 0.9
tau = 0.1
num_action_samples = 8
loss_fun = tf.keras.losses.MeanSquaredError()


# dense model 만드는 함수
def make_dense_model(hidden_structure,input_shape,act_function,output_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    for i, val in enumerate(hidden_structure):
        n_percep = val 
        if i == 0:
            x = keras.layers.Dense(n_percep, activation= act_function)(inputs)

        elif  i != 0 and i != len(hidden_structure)-1:
            x = keras.layers.Dense(n_percep, activation= act_function)(x)

        elif  i == len(hidden_structure)-1:
            x = keras.layers.Dense(n_percep,activation= act_function)(x)
    x = keras.layers.Dense(output_shape)(x)

    dense_model = tf.keras.Model(inputs=inputs, outputs=x)
    return dense_model

# multi_agent structure 만드는 함수
def build_structure(agent_num, hidden_structure,input_shape,act_function,output_shape):
    actor_list = []
    critic_qnet_list = []
    critic_qnet_target_list = []
    for i in range(agent_num):
        actor_list.append(make_dense_model(hidden_structure,input_shape,act_function,output_shape))
        critic_qnet_list.append(make_dense_model(hidden_structure,input_shape,act_function,output_shape))
        critic_qnet_target_list.append(make_dense_model(hidden_structure,input_shape,act_function,output_shape))
    return actor_list, critic_qnet_list, critic_qnet_target_list


def get_action(actor,state, num_samples: int = 3):
    logit_sam = actor(state)
    m = tfp.distributions.Categorical(logits = logit_sam)
    return tf.reshape(m.sample(num_samples),[-1,num_samples])

def get_mean_qsa(qs,sampled_as):
    mean_q = tf.concat([tf.reshape(tf.gather(qs[i],sampled_as[i], axis=0),[1,-1]) for i in range(len(qs))],axis=0)
    mean_q = tf.math.reduce_mean(mean_q,axis=1,keepdims=True)
    return mean_q




# CPU 학습
print("CPU를 사용한 학습")
with tf.device("/device:CPU:0"):
    start_time = time()
    actor_list, critic_qnet_list, critic_qnet_target_list = build_structure(agent_num, hidden_structure,input_shape,act_function,output_shape)
    # 메모리에서 아래와같이 샘플링됐다고 가정하고 테스트(메모리 아직 안만들어짐)
    s = np.array([[1,2,1,2,1,2,1,2,1,2,],[2,3,1,2,1,2,3,2,6,7,],[1,4,14,2,1,2,11,2,9,2,]])
    ns = np.array([[1,29,1,2,10,2,1,29,1,28,],[17,2,16,2,1,21,1,2,1,22,],[1,21,1,22,1,2,7,2,5,2,]])
    a = np.array([[0],[1],[1],])
    r = np.array([[2],[4.2],[2.7],])
    done = np.array([[0],[0],[0],])
    sample_dataset_by_agent = [s, ns, a, r, done]
    dataset_list = [sample_dataset_by_agent,sample_dataset_by_agent,sample_dataset_by_agent,sample_dataset_by_agent,sample_dataset_by_agent]
    epoch = 100
    for i in range(epoch):
        with tf.GradientTape() as t:
            loss_list =[]
            for j in range(agent_num):
                ##데이터 
                data = dataset_list[j]
                s = data[0]
                ns = data[1]
                a = data[2]
                r = data[3]
                done = data[4]
                ##에이전트
                actor = actor_list[j]
                critic_qnet = critic_qnet_list[j]
                critic_qnet_target = critic_qnet_target_list[j]
                ##로스 계산
                qs = critic_qnet_target(ns)
                sampled_as = get_action(actor,ns, num_action_samples)
                mean_qsa = get_mean_qsa(qs,sampled_as)
                q_target = r + gamma * mean_qsa * (1 - done)

                # 식 3에서 봤던 것처럼 mse형태로 로스함수 설정
                q_val = tf.concat([tf.reshape(tf.gather(critic_qnet(s)[k],a[k],axis=0),[-1,1]) for k in range(len(s))],axis=0)
                loss = loss_fun(q_val, q_target)
                loss_list.append(loss)
            joint_loss = tf.math.reduce_mean(loss_list, axis=None, keepdims=False, name=None)

        critic_qnet_weight_list = [critic_qnet_list[p].trainable_variables for p in range(agent_num)]
        critic_qnet_target_weight_list = [critic_qnet_target_list[p].trainable_variables for p in range(agent_num)]
        grads = t.gradient(joint_loss, critic_qnet_weight_list)

        for q in range(agent_num):
            # qnet 업데이트
            qnet_weights = critic_qnet_weight_list[q]
            qnet_target_weights = critic_qnet_target_weight_list[q]
            grad = grads[q]

            optimizer.apply_gradients(zip(grad, qnet_weights))
            # target net 업데이트
            new_target_weights = []
            for p, target_weights in enumerate(qnet_target_weights):
                qnet_weights_s = qnet_weights[p]
                updated_target_weights_s = target_weights*(1 - tau) + qnet_weights_s*tau
                new_target_weights.append(updated_target_weights_s)
            critic_qnet_target_list[q].set_weights(new_target_weights)

            #타겟네트워크까지 업데이트한 후 actor net 업데이트
            with tf.GradientTape() as tp:
                # log_probability 계산
                logits = actor_list[q](s)
                m = tfp.distributions.Categorical(logits = logits)
                log_prob = tf.reshape(m.log_prob(a.squeeze()),[-1,1])

                #가중치항 계산
                qs = critic_qnet_target_list[q](s)
                action_probs = tf.nn.softmax(logits, axis=None, name=None)
                vs = tf.math.reduce_sum((qs * action_probs),axis=1, keepdims=True, name=None)
                qas = tf.concat([tf.reshape(tf.gather(qs[k],a[k],axis=0),[-1,1]) for k in range(len(s))],axis=0)
                adv = qas - vs
                weight_term = tf.math.exp((1/lam*adv), name=None)

                #loss
                loss = tf.math.reduce_mean(log_prob * weight_term*-1)
            actor_grad = tp.gradient(loss, actor_list[q].trainable_variables)
            optimizer.apply_gradients(zip(actor_grad, actor_list[q].trainable_variables))
    print(time() - start_time)        

print("GPU를 사용한 학습")
with tf.device("/device:GPU:0"):
    start_time = time()
    actor_list, critic_qnet_list, critic_qnet_target_list = build_structure(agent_num, hidden_structure,input_shape,act_function,output_shape)
    # 메모리에서 아래와같이 샘플링됐다고 가정하고 테스트(메모리 아직 안만들어짐)
    s = np.array([[1,2,1,2,1,2,1,2,1,2,],[2,3,1,2,1,2,3,2,6,7,],[1,4,14,2,1,2,11,2,9,2,]])
    ns = np.array([[1,29,1,2,10,2,1,29,1,28,],[17,2,16,2,1,21,1,2,1,22,],[1,21,1,22,1,2,7,2,5,2,]])
    a = np.array([[0],[1],[1],])
    r = np.array([[2],[4.2],[2.7],])
    done = np.array([[0],[0],[0],])
    sample_dataset_by_agent = [s, ns, a, r, done]
    dataset_list = [sample_dataset_by_agent,sample_dataset_by_agent,sample_dataset_by_agent,sample_dataset_by_agent,sample_dataset_by_agent]
    epoch = 100
    for i in range(epoch):
        with tf.GradientTape() as t:
            loss_list =[]
            for j in range(agent_num):
                ##데이터 
                data = dataset_list[j]
                s = data[0]
                ns = data[1]
                a = data[2]
                r = data[3]
                done = data[4]
                ##에이전트
                actor = actor_list[j]
                critic_qnet = critic_qnet_list[j]
                critic_qnet_target = critic_qnet_target_list[j]
                ##로스 계산
                qs = critic_qnet_target(ns)
                sampled_as = get_action(actor,ns, num_action_samples)
                mean_qsa = get_mean_qsa(qs,sampled_as)
                q_target = r + gamma * mean_qsa * (1 - done)

                # 식 3에서 봤던 것처럼 mse형태로 로스함수 설정
                q_val = tf.concat([tf.reshape(tf.gather(critic_qnet(s)[k],a[k],axis=0),[-1,1]) for k in range(len(s))],axis=0)
                loss = loss_fun(q_val, q_target)
                loss_list.append(loss)
            joint_loss = tf.math.reduce_mean(loss_list, axis=None, keepdims=False, name=None)

        critic_qnet_weight_list = [critic_qnet_list[p].trainable_variables for p in range(agent_num)]
        critic_qnet_target_weight_list = [critic_qnet_target_list[p].trainable_variables for p in range(agent_num)]
        grads = t.gradient(joint_loss, critic_qnet_weight_list)

        for q in range(agent_num):
            # qnet 업데이트
            qnet_weights = critic_qnet_weight_list[q]
            qnet_target_weights = critic_qnet_target_weight_list[q]
            grad = grads[q]

            optimizer.apply_gradients(zip(grad, qnet_weights))
            # target net 업데이트
            new_target_weights = []
            for p, target_weights in enumerate(qnet_target_weights):
                qnet_weights_s = qnet_weights[p]
                updated_target_weights_s = target_weights*(1 - tau) + qnet_weights_s*tau
                new_target_weights.append(updated_target_weights_s)
            critic_qnet_target_list[q].set_weights(new_target_weights)

            #타겟네트워크까지 업데이트한 후 actor net 업데이트
            with tf.GradientTape() as tp:
                # log_probability 계산
                logits = actor_list[q](s)
                m = tfp.distributions.Categorical(logits = logits)
                log_prob = tf.reshape(m.log_prob(a.squeeze()),[-1,1])

                #가중치항 계산
                qs = critic_qnet_target_list[q](s)
                action_probs = tf.nn.softmax(logits, axis=None, name=None)
                vs = tf.math.reduce_sum((qs * action_probs),axis=1, keepdims=True, name=None)
                qas = tf.concat([tf.reshape(tf.gather(qs[k],a[k],axis=0),[-1,1]) for k in range(len(s))],axis=0)
                adv = qas - vs
                weight_term = tf.math.exp((1/lam*adv), name=None)

                #loss
                loss = tf.math.reduce_mean(log_prob * weight_term*-1)
            actor_grad = tp.gradient(loss, actor_list[q].trainable_variables)
            optimizer.apply_gradients(zip(actor_grad, actor_list[q].trainable_variables))
    print(time() - start_time)     