"""
Balance Cart Pole task via Critic Actor algorithm
Author: Dmitriy Movchan

"""

import gym
import numpy as np
import random

from keras.layers.core import Dense
from keras.optimizers import Adam
import tensorflow as tf

from replay import Buffer
from ca_network import CANetwork

# Switch off GPU usage for CUDA, only CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]=''


def get_eps(mean):
    """
    mean - mean 100 value 
    Epsilon for e-greedy police

    """
    eps = 0
    if mean <= 20:
        eps = 0.8
    elif mean <= 30:
        eps = 0.3
    elif mean <= 40:
        eps = 0.2
    else:
        eps = 0.1
    return eps

def is_exploer(eps):
    r = np.random.rand()
    if r <= eps:
        return True
    return False

def get_state(observation):
    """
    Normalize obzervation
    """
    hight = np.array([ 2.45,  4,  0.28,  4 ])
    return observation/hight
    

def eval():
    REPLAY_BUFFER_SIZE = 100000
    BATCH_SIZE = 512
    TAU = 0.01     # Update target network param 
    LR = 0.001     # Lerning rate

    action_dim = 2  # Left/Right
    state_dim = 4  # for card-pole

    episode_count = 500
    max_steps = 500
    train_on_episode = 4
    reward = 0

    sess = tf.Session()
    from keras import backend as K
    K.set_session(sess)

    # Create Actor-Critic network
    ac_net = CANetwork(sess, action_dim, state_dim, TAU, LR)
    # Create replay buffer
    buff = Buffer(REPLAY_BUFFER_SIZE)

    # CartPole environment
    env = gym.make('CartPole-v1')
    
    # Record for submission
    #env.monitor.start('../cartpole-experiment', force=True)

    rewards_all = []
    mean_rewards = 0
    print("CardPole Experiment Start.")
    for ep in range(episode_count):

        ob = env.reset()
        s_t = get_state(ob)
        # total episode reward
        total_reward = 0.
        # (state, action, next_state) buffer for whole episode
        ep_buffer = []

        for step in range(max_steps): 
            # Action
            a_t = np.zeros([action_dim])
            if is_exploer(get_eps(mean_rewards)):
                a_t[np.random.choice([0,1])] = 1
            else:
                a_t = ac_net.predict_policy(s_t.reshape(1, s_t.shape[0]))[0]

            ob, r_t, done, info = env.step(np.argmax(a_t))
            s_t1 = get_state(ob)        

            ep_buffer.append((s_t, a_t, s_t1))
            total_reward += r_t
            s_t = s_t1
        
            if done:
                break
        rewards_all.append(total_reward)

        # Calc reward as inerted range
        ep_rw = np.arange(step)[::-1]
        if step == max_steps - 1:
            # if not fail - predict value from last state
            ls = ep_buffer[-1][2] # last observed state
            ep_rw = ep_rw.astype(np.float32) + ac_net.predict_value(ls.reshape(1, ls.shape[0]))[0]
            print("Last", ep_rw[-1]) 
        
        # Add all episode states to replay buffer 
        for i in range(step):
            buff.add((ep_buffer[i][0],
                      ep_buffer[i][1],
                      ep_rw[i],
                      ep_buffer[i][2]))

        # train on batchs from replay buffer
        for i in range(train_on_episode):
            batch = buff.get_batch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])

            ac_net.train(states, actions, rewards)

        # Only one update for target network    
        ac_net.train_target()

        mean_rewards = np.mean(rewards_all[-100:])
        print(str(ep) +"-th Episode  : Reward " + str(total_reward), "eps", get_eps(mean_rewards) ,"Mean reward: ", mean_rewards)
        if mean_rewards > 495:
            print("EVRIKA!")
            break    

    #env.monitor.close()
    print("Finish.")

if __name__ == "__main__":
    eval()
