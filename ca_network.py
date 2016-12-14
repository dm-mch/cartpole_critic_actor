"""
Critic Actor network for Cart Pole balance task
Author: Dmitriy Movchan
 
"""

import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.regularizers import l2, l1

H1_LAYER = 4*4
H2_LAYER = 4*2

class CANetwork(object):

    def __init__(self,sess, action_dim = 2, state_size = 4, tau = 0.001, lr = 0.001):
        self.TAU = tau
        self.LEARNING_RATE = lr
        self.sess = sess
        self.action_dim = action_dim
        self.state_size = state_size
        
        self.debug = []
        self.entropy_betta = 0.1

        K.set_session(self.sess)

        self.policy_network, self.value_network, self.states = self.create_network(state_size, action_dim, name = 'train')
        self.target_policy_network, self.target_value_network, self.target_states = self.create_network(state_size, action_dim, name = 'target', summury = False)

        self.p_out = self.policy_network(self.states)
        self.v_out = self.value_network(self.states)

        self.tp_out = self.target_policy_network(self.states)
        self.tv_out = self.target_value_network(self.states)

        self.compile()


    def compile(self):
        """
        Prepare optimizer and loss functiun for both nets

        """
        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        
        self.rewards = tf.placeholder("float", [None])
        self.actions = tf.placeholder("float", [None, self.action_dim])

        log_prob = tf.log(tf.clip_by_value(self.p_out,1e-10,1e+10))

        # Policy entropy
        entropy = tf.reduce_sum(self.p_out * log_prob, reduction_indices=1)

        adv = self.rewards - tf.reshape(self.v_out, (-1,))
        self.p_loss = -tf.reduce_sum( tf.reduce_sum( tf.mul( log_prob, self.actions ), reduction_indices=1 ) * adv + entropy * self.entropy_betta)
        self.v_loss = 0.5 * tf.nn.l2_loss(adv)

        self.debug += [tf.Print(self.actions, [self.actions], message = 'actions')]
        self.debug += [tf.Print(self.p_loss, [self.p_loss], message = 'p_loss')]
        self.debug += [tf.Print(self.v_loss, [self.v_loss], message = 'v_loss')]

        self.total_loss = self.p_loss + self.v_loss

        self.minimize = [self.optimizer.minimize(self.total_loss)]

        self.sess.run(tf.initialize_all_variables())
    

    def create_network(self, state_size, action_dim, name = '', summury = True):
        """
        Create Policy and Value net

        """
        # Regularization
        lv = 0.1
        ll = l1
        
        input = Input(shape=(state_size, ))

        # Shared layer for both policy and value net
        shared = Dense(input_dim = state_size, output_dim=H1_LAYER, name=name + "/h1", activation='relu', W_regularizer = ll(lv))(input)

        # Policy 
        pd  = Dense(output_dim=H2_LAYER, name=name + "/ph", activation='relu', W_regularizer = ll(lv))(shared)
        action_probs = Dense(name = name + '/policy', output_dim=action_dim, activation='softmax', W_regularizer = ll(lv))(pd)
        
        # Value
        vd  = Dense(output_dim=H2_LAYER, name=name + "/ph", activation='relu', W_regularizer = ll(lv))(shared)
        state_value = Dense(name = name + '/value', output_dim=1, activation='linear', W_regularizer = ll(lv))(vd)

        policy_network = Model(input=input, output=action_probs)
        value_network = Model(input=input, output=state_value)

        return policy_network, value_network, input


    def train(self, states, actions, rewards):
        ops = self.minimize + [self.p_out] # main train op
        #ops += self.debug # debug prints

        r = self.sess.run(ops, feed_dict={
            self.states: states,
            self.actions: actions,
            self.rewards: rewards
        })

    def train_target(self):
        self.update_target(self.policy_network, self.target_policy_network)
        self.update_target(self.value_network, self.target_value_network)

    def update_target(self, source, target):
        """
        Update target weights from source with self.TAU

        """
        source_weights = source.get_weights()
        target_weights = target.get_weights()
        for i in range(len(source_weights)):
            target_weights[i] = self.TAU * source_weights[i] + (1 - self.TAU) * target_weights[i]
        target.set_weights(target_weights)

    def predict_policy(self, states):
        """
        Via target network

        """
        return self.sess.run(self.tp_out, feed_dict={
            self.states: states
        })

    def predict_value(self, states):
        """
        Via target network

        """
        return self.sess.run(self.tv_out, feed_dict={
            self.states: states
        })


def test():
    print("Start test")
    
    import numpy as np
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=''


    sess = tf.Session()
    K.set_session(sess)

    print("Create network")
    ca = CANetwork(sess)
    states = np.random.randn(10, 4)
    actions = np.random.rand(10, 2)
    rewards = np.random.randn(10)

    print("Train:", ca.train(states, actions, rewards))
    ca.train_target()

    print("Predict:", ca.predict(states))

if __name__ == '__main__':
    test()


