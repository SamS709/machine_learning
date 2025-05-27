
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import deque
import tensorflow_probability as tfp
import gymnasium as gym
import keras
from gymnasium import *





class Critic(keras.Model):

    def __init__(self,n_obs=0,n_outputs=0,n_layers=1,lr=1e-6,**kwargs):
        super().__init__(**kwargs)
        self.input_layer = tf.keras.layers.Dense((512), activation="elu", kernel_initializer="he_normal",use_bias=True)
        self.hidden_layers = [tf.keras.layers.Dense(512, activation="elu",kernel_initializer="he_normal") for i in range(n_layers)]
        self.output_layer = tf.keras.layers.Dense(1)
    
    def call(self, states, actions):
        X = tf.concat([states,actions],axis = 1)
        X = self.input_layer(X)
        for hidden_layer in self.hidden_layers:
            X = hidden_layer(X)
        X = self.output_layer(X)
        return X
    
critic = Critic()
state = np.array([1.,2.,3.])
action = np.array([1.5])
pred = critic(state[np.newaxis],action[np.newaxis])
print(pred)






