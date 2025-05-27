import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import cv2
import keras as keras
from gymnasium import *

class Actor(keras.Model):

    def __init__(self,n_obs,n_outputs,n_layers=1,action_range=1,lr=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.input_layer = tf.keras.layers.Dense(512, activation="elu",kernel_initializer="he_normal")
        self.hidden_layers = [tf.keras.layers.Dense(512, activation="elu",kernel_initializer="he_normal") for i in range(n_layers)]
        self.prob = tf.keras.layers.Dense(n_outputs, activation="tanh",kernel_initializer="he_normal")
        self.action = tf.keras.layers.Lambda(lambda x: tf.multiply(x,action_range))

    def call(self, states):
        X = self.input_layer(states)
        for hidden_layer in self.hidden_layers:
            X = hidden_layer(X)
        X = self.prob(X)
        X = self.action(X)
        return X


"""try:
    self.actor = keras.models.load_model(self.model_name+"_actor")
    print(f'[INFO] : Model "{self.model_name}" loaded')
except:
    self.actor = Actor()
    self.actor.save(self.model_name+"_actor")
    print(f'[INFO] : new Model : "{self.model_name}" created')
self.actor_optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
"""

class Critic(keras.Model):

    def __init__(self,n_obs=0,n_outputs=0,n_layers=1,lr=1e-6, **kwargs):
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
        
"""try:
    self.critic = keras.models.load_model(self.model_name+"_critic")
except:
    self.critic = Critic()
self.critic_optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
"""
class Replay_buffer:

    def __init__(self,max_len,batch_size):
        self.memory = deque(maxlen=max_len)
        self.batch_size = batch_size

    def append(self,tuple):
        self.memory.append((tuple))

    def sample_experiences(self):
        indexes = np.random.randint(len(self.memory),size=self.batch_size)
        batch = [self.memory[index] for index in indexes]
        return [
                np.array([experience[field_index] for experience in batch])
                for field_index in range(len(self.memory[0]))
            ]    
    
class Agent:

    def __init__(self,env,model_name,gamma = 0.99,lr_a = 0.001,lr_c = 0.002, noise = 0.1,
                 max_len = 1000000, tau = 0.005, batch_size = 64):
        
        self.model_name = model_name
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.batch_size = batch_size
        self.replay_buffer = Replay_buffer(max_len = max_len, batch_size= batch_size)
        
        n_obs = env.reset(seed=42)[0].shape[0]
        n_layers= 1
        action_range = 0
        if type(env.action_space)==spaces.Box:
            self.n_outputs = env.action_space.shape[0]
            self.action_range = env.action_space.high[0]
        else:
            print(["[ERROR] This DPGP is only avaible for continous action spaces."])
            return
        
        try:
            self.load_models()
        except:
            # creating models
            self.critic = Critic(n_obs = n_obs, n_outputs = self.n_outputs, n_layers = n_layers, lr= lr_c)
            self.actor = Actor(n_obs = n_obs, n_outputs = self.n_outputs, n_layers=n_layers, lr= lr_a, action_range = action_range)
            # creating target models
            self.target_actor = keras.models.clone_model(self.actor)
            self.target_critic = keras.models.clone_model(self.critic)
            # building models by passing data
            state, info = env.reset()
            action = self.actor(state[np.newaxis])
            new_state = self.critic(state[np.newaxis],action)
            action = self.target_actor(state[np.newaxis])
            new_state = self.target_critic(state[np.newaxis],action)
            self.update_target_networks(tau = 1)
            self.save_models()
            print(f'[INFO] New model "{self.model_name}" has successfully been created.')
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_c)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_a)


        
    
    def update_target_networks(self, tau=None):

        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)
 
    def choose_action(self,state, evaluate = False):
        
        action = self.actor(state[np.newaxis])
        if not evaluate:
            action += tf.random.normal(shape=[self.n_outputs],mean = 0., stddev= self.noise)
        action = tf.clip_by_value(action, -self.action_range, self.action_range)
        return action[0]
    
    def load_models(self):
        self.critic = keras.models.load_model(self.model_name+"_critic")
        self.target_critic = keras.models.load_model(self.model_name+"_target_critic")
        self.actor = keras.models.load_model(self.model_name+"_actor")
        self.target_actor = keras.models.load_model(self.model_name+"_target_actor")
        print(f'[INFO] Model "{self.model_name}" loaded.')
        
    def save_models(self):
        self.actor.save(self.model_name+"_actor",overwrite=True)
        self.critic.save(self.model_name+"_critic",overwrite=True)
        self.target_actor.save(self.model_name+"_target_actor",overwrite=True)
        self.target_critic.save(self.model_name+"_target_critic",overwrite=True)

    
    def learn(self):

        if len(self.replay_buffer.memory) < self.batch_size:
            return
        
        experiences = self.replay_buffer.sample_experiences()
        states, actions, rewards, next_states, dones, truncateds = experiences
        runs = 1 - (dones | truncateds)
        with tf.GradientTape() as tape:
            next_actions = self.target_actor(next_states)
            Q_val = tf.squeeze(self.critic(states, actions),1)
            next_Q_val = tf.squeeze(self.target_critic(next_states, next_actions),1)
            target_Q_val = rewards + runs*self.gamma*next_Q_val
            critic_loss = keras.losses.MSE(target_Q_val,Q_val)
        critic_grad = tape.gradient(critic_loss,self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(
            critic_grad,self.critic.trainable_variables
        ))
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states,new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
        actor_grad = tape.gradient(actor_loss,self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(
            actor_grad,self.actor.trainable_variables
        ))
        self.update_target_networks()

    
class Player:

    def __init__(self, env, model_name):
        self.env = env
        self.agent = Agent(env, model_name = model_name)
        self.replay_buffer = self.agent.replay_buffer

    def play_one_step(self, state, deterministic = False):
        action = self.agent.choose_action(state, deterministic)
        next_state, reward, done, truncated, info = self.env.step(action)
        self.replay_buffer.append((state, action, reward, next_state, done, truncated))
        return next_state, reward, done, truncated, info 

    def train_n_games(self,n,figure_file):
        rewards = []
        best_avg_score = -1000000
        for episode in range(n):
            obs, info = self.env.reset()   
            game_reward = 0
            done = False
            truncated = False
            N = 0
            while N <= 205 and done == False and truncated == False:
                obs, reward, done, truncated, info = self.play_one_step(obs)
                game_reward+=reward
                N+=1
                self.agent.learn()
            rewards.append(game_reward)
            self.game_mus = []
            self.game_sigmas = []
            avg_score = np.mean(rewards[-100:])
            if avg_score >= best_avg_score and episode >= 20:
                best_avg_score = avg_score
                print("[INFO] Target variables updated.")
                self.agent.save_models()
            print(f"Episode: {episode + 1}, Steps: {N + 1}, reward: {game_reward}, avg_score: {avg_score}")
        plt.figure("Rewards")
        plt.plot(range(len(rewards)),rewards,label="Rewards per step",color = "green")
        plt.legend()
        plt.savefig(figure_file)
        plt.show()

    def make_video(self,video_name):
        video = cv2.VideoWriter(video_name+str(".avi"),cv2.VideoWriter_fourcc(*'MJPG'),10,(600,400))
        obs, info = self.env.reset()
        for step in range(200):
            obs, reward, done, truncated, info = self.play_one_step(obs,deterministic = True)
            img = self.env.render()
            img = cv2.resize(img,(600,400))
            video.write(img)
            if done or truncated:
                break
        video.release()
        cv2.destroyAllWindows() 

if __name__=="__main__":
    env = gym.make("Pendulum-v1")
    model_name = "my_DPGP_model"
    plot_file = "figure1"
    n = 250
    player = Player(env,model_name)
    player.train_n_games(n,plot_file)


    

        
