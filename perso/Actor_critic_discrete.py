import numpy as np
import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import keras 
import gymnasium as gym
from collections import deque
import cv2
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

#https://towardsdatascience.com/actor-critic-with-tensorflow-2-x-part-1-of-2-d1e26a54ce97/

class DQN:

    def __init__(self,model_name,env_name,n_layers):
        self.env = gym.make(env_name,render_mode="rgb_array")
        #self.n_outputs = self.env.action_space.shape[0]
        n_obs = self.env.reset(seed=42)[0].shape[0]
        self.init_containers()
        self.replay_buffer = deque(maxlen=500)
        self.batch_size = 32
        self.discount_factor = 0.95
        self.model_name = model_name
        self.init_agent(n_layers,n_obs)

    def init_containers(self):
        self.game_mus = []
        self.game_sigmas = []
        self.all_mus = []
        self.all_sigmas = []
        self.losses = []
        self.rewards = []


    def init_agent(self,n_layers,n_obs):
        #Actor Initialization
        try:
            self.actor = keras.models.load_model(self.model_name+"_actor")
            print(f'[INFO] : Model "{self.model_name}" loaded')
        except:
            input_layer = tf.keras.layers.Input([n_obs],name="input_layer")
            hidden = tf.keras.layers.Dense(2048, activation="elu",kernel_initializer="he_normal")(input_layer)
            for i in range(n_layers):
                hidden = tf.keras.layers.Dense(1536, activation="elu",kernel_initializer="he_normal")(hidden)
            output = tf.keras.layers.Dense(4, activation="softmax",kernel_initializer="he_normal")(hidden)
            self.actor = tf.keras.Model(inputs=[input_layer],outputs=[output])
            self.actor.save(self.model_name+"_actor")
            print(f'[INFO] : new Model : "{self.model_name}" created')

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)

        #Critic Initialization

        try:
            self.critic = keras.models.load_model(self.model_name+"_critic")
            print(self.critic.summary())
        except:
            self.critic = tf.keras.Sequential()
            self.critic.add(tf.keras.layers.Dense(2048, activation="elu", kernel_initializer="he_normal",use_bias=False, input_shape=[n_obs]))
            for i in range(n_layers):
                self.critic.add(tf.keras.layers.Dense(1536, activation="elu", kernel_initializer="he_normal",use_bias=False))
            self.critic.add(tf.keras.layers.Dense(1))
            self.critic.save(self.model_name+"_critic")
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
        self.critic_loss_fn = tf.keras.losses.mean_squared_error
    
    

    def actor_loss_fn(self,probs, actions, deltas):
        dist = tfp.distributions.Categorical(probs=probs,dtype=tf.float32)
        # Convert pdf value to log probability
        probability = dist.prob(actions)
        log_probability = dist.log_prob(actions)
        # Compute weighted loss
        loss_actor = - deltas * log_probability #- 0.1*tf.math.log(sigmas*tf.sqrt(2*np.pi*np.e))
        e_loss = - tf.math.multiply(probability,log_probability)
        print("e_loss=",tf.reduce_mean(e_loss))
        print("log_loss=",tf.reduce_mean(loss_actor))
        return loss_actor + 0.001*e_loss

    def policy(self,state,deterministic=False):
        probs = self.actor.predict(state[np.newaxis],verbose=0)
        dist = tfp.distributions.Categorical(probs=probs,dtype=tf.float32)
        action = dist.sample()        
        return int(action)


    def play_one_step(self, state, deterministic = False):
        action = self.policy(state, deterministic)
        next_state, reward, done, truncated, info = self.env.step(action)
        self.replay_buffer.append((state, action, reward, next_state, done, truncated))
        return next_state, reward, done, truncated, info    
    
    def sample_experiences(self):
        indexes = np.random.randint(len(self.replay_buffer),size=self.batch_size)
        batch = [self.replay_buffer[index] for index in indexes]
        return [
                np.array([experience[field_index] for experience in batch])
                for field_index in range(6)
            ]    
    def training_step(self):

        experiences = self.sample_experiences()
        states, actions, rewards, next_states, dones, truncateds = experiences
        next_Q_values = self.critic.predict(next_states, verbose=0)
        runs = 1.0 - (dones | truncateds)  # episode is not done or truncated
        runs = runs.reshape([runs.size,1])
        rewards = rewards.reshape([rewards.size,1])
        target_Q_values = rewards + runs * self.discount_factor * next_Q_values
        with tf.GradientTape() as tape_critic:
            Q_values = self.critic(states)
            loss = tf.reduce_mean(self.critic_loss_fn(target_Q_values, Q_values))
        grads = tape_critic.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        self.critic.save(self.model_name+"_critic",overwrite=True)
        Q_values = self.critic.predict(states, verbose=0)
        deltas = target_Q_values-Q_values
        with tf.GradientTape() as tape_actor:
            probs = self.actor(states)
            loss = tf.math.reduce_mean(self.actor_loss_fn(probs,actions,deltas))
        grads = tape_actor.gradient(loss, self.actor.trainable_variables)
        self.losses.append(loss)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        self.actor.save(self.model_name+"_actor",overwrite=True)


    def train_n_games(self,n):
        for episode in range(n):
            obs, info = self.env.reset()   
            game_reward = 0
            done , truncated = False , False
            step = 0
            while done == False and truncated == False:
                obs, reward, done, truncated, info = self.play_one_step(obs)
                game_reward+=reward
                step+=1
                if step%10 == 0 :
                    self.training_step()
            self.rewards.append(game_reward)
            self.all_mus.append(self.game_mus)
            self.all_sigmas.append(self.game_sigmas)
            self.game_mus = []
            self.game_sigmas = []
            print(f"Episode: {episode + 1}, Steps: {step + 1}, reward: {game_reward}, loss: {self.losses[-1]}")
            step = 0 
        self.all_sigmas = np.array(self.all_sigmas).flatten()
        self.all_mus = np.array(self.all_mus).flatten()
        plt.figure("Rewards")
        plt.plot(range(len(self.rewards)),self.rewards,label="Rewards per step",color = "lightgreen")
        plt.plot(range(len(self.losses)),self.losses,label = "Looses", color = "orange")
        plt.legend()
        plt.figure("mus et sigmas")
        plt.plot(range(len(self.all_sigmas)),self.all_sigmas,label="sigmas",color = "red")
        plt.plot(range(len(self.all_mus)),self.all_mus,label="mus",color = "green")
        plt.legend()
        plt.show()

    def make_video(self,video_name):
        video = cv2.VideoWriter(video_name+str(".avi"),cv2.VideoWriter_fourcc(*'MJPG'),10,(600,400))
        obs, info = self.env.reset()
        done, truncated == False, False
        while done == False or truncated == False:
            obs, reward, done, truncated, info = self.play_one_step(obs,deterministic = False)
            img = self.env.render()
            img = cv2.resize(img,(600,400))
            video.write(img)
        video.release()
        cv2.destroyAllWindows() 
        


if __name__=="__main__":
    model_name = "space_model3"
    env_name = "LunarLander-v3"
    n_layers = 1
    dqn = DQN(model_name,env_name,n_layers)
    dqn.train_n_games(50)   
    dqn.make_video("test_vid1")


