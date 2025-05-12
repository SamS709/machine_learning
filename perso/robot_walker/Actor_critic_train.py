import numpy as np
import tensorflow as tf
import keras 
import gymnasium as gym
from collections import deque
import cv2
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

class DQN:

    def __init__(self,model_name,env_name,n_layers):
        self.env = gym.make(env_name,render_mode="rgb_array")
        self.n_outputs = self.env.action_space.shape[0]
        n_obs = self.env.reset(seed=42)[0].shape[0]
        self.replay_buffer = deque(maxlen=2000)
        self.batch_size = 32
        self.discount_factor = 0.98
        self.model_name = model_name
        self.init_agent(n_layers,n_obs)


    def init_agent(self,n_layers,n_obs):
        #Actor Initialization
        try:
            self.actor = keras.models.load_model(self.model_name+"_actor")
        except:
            input_layer = tf.keras.layers.Input([n_obs],name="input_layer")
            hidden = tf.keras.layers.Dense(128, activation="elu")(input_layer)
            for i in range(n_layers):
                hidden = tf.keras.layers.Dense(128, activation="elu")(hidden)
            mu = tf.keras.layers.Dense(self.n_outputs, activation="tanh")(hidden)
            sigma = tf.keras.layers.Dense(self.n_outputs, activation="softplus")(hidden)
            self.actor = tf.keras.Model(inputs=[input_layer],outputs=[mu,sigma])
            self.actor.save(self.model_name+"_actor")
        self.actor_optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)

        #Critic Initialization

        try:
            self.critic = keras.models.load_model(self.model_name+"_critic")
        except:
            self.critic = tf.keras.Sequential()
            self.critic.add(tf.keras.layers.Dense(128, activation="elu", input_shape=[n_obs]))
            for i in range(n_layers):
                self.critic.add(tf.keras.layers.Dense(128, activation="elu"))
            self.critic.add(tf.keras.layers.Dense(1))
            self.critic.save(self.model_name+"_critic")
        self.critic_optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)
        self.critic_loss_fn = tf.keras.losses.mean_squared_error
    
    

    def actor_loss_fn(self,norm_dists, actions, deltas):
        deltas = tf.cast(deltas,dtype="float32")
        proba = norm_dists[0].prob(actions[0])
        return [tf.math.multiply(-tf.math.log(norm_dists[i].prob(actions[i])),deltas[i]) for i in range(deltas.shape[0])]


    def policy(self,state,deterministic=False):
        pred = self.actor.predict(state[np.newaxis],verbose=0)
        mu = pred[0][0]
        sigma = pred[1][0]
        norm_dist = tfp.distributions.Normal(mu,sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
        action_tf_var = tf.clip_by_value(action_tf_var,-1,1)
        if deterministic:
            action_tf_var = mu
        return action_tf_var, norm_dist


    def play_one_step(self, state, deterministic = False):
        action,norm_dist = self.policy(state, deterministic)
        next_state, reward, done, truncated, info = self.env.step(action)
        self.replay_buffer.append((state, action, reward, next_state, done, truncated,norm_dist))
        return next_state, reward, done, truncated, info    
    
    def sample_experiences(self):
        indexes = np.random.randint(len(self.replay_buffer),size=self.batch_size)
        batch = [self.replay_buffer[index] for index in indexes]
        return [
                np.array([experience[field_index] for experience in batch])
                for field_index in range(7)
            ]    
    def training_step(self):

        experiences = self.sample_experiences()
        states, actions, rewards, next_states, dones, truncateds, norm_dists = experiences
        next_Q_values = self.critic.predict(next_states, verbose=0)
        runs = 1.0 - (dones | truncateds)  # episode is not done or truncated
        runs = runs.reshape([runs.size,1])
        rewards = rewards.reshape([runs.size,1])
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
            muandsigma = self.actor(states)
            mus = muandsigma[0]
            sigmas = muandsigma[1]
            norm_dists = tfp.distributions.Normal(mus,sigmas)
            actions_pred = tf.squeeze(norm_dists.sample(1), axis=0)
            actions_pred = tf.clip_by_value(actions_pred,-1,1)
            loss = tf.reduce_mean(self.actor_loss_fn(norm_dists,actions_pred,deltas))
            print(loss)
        grads = tape_actor.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        self.actor.save(self.model_name+"_actor",overwrite=True)


    def train_n_games(self,n):
        all_rewards = []
        for episode in range(n):
            obs, info = self.env.reset()   
            game_reward = 0
            for step in range(1000):
                obs, reward, done, truncated, info = self.play_one_step(obs)
                final_step=step
                game_reward+=reward
                if done or truncated:
                    break
            print(f"\rEpisode: {episode + 1}, Steps: {step + 1}, reward: {game_reward}",end="")
            all_rewards.append(game_reward)
            if episode >= 0 :
                self.training_step()
        plt.figure("Rewards")
        plt.plot(range(len(all_rewards)),all_rewards)
        plt.show()

    def make_video(self,video_name):
        video = cv2.VideoWriter(video_name+str(".avi"),cv2.VideoWriter_fourcc(*'MJPG'),10,(600,400))
        obs, info = self.env.reset()
        for step in range(1000):
            obs, reward, done, truncated, info = self.play_one_step(obs,deterministic = False)
            img = self.env.render()
            img = cv2.resize(img,(600,400))
            video.write(img)
            if done or truncated:
                break
        video.release()
        cv2.destroyAllWindows() 
        


if __name__=="__main__":
    model_name = "test_model3"
    env_name = "MountainCarContinuous-v0"
    n_layers = 2
    dqn = DQN(model_name,env_name,n_layers)
    dqn.train_n_games(30)   
    dqn.make_video("videocar8") 


