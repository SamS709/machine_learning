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
        self.init_containers()
        self.replay_buffer = deque(maxlen=500)
        self.batch_size = 32
        self.discount_factor = 0.99
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
            mu = tf.keras.layers.Dense(self.n_outputs, activation="tanh",kernel_initializer="he_normal")(hidden)
            sigma = tf.keras.layers.Dense(self.n_outputs, activation="softplus")(hidden)
            self.actor = tf.keras.Model(inputs=[input_layer],outputs=[mu,sigma])
            self.actor.save(self.model_name+"_actor")
            print(f'[INFO] : new Model : "{self.model_name}" created')

        self.actor_optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-6)

        #Critic Initialization

        try:
            self.critic = keras.models.load_model(self.model_name+"_critic")
        except:
            self.critic = tf.keras.Sequential()
            self.critic.add(tf.keras.layers.Dense(2048, activation="elu", kernel_initializer="he_normal",use_bias=True, input_shape=[n_obs]))
            for i in range(n_layers):
                self.critic.add(tf.keras.layers.Dense(1536, activation="elu", kernel_initializer="he_normal",use_bias=True))
            self.critic.add(tf.keras.layers.Dense(1))
            self.critic.save(self.model_name+"_critic")
        self.critic_optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-6)
        self.critic_loss_fn = tf.keras.losses.mean_squared_error
    
    

    def actor_loss_fn(self,mus,sigmas, actions, deltas):
        
        pdf_value = tf.exp(-0.5 *((actions - mus) / (sigmas))**2) * 1 / (sigmas * tf.sqrt(2 * np.pi))
        dist = tfp.distributions.Normal(mus,sigmas)
        # Convert pdf value to log probability
        log_probability = dist.log_prob(actions)
        
        # Compute weighted loss
        loss_actor = - deltas * log_probability #- 0.1*tf.math.log(sigmas*tf.sqrt(2*np.pi*np.e))

        return loss_actor

    def policy(self,state,deterministic=False):
        pred = self.actor.predict(state[np.newaxis],verbose=0)
        mu = pred[0][0]
        sigma = pred[1][0]
        norm_dist = tfp.distributions.Normal(mu,sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
        action_tf_var = tf.clip_by_value(action_tf_var,-2,2)
        if deterministic:
            action_tf_var = mu
        self.game_mus.append(float(mu))
        self.game_sigmas.append(float(sigma))
        return action_tf_var


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
            print(loss)
        grads = tape_critic.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        self.critic.save(self.model_name+"_critic",overwrite=True)
        Q_values = self.critic.predict(states, verbose=0)
        deltas = target_Q_values-Q_values
        with tf.GradientTape() as tape_actor:
            muandsigma = self.actor(states)
            mus = muandsigma[0]
            sigmas = muandsigma[1] 
            loss = tf.math.reduce_mean(self.actor_loss_fn(mus,sigmas,actions,deltas))
        grads = tape_actor.gradient(loss, self.actor.trainable_variables)
        self.losses.append(loss)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        self.actor.save(self.model_name+"_actor",overwrite=True)


    def train_n_games(self,n):
        for episode in range(n):
            obs, info = self.env.reset()   
            game_reward = 0
            for step in range(1000):
                obs, reward, done, truncated, info = self.play_one_step(obs)
                game_reward+=reward
                if step%32 == 0 :
                    self.training_step()
                if done or truncated:
                    break
            self.rewards.append(game_reward)
            self.all_mus.append(self.game_mus)
            self.all_sigmas.append(self.game_sigmas)
            self.game_mus = []
            self.game_sigmas = []
            
            print(f"\rEpisode: {episode + 1}, Steps: {step + 1}, reward: {game_reward}, loss: {self.losses[-1]}",end="")
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
        for step in range(200):
            obs, reward, done, truncated, info = self.play_one_step(obs,deterministic = False)
            img = self.env.render()
            img = cv2.resize(img,(600,400))
            video.write(img)
            if done or truncated:
                break
        video.release()
        cv2.destroyAllWindows() 
        


if __name__=="__main__":
    model_name = "test_pendulum6"
    env_name = "Pendulum-v1"
    n_layers = 1
    dqn = DQN(model_name,env_name,n_layers)
    """a=dqn.critic(tf.constant([-9.98091042e-01,-6.17602132e-02,1.44349867e-02])[np.newaxis])
    b=dqn.critic(tf.constant([-9.79602814e-01,-2.00943738e-01,-7.40392148e-01])[np.newaxis])
    c = dqn.critic(tf.constant([1,1,1])[np.newaxis])
    print(a,b,c)"""

    dqn.train_n_games(50)   
    dqn.make_video("test_vid1")


