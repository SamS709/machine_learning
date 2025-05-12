
import tensorflow as tf
import tensorflow_probability as tfp
input_layer = tf.keras.layers.Input([10],name="input_layer")
hidden = tf.keras.layers.Dense(32, activation="elu")(input_layer)
for i in range(3):
    hidden = tf.keras.layers.Dense(32, activation="elu")(hidden)
mu = tf.keras.layers.Dense(4, activation="tanh")(hidden)
sigma = tf.keras.layers.Dense(4, activation="softplus")(hidden)
actor = tf.keras.Model(inputs=[input_layer],outputs=[mu,sigma])
a = tf.constant([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
pred = actor.predict(a)
mu = pred[0]
sigma = pred[1]
print(mu,sigma)
norm_dist = tfp.distributions.Normal(mu,sigma)
action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
action_tf_var = tf.clip_by_value(action_tf_var, 
                                    -1, 
                                    1)
print(action_tf_var)
"""norm1 = tfp.distributions.Normal(0,1)
norm2 = tfp.distributions.Normal(0,1)

norm_dist = [norm1,norm2]

norm_dist = tfp.distributions.Blockwise(norm_dist)
print(norm_dist)
print(norm_dist.prob([0,0]))"""
"""inputlayer = tf.keras.layers.Input([6])
outputlayer = tf.keras.layers.Dense(1,activation="softplus")(inputlayer)
model = tf.keras.Model(inputs=[inputlayer],outputs=[outputlayer])
pred=model.predict([[1,2,3,4,5,6],[0,1,2,3,4,5]])
print(pred)"""
a = tf.Variable([[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]])
b = tf.Variable([[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]])
L = [tf.math.multiply(-tf.math.log(a[i]),b[i]) for i in range(5)]
print(L)
#-tf.log(norm_dist.prob(action))*delta





