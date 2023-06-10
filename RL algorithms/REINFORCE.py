import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

gamma = 0.99
num_episodes = 250
optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_function = keras.losses.binary_crossentropy

env = gym.make("CartPole-v0")


def discount_and_normalize_rewards(rewards):
    discounted_rewards = []
    for r in rewards:
      discounted = np.array(r)
      for step in range(len(r) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * gamma
      discounted_rewards.append(discounted)
    flat_rewards = np.concatenate(discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(dr - reward_mean) / reward_std
            for dr in discounted_rewards]


for episode in range(num_episodes):
  current_rewards, current_grads = [], []
  obs = env.reset()
  while True:
    with tf.GradientTape() as tape:
      left_proba = model(obs[np.newaxis])
      action = (tf.random.uniform([1, 1]) > left_proba)
      y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
      loss = tf.reduce_mean(loss_function(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, truncated = env.step(int(action))
    current_rewards.append(reward)
    current_grads.append(grads)
    if done or truncated:
        break

  all_rewards, all_grads = [current_rewards], [current_grads]

  print(f"\rEpisode: {episode + 1}, Steps: {sum(map(sum, all_rewards))}", end="")

  all_final_rewards = discount_and_normalize_rewards(all_rewards)
  all_mean_grads = []
  for var_index in range(len(model.trainable_variables)):
      mean_grads = tf.reduce_mean(
          [final_reward * all_grads[episode_index][step][var_index]
          for episode_index, final_rewards in enumerate(all_final_rewards)
              for step, final_reward in enumerate(final_rewards)], axis=0)
      all_mean_grads.append(mean_grads)

  optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))