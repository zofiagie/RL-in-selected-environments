import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque


# Model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation="relu", input_shape=[4]),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(2)
])

batch_size = 64
gamma = 0.99
num_episodes = 500
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.mean_squared_error
replay_buffer = deque(maxlen=2000)

env = gym.make("CartPole-v0")

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = np.empty(max_size, dtype=object)
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.randint(self.size, size=batch_size)
        return self.buffer[indices]


def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis], verbose=0)[0]
        return Q_values.argmax()


for episode in range(num_episodes):
    obs, info = env.reset()
    step = 0
    while True:
      step += 1
      epsilon = max(1 - episode / 400, 0.01)
      action = epsilon_greedy_policy(obs, epsilon)
      obs_old = obs
      obs, reward, done, truncated, info = env.step(action)
      replay_buffer.append((obs_old, action, reward, obs, done, truncated))
      if done or truncated:
          break

    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    experiences = [np.array([experience[field_index] for experience in batch])
        for field_index in range(6)]
    states, actions, rewards, next_states, dones, truncateds = experiences

    next_Q_values = model.predict(next_states, verbose=0)
    max_next_Q_values = next_Q_values.max(axis=1)
    runs = 1.0 - (dones | truncateds)
    target_Q_values = rewards + runs * gamma * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, 2)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print(f"\rEpisode: {episode + 1}, Steps: {step}", end="")