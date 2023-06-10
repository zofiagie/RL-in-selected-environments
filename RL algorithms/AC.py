import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Model
inputs = keras.layers.Input(shape=[4])
common = keras.layers.Dense(64, activation="relu")(inputs)
action = keras.layers.Dense(2, activation="softmax")(common)
critic = keras.layers.Dense(1)(common)
model = keras.Model(inputs=inputs, outputs=[action, critic])

gamma = 0.99
num_episodes = 250
optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_function = keras.losses.MeanSquaredError()

action_history, critic_history, rewards_history = [], [], []

env = gym.make("CartPole-v0")

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        while True:
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            action_probs, critic_value = model(state)
            critic_history.append(critic_value[0, 0])

            action = np.random.choice(2, p=np.squeeze(action_probs))
            action_history.append(tf.math.log(action_probs[0, action]))

            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        rewards = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            rewards.insert(0, discounted_sum)

        # Normalize
        rewards = np.array(rewards)
        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        rewards = rewards.tolist()

        # Calculating loss values
        history = zip(action_history, critic_history, rewards)
        actor_losses, critic_losses = [], []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)
            critic_losses.append(
                loss_function(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        action_history, critic_history, rewards_history = [], [], []

    print(f"\rEpisode: {episode + 1}, Steps: {episode_reward}", end="")