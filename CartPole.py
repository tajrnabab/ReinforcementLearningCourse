from operator import mod
import gym
from requests import patch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import os
from stable_baselines3.common.evaluation import evaluate_policy


env = gym.make('CartPole-v0')
env = DummyVecEnv([lambda: env])

path = os.path.join('Training', 'Logs')

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=path)

model.learn(total_timesteps=20000)

# value = evaluate_policy(model, env, n_eval_episodes=10, render=True)
# print(value)

for episode in range(0,5):
    obs = env.reset()
    done=False
    score=0

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score+=reward
    print("Episode: {}, Score: {}".format(episode, score))