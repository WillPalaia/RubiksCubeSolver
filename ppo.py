from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

# Parallel environments
vec_env = make_vec_env("Rubiks-v1", n_envs=4) # make env

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole") # save file

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole") # load file

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human") # replace this with printcube
