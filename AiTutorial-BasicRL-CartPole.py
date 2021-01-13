'''
Ian MacMillan 
January 2021

See RL Zoo: https://github.com/CaltechExperimentalGravity/RLzoo
'''

#RLzoo imports
from rlzoo.common.env_wrappers import build_env
from rlzoo.common.utils import call_default_params
from rlzoo.algorithms import PG  # import the algorithm to use

AlgName = 'PG'
EnvName = 'CartPole-v1'
EnvType = 'classic_control'  # the name of env needs to match the type of env

env = build_env(EnvName, EnvType)

alg_params, learn_params = call_default_params(env, EnvType, AlgName)
alg = eval(AlgName+'(**alg_params)')
alg.learn(env=env, mode='train',train_episodes=1000, test_episodes=10, max_steps=400, save_interval=100, render=False)
alg.learn(env=env, mode='test',train_episodes=1000, test_episodes=10, max_steps=400, save_interval=100, render=False)







