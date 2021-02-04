'''
Ian MacMillan 
January 2021

See RL Zoo: https://github.com/CaltechExperimentalGravity/RLzoo
'''

#RLzoo imports
import os
from rlzoo.common.env_wrappers import build_env
from rlzoo.common.utils import call_default_params
from rlzoo.algorithms import *  # import the algorithm to use

AlgName = 'SAC'
EnvName = 'BipedalWalker-v2'
#EnvName = 'bipedal_walker'
EnvType = 'box2d'

env = build_env(EnvName, EnvType)
alg_params, learn_params = call_default_params(env, EnvType, AlgName)
alg = eval(AlgName+'(**alg_params)')
alg.learn(env=env, mode='train', train_episodes=20000, test_episodes=100, max_steps=4000, save_interval=25, render=False)
imp='n'
os.system( "say Your model is done training" ) #this is macOS specific
while imp=='n':
   imp=input("Are you ready to test (y/n)")
alg.learn(env=env, mode='test', train_episodes=20000, test_episodes=100, max_steps=4000, save_interval=25, render=True)
