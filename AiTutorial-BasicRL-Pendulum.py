'''
Ian MacMillan 
January 2021

See RL Zoo: https://github.com/CaltechExperimentalGravity/RLzoo
'''

#RLzoo imports
from rlzoo.common.env_wrappers import build_env
from rlzoo.common.utils import call_default_params
from rlzoo.algorithms import AC  # import the algorithm to use


AlgName = 'AC'
EnvName = 'Pendulum-v0'
EnvType = 'classic_control'


env = build_env(EnvName, EnvType)
alg_params, learn_params = call_default_params(env, EnvType, AlgName)
alg = eval(AlgName+'(**alg_params)')
#alg.learn(env=env, mode='train', render=False, **learn_params)
#alg.learn(env=env, mode='test', render=True, **learn_params)

alg.learn(env=env, mode='train',train_episodes=500, test_episodes=5, max_steps=4000, save_interval=100, render=False)
alg.learn(env=env, mode='test',train_episodes=500, test_episodes=5, max_steps=4000, save_interval=100, render=True)

'''
1) ella's report
2) add code to git in non lin
3) stay biped for a few days
4) make more complex pendlium (like our isolation sys. add sismic noise) find transferfunction from top to bottom (see if code can do better)
'''
