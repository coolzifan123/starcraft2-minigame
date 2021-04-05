import sys
import os
import shutil
import argparse
from functools import partial
from pre_processing import Preprocessor as pro
import numpy
import time
from pysc2.lib import actions

from a2c import RandomAgent,A2C
import torch
from net import CNN

from environment import SubprocVecEnv, make_sc2env, SingleEnv

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['main.py'])

parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')


parser.add_argument('--envs', type=int, default=1,
                    help='number of environments simulated in parallel')
parser.add_argument('--save_eposides', type=int, default=10,
                    help='store checkpoint after this many episodes')
parser.add_argument('--entropy_weight', type=float, default=1e-3,
                    help='weight of entropy loss')
parser.add_argument('--value_loss_weight', type=float, default=0.5,
                    help='weight of value function loss')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--save_dir', type=str, default='save',
                    help='root directory for checkpoint storage')

# 下面是有用的参数
parser.add_argument('--map', type=str, default='MoveToBeacon',
                    help='name of SC2 map')
parser.add_argument('--process_screen',action='store_true',
                    help='process screen and minimap')
parser.add_argument('--total', type=int, default=100,
                    help='episodes')
parser.add_argument('--load_model', type=str, default='save/MoveToBeacon_model.pkl',
                    help='load_model')
parser.add_argument('--determined',action='store_true',
                    help='actions selection：determine or sample')
args = parser.parse_args()

def main():
    model=args.load_model
    map_name=args.map
    envs_num=1
    max_windows=1
    env_args = dict(
        map_name=map_name,
        battle_net_map=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=32,
            feature_minimap=32,
            rgb_screen=None,
            rgb_minimap=None,
            action_space=None,
            use_feature_units=False,
            use_raw_units=False),
        step_mul=8,
        game_steps_per_episode=None,
        disable_fog=False,
        visualize=False
    )
    vis_env_args = env_args.copy()
    vis_env_args['visualize'] = True
    num_vis = min(envs_num, max_windows)
    env_fns = [partial(make_sc2env, **vis_env_args)] * num_vis
    num_no_vis = envs_num - num_vis
    if num_no_vis > 0:
      env_fns.extend([partial(make_sc2env, **env_args)] * num_no_vis)
    envs = SubprocVecEnv(env_fns)

    while True:
        agent=A2C(envs,args)
        agent.reset()
        agent.net.load_state_dict(torch.load(model))
        #try:

        t=0
        max_score=0
        episode=args.total
        while True:
            policy, value = agent.step(agent.last_obs)
            if not args.determined:
                actions = agent.select_actions(policy, agent.last_obs)
            else:
                actions = agent.determined_actions(policy, agent.last_obs)
            actions = agent.mask_unused_action(actions)
            size = agent.last_obs['screen'].shape[2:4]
            pysc2_action = agent.functioncall_action(actions, size)
            obs_raw = agent.envs.step(pysc2_action)
            agent.last_obs = agent.processor.preprocess_obs(obs_raw)
            for i in obs_raw:
                if i.last():
                    t+=1
                    score = i.observation['score_cumulative'][0]
                    agent.sum_score += score
                    agent.sum_episode += 1
                    print("episode %d: score = %f" % (agent.sum_episode, score))
                    if score>max_score:
                        max_score=score
            if t>=episode:
                print('max score=%d,average score=%.2f\n\n\n' % (max_score,agent.sum_score/(agent.sum_episode)))
                break

    #except :
        #print(agent.last_obs['available_actions'])

    envs.close()



if __name__=='__main__':
    main()