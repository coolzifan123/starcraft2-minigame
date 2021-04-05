import torch
import torch.nn as nn
import torch.optim as optim
import os
from functools import partial
from pre_processing import Preprocessor, is_spatial_action, stack_ndarray_dicts
from pysc2.lib.actions import TYPES as ACTION_TYPES
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from torch.distributions.categorical import Categorical
import random
import numpy
import torch.optim as optim
import time
from pysc2.lib import actions
from script_agent import DefeatRoaches,CollectMineralShards
from pysc2.agents import base_agent
from net import CNN
import copy

from final_agent import BuildMarines,DefeatZerglingsAndBanelings,CollectMineralsAndGas



from pysc2.lib import features
# pysc2.agents.scripted_agent import CollectMineralShards
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

from environment import SubprocVecEnv, make_sc2env, SingleEnv


def flatten_first_dims(x):
    new_shape = [x.shape[0] * x.shape[1]] + list(x.shape[2:])
    return x.reshape(*new_shape)


def flatten_first_dims_dict(x):
    return {k: flatten_first_dims(v) for k, v in x.items()}


def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['main.py'])


def RuleBase2(net,map,process):

    map_name = 'CollectMineralShards'
    value_coef=0.25
    total_episodes = 20
    total_updates = -1
    sum_score = 0
    n_steps = 8
    learning_rate = 1e-4
    optimizer = optim.Adam(
        net.parameters(), learning_rate, weight_decay=0.01)
    env = make_sc2env(
        map_name=map_name,
        battle_net_map=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=32,
            feature_minimap=32,
            rgb_screen=None,
            rgb_minimap=None,
            action_space=None,
            use_feature_units=True,
            use_raw_units=False),
        step_mul=8,
        game_steps_per_episode=None,
        disable_fog=False,
        visualize=True
    )

    processor = Preprocessor(env.observation_spec()[0],map,process)
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    agent=CollectMineralShards()
    episodes = 0
    agent.reset()
    timesteps = env.reset()
    while True:
        fn_ids = []
        args_ids = []
        observations = []
        rewards=[]
        dones=[]
        for step in range(n_steps):
            a_0, a_1 = agent.step(timesteps[0])
            obs = processor.preprocess_obs(timesteps)
            observations.append(obs)
            actions = FunctionCall(a_0, a_1)
            fn_id = torch.LongTensor([a_0]).cuda()
            args_id = {}
            if a_0 == 7:
                for type in ACTION_TYPES:
                    if type.name == 'select_add':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            elif a_0 == 331:
                for type in ACTION_TYPES:
                    if type.name == 'queued':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    elif type.name == 'screen':

                        args_id[type] = torch.LongTensor([a_1[1][1] * 32 + a_1[1][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            elif a_0==2:
                for type in ACTION_TYPES:
                    if type.name == 'select_point_act':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    elif type.name == 'screen':

                        args_id[type] = torch.LongTensor([a_1[1][1] * 32 + a_1[1][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            elif a_0==0:
                for type in ACTION_TYPES:
                    args_id[type] = torch.LongTensor([-1]).cuda()
            action = (fn_id, args_id)
            fn_ids.append(fn_id)
            args_ids.append(args_id)
            timesteps = env.step([actions])
            rewards.append(torch.FloatTensor([timesteps[0].reward]).cuda())
            dones.append(torch.IntTensor([timesteps[0].last()]).cuda())

            if timesteps[0].last():
                i = timesteps[0]
                score = i.observation['score_cumulative'][0]
                sum_score += score
                episodes += 1
                if episodes%50==0:
                    torch.save(net.state_dict(), './save/episode' +str(episodes)
                               +str('.pkl'))
                print("episode %d: score = %f" % (episodes, score))
            #obs = processor.preprocess_obs(timesteps)
            #observations.append(obs)
        rewards = torch.cat(rewards)
        dones = torch.cat(dones)
        with torch.no_grad():
            obs = processor.preprocess_obs(timesteps)
            screen = torch.FloatTensor(obs['screen']).cuda()
            minimap = torch.FloatTensor(obs['minimap']).cuda()
            flat = torch.FloatTensor(obs['flat']).cuda()
            _,next_value = net(screen, minimap, flat)

        observations = flatten_first_dims_dict(
            stack_ndarray_dicts(observations))

        train_fn_ids = torch.cat(fn_ids)
        train_arg_ids = {}

        for k in args_ids[0].keys():
            temp = []
            temp = [d[k] for d in args_ids]

            train_arg_ids[k] = torch.cat(temp, dim=0)

        screen = torch.FloatTensor(observations['screen']).cuda()
        minimap = torch.FloatTensor(observations['minimap']).cuda()
        flat = torch.FloatTensor(observations['flat']).cuda()
        policy, value = net(screen, minimap, flat)

        returns=torch.zeros((rewards.shape[0]+1,),dtype=float)
        returns[-1] = next_value
        for i in reversed(range(rewards.shape[0])):
            next_rewards=0.999*returns[i+1]*(1-dones[i])
            returns[i]=rewards[i]+next_rewards
        returns=returns[:-1].cuda()

        fn_pi, args_pi = policy
        available_actions = torch.FloatTensor(observations['available_actions']).cuda()
        function_pi = available_actions * fn_pi
        function_pi /= torch.sum(function_pi, dim=1, keepdim=True)
        Loss = nn.CrossEntropyLoss(reduction='none')
        policy_loss = Loss(function_pi, train_fn_ids)

        for type in train_arg_ids.keys():
            id = train_arg_ids[type]
            pi = args_pi[type]
            arg_loss_list = []
            for i, p in zip(id, pi):
                if i == -1:
                    temp = torch.zeros((1)).cuda()
                else:
                    a = torch.LongTensor([i]).cuda()
                    b = torch.unsqueeze(p, dim=0).cuda()
                    temp = Loss(b, a)
                arg_loss_list.append(temp)

            arg_loss = torch.cat(arg_loss_list)
            policy_loss += arg_loss
        policy_loss = policy_loss.mean()
        value_loss = (returns - value).pow(2).mean()
        print(policy_loss,value_loss)
        loss=policy_loss+value_coef*value_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if episodes >= total_episodes:
            break
    torch.save(net.state_dict(), './save/episode2_1' +
               str('.pkl'))



def RuleBase4(net,map,process):
    map_name = 'DefeatRoaches'
    value_coef = 0.25
    total_episodes = 100
    total_updates = -1
    sum_score = 0
    n_steps = 8
    learning_rate = 1e-4
    optimizer = optim.Adam(
        net.parameters(), learning_rate, weight_decay=0.01)
    env = make_sc2env(
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
        visualize=True
    )

    processor = Preprocessor(env.observation_spec()[0],map,process)
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    agent = DefeatRoaches()
    episodes = 0
    agent.reset()
    timesteps = env.reset()
    while True:
        fn_ids = []
        args_ids = []
        observations = []
        rewards = []
        dones = []
        for step in range(n_steps):
            a_0, a_1 = agent.step(timesteps[0])
            obs = processor.preprocess_obs(timesteps)
            observations.append(obs)
            actions = FunctionCall(a_0, a_1)
            fn_id = torch.LongTensor([a_0]).cuda()
            args_id = {}
            if a_0 == 7:
                for type in ACTION_TYPES:
                    if type.name == 'select_add':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            elif a_0 == 12:
                for type in ACTION_TYPES:
                    if type.name == 'queued':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    elif type.name == 'screen':

                        args_id[type] = torch.LongTensor([a_1[1][1] * 32 + a_1[1][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            elif a_0 == 0:
                for type in ACTION_TYPES:
                    args_id[type] = torch.LongTensor([-1]).cuda()
            action = (fn_id, args_id)
            fn_ids.append(fn_id)
            args_ids.append(args_id)
            timesteps = env.step([actions])
            rewards.append(torch.FloatTensor([timesteps[0].reward]).cuda())
            dones.append(torch.IntTensor([timesteps[0].last()]).cuda())

            if timesteps[0].last():
                i = timesteps[0]
                score = i.observation['score_cumulative'][0]
                sum_score += score
                episodes += 1
                if episodes % 20 == 0:
                    torch.save(net.state_dict(), './save/episode3_' + str(episodes)
                               + str('.pkl'))
                print("episode %d: score = %f" % (episodes, score))
            # obs = processor.preprocess_obs(timesteps)
            # observations.append(obs)
        rewards = torch.cat(rewards)
        dones = torch.cat(dones)
        with torch.no_grad():
            obs = processor.preprocess_obs(timesteps)
            screen = torch.FloatTensor(obs['screen']).cuda()
            minimap = torch.FloatTensor(obs['minimap']).cuda()
            flat = torch.FloatTensor(obs['flat']).cuda()
            _, next_value = net(screen, minimap, flat)

        observations = flatten_first_dims_dict(
            stack_ndarray_dicts(observations))

        train_fn_ids = torch.cat(fn_ids)
        train_arg_ids = {}

        for k in args_ids[0].keys():
            temp = []
            temp = [d[k] for d in args_ids]

            train_arg_ids[k] = torch.cat(temp, dim=0)

        screen = torch.FloatTensor(observations['screen']).cuda()
        minimap = torch.FloatTensor(observations['minimap']).cuda()
        flat = torch.FloatTensor(observations['flat']).cuda()
        policy, value = net(screen, minimap, flat)

        returns = torch.zeros((rewards.shape[0] + 1,), dtype=float)
        returns[-1] = next_value
        for i in reversed(range(rewards.shape[0])):
            next_rewards = 0.999 * returns[i + 1] * (1 - dones[i])
            returns[i] = rewards[i] + next_rewards
        returns = returns[:-1].cuda()

        fn_pi, args_pi = policy
        available_actions = torch.FloatTensor(observations['available_actions']).cuda()
        function_pi = available_actions * fn_pi
        function_pi /= torch.sum(function_pi, dim=1, keepdim=True)
        Loss = nn.CrossEntropyLoss(reduction='none')
        policy_loss = Loss(function_pi, train_fn_ids)

        for type in train_arg_ids.keys():
            id = train_arg_ids[type]
            pi = args_pi[type]
            arg_loss_list = []
            for i, p in zip(id, pi):
                if i == -1:
                    temp = torch.zeros((1)).cuda()
                else:
                    a = torch.LongTensor([i]).cuda()
                    b = torch.unsqueeze(p, dim=0).cuda()
                    temp = Loss(b, a)
                arg_loss_list.append(temp)

            arg_loss = torch.cat(arg_loss_list)
            policy_loss += arg_loss
        policy_loss = policy_loss.mean()
        value_loss = (returns - value).pow(2).mean()
        print(policy_loss, value_loss)
        loss = policy_loss + value_coef * value_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if episodes >= total_episodes:
            break
    torch.save(net.state_dict(), './save/episode4_1' +
               str('.pkl'))

def RuleBase7(net,map,process):

    map_name = 'BuildMarines'
    value_coef = 1
    total_episodes = 5
    total_updates = -1
    sum_score = 0
    n_steps = 8
    learning_rate = 1e-5
    optimizer = optim.Adam(
        net.parameters(), learning_rate, weight_decay=0.01)
    env = make_sc2env(
        map_name=map_name,
        battle_net_map=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=32,
            feature_minimap=32,
            rgb_screen=None,
            rgb_minimap=None,
            action_space=None,
            use_feature_units=True,
            use_raw_units=False),
        step_mul=8,
        game_steps_per_episode=None,
        disable_fog=False,
        visualize=True
    )

    processor = Preprocessor(env.observation_spec()[0],map,process)
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    agent = BuildMarines()
    agent.setup(observation_spec[0],action_spec[0])
    episodes = 0
    agent.reset()
    timesteps = env.reset()
    while True:
        fn_ids = []
        args_ids = []
        observations = []
        rewards = []
        dones = []
        for step in range(n_steps):
            a_0, a_1 = agent.step(timesteps[0])
            obs = processor.preprocess_obs(timesteps)
            observations.append(obs)

            actions = FunctionCall(a_0, a_1)
            fn_id = torch.LongTensor([a_0]).cuda()
            args_id = {}
            if a_0 == 2:
                for type in ACTION_TYPES:
                    if type.name == 'select_point_act':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    elif type.name=='screen':
                        args_id[type] = torch.LongTensor([a_1[1][1]* 32 + a_1[1][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()

            elif a_0 == 42 or a_0==91 or a_0==264:
                for type in ACTION_TYPES:
                    if type.name == 'queued':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    elif type.name == 'screen':

                        args_id[type] = torch.LongTensor([a_1[1][1] * 32 + a_1[1][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            elif a_0 == 0:
                for type in ACTION_TYPES:
                    args_id[type] = torch.LongTensor([-1]).cuda()
            elif a_0 == 477 or a_0==490:
                for type in ACTION_TYPES:
                    if type.name == 'queued':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            action = (fn_id, args_id)
            fn_ids.append(fn_id)
            args_ids.append(args_id)
            timesteps = env.step([actions])
            rewards.append(torch.FloatTensor([timesteps[0].reward]).cuda())
            dones.append(torch.IntTensor([timesteps[0].last()]).cuda())

            if timesteps[0].observation['score_cumulative'][0]>0:
                i = timesteps[0]
                score = i.observation['score_cumulative'][0]
                sum_score += score

                if sum_score  == 3:
                    torch.save(net.state_dict(), './save/episode7_' + str(episodes)
                               + str('.pkl'))
                #print("episode %d: score = %f" % (episodes, score))
            if timesteps[0].last():
                episodes+=1
                sum_score=0
            # obs = processor.preprocess_obs(timesteps)
            # observations.append(obs)
        rewards = torch.cat(rewards)
        dones = torch.cat(dones)
        with torch.no_grad():
            obs = processor.preprocess_obs(timesteps)
            screen = torch.FloatTensor(obs['screen']).cuda()
            minimap = torch.FloatTensor(obs['minimap']).cuda()
            flat = torch.FloatTensor(obs['flat']).cuda()
            _, next_value = net(screen, minimap, flat)

        observations = flatten_first_dims_dict(
            stack_ndarray_dicts(observations))

        train_fn_ids = torch.cat(fn_ids)
        train_arg_ids = {}

        for k in args_ids[0].keys():
            temp = []
            temp = [d[k] for d in args_ids]

            train_arg_ids[k] = torch.cat(temp, dim=0)

        screen = torch.FloatTensor(observations['screen']).cuda()
        minimap = torch.FloatTensor(observations['minimap']).cuda()
        flat = torch.FloatTensor(observations['flat']).cuda()
        policy, value = net(screen, minimap, flat)

        returns = torch.zeros((rewards.shape[0] + 1,), dtype=float)
        returns[-1] = next_value
        for i in reversed(range(rewards.shape[0])):
            next_rewards = 0.999 * returns[i + 1] * (1 - dones[i])
            returns[i] = rewards[i] + next_rewards
        returns = returns[:-1].cuda()

        fn_pi, args_pi = policy
        available_actions = torch.FloatTensor(observations['available_actions']).cuda()
        function_pi = available_actions * fn_pi
        function_pi /= torch.sum(function_pi, dim=1, keepdim=True)
        Loss = nn.CrossEntropyLoss(reduction='none')
        policy_loss = Loss(function_pi, train_fn_ids)

        for type in train_arg_ids.keys():
            id = train_arg_ids[type]
            pi = args_pi[type]
            arg_loss_list = []
            for i, p in zip(id, pi):
                if i == -1:
                    temp = torch.zeros((1)).cuda()
                else:
                    a = torch.LongTensor([i]).cuda()
                    b = torch.unsqueeze(p, dim=0).cuda()
                    temp = Loss(b, a)
                arg_loss_list.append(temp)

            arg_loss = torch.cat(arg_loss_list)
            policy_loss += arg_loss
        policy_loss = policy_loss.mean()
        value_loss = (returns - value).pow(2).mean()
        print(policy_loss, value_loss)
        loss = policy_loss + value_coef * value_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if episodes >= total_episodes:
            break
    torch.save(net.state_dict(), './save/episode7_1' +
               str('.pkl'))



def RuleBase6(net,map,process):
    map_name = 'CollectMineralsAndGas'
    value_coef = 0.01
    total_episodes = 20
    total_updates = -1
    sum_score = 0
    n_steps = 8
    learning_rate = 1e-5
    optimizer = optim.Adam(
        net.parameters(), learning_rate, weight_decay=0.01)
    env = make_sc2env(
        map_name=map_name,
        battle_net_map=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=32,
            feature_minimap=32,
            rgb_screen=None,
            rgb_minimap=None,
            action_space=None,
            use_feature_units=True,
            use_raw_units=False),
        step_mul=8,
        game_steps_per_episode=None,
        disable_fog=False,
        visualize=True
    )

    processor = Preprocessor(env.observation_spec()[0],map,process)
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    agent = CollectMineralsAndGas()
    agent.setup(observation_spec[0], action_spec[0])
    episodes = 0
    agent.reset()
    timesteps = env.reset()
    while True:
        fn_ids = []
        args_ids = []
        observations = []
        rewards = []
        dones = []
        for step in range(n_steps):
            a_0, a_1 = agent.step(timesteps[0])
            obs = processor.preprocess_obs(timesteps)
            observations.append(obs)
            actions = FunctionCall(a_0, a_1)
            fn_id = torch.LongTensor([a_0]).cuda()
            args_id = {}
            if a_0 == 2:
                for type in ACTION_TYPES:
                    if type.name == 'select_point_act':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    elif type.name=='screen':
                        args_id[type] = torch.LongTensor([a_1[1][1]* 32 + a_1[1][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            elif a_0 == 91 or a_0==44 or a_0==264:
                for type in ACTION_TYPES:
                    if type.name == 'queued':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    elif type.name == 'screen':

                        args_id[type] = torch.LongTensor([a_1[1][1] * 32 + a_1[1][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            elif a_0==490:
                for type in ACTION_TYPES:
                    if type.name == 'queued':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            elif a_0 == 0:
                for type in ACTION_TYPES:
                    args_id[type] = torch.LongTensor([-1]).cuda()
            action = (fn_id, args_id)
            fn_ids.append(fn_id)
            args_ids.append(args_id)
            timesteps = env.step([actions])
            rewards.append(torch.FloatTensor([timesteps[0].reward]).cuda())
            dones.append(torch.IntTensor([timesteps[0].last()]).cuda())

            if timesteps[0].last():
                i = timesteps[0]
                score = i.observation['score_cumulative'][0]
                sum_score += score
                episodes += 1
                if episodes % 1 == 0:
                    torch.save(net.state_dict(), './save/game6_' + str(episodes)
                               + str('.pkl'))
                print("episode %d: score = %f" % (episodes, score))
            # obs = processor.preprocess_obs(timesteps)
            # observations.append(obs)
        rewards = torch.cat(rewards)
        dones = torch.cat(dones)
        with torch.no_grad():
            obs = processor.preprocess_obs(timesteps)
            screen = torch.FloatTensor(obs['screen']).cuda()
            minimap = torch.FloatTensor(obs['minimap']).cuda()
            flat = torch.FloatTensor(obs['flat']).cuda()
            _, next_value = net(screen, minimap, flat)

        observations = flatten_first_dims_dict(
            stack_ndarray_dicts(observations))

        train_fn_ids = torch.cat(fn_ids)
        train_arg_ids = {}

        for k in args_ids[0].keys():
            temp = []
            temp = [d[k] for d in args_ids]

            train_arg_ids[k] = torch.cat(temp, dim=0)

        screen = torch.FloatTensor(observations['screen']).cuda()
        minimap = torch.FloatTensor(observations['minimap']).cuda()
        flat = torch.FloatTensor(observations['flat']).cuda()
        policy, value = net(screen, minimap, flat)

        returns = torch.zeros((rewards.shape[0] + 1,), dtype=float)
        returns[-1] = next_value
        for i in reversed(range(rewards.shape[0])):
            next_rewards = 0.999 * returns[i + 1] * (1 - dones[i])
            returns[i] = rewards[i] + next_rewards
        returns = returns[:-1].cuda()

        fn_pi, args_pi = policy
        available_actions = torch.FloatTensor(observations['available_actions']).cuda()
        function_pi = available_actions * fn_pi
        function_pi /= torch.sum(function_pi, dim=1, keepdim=True)
        Loss = nn.CrossEntropyLoss(reduction='none')
        function_pi = torch.clamp(function_pi, 1e-4, 1-(1e-4))
        policy_loss = Loss(function_pi, train_fn_ids)

        for type in train_arg_ids.keys():
            id = train_arg_ids[type]
            pi = args_pi[type]
            arg_loss_list = []
            for i, p in zip(id, pi):
                if i == -1:
                    temp = torch.zeros((1)).cuda()
                else:
                    a = torch.LongTensor([i]).cuda()
                    b = torch.unsqueeze(p, dim=0).cuda()
                    b = torch.clamp(b, 1e-4, 1-(1e-4))
                    temp = Loss(b, a)
                arg_loss_list.append(temp)

            arg_loss = torch.cat(arg_loss_list)
            policy_loss += arg_loss
        policy_loss = policy_loss.mean()
        value_loss = (returns - value).pow(2).mean()
        print(policy_loss, value_loss)
        loss = policy_loss + value_coef * value_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if episodes >= total_episodes:
            break
    torch.save(net.state_dict(), './save/game6_final' +
               str('.pkl'))


