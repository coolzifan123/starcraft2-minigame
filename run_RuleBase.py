import argparse
from pysc2.env import sc2_env
from environment import SubprocVecEnv, make_sc2env, SingleEnv
from script_agent import DefeatRoaches,CollectMineralShards
from final_agent import BuildMarines,DefeatZerglingsAndBanelings,CollectMineralsAndGas
from pysc2.lib.actions import TYPES as ACTION_TYPES
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['main.py'])

parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')
parser.add_argument('--map', type=str, default='CollectMineralShards',
                    help='name of SC2 map')
args = parser.parse_args()
def main():
    map_dict = dict()

    map_dict['CollectMineralShards'] = CollectMineralShards
    map_dict['DefeatRoaches'] = DefeatRoaches
    map_dict['DefeatZerglingsAndBanelings'] = DefeatZerglingsAndBanelings
    map_dict['CollectMineralsAndGas'] = CollectMineralsAndGas
    map_dict['BuildMarines'] = BuildMarines
    agent=map_dict[args.map]()
    env = make_sc2env(
        map_name=args.map,
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
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    if args.map!='CollectMineralShards' or args.map!='DefeatRoaches':
        agent.setup(observation_spec[0], action_spec[0])
    agent.reset()

    timesteps = env.reset()
    episodes=0
    sum_score=0
    while True:

        a_0, a_1 = agent.step(timesteps[0])

        actions = FunctionCall(a_0, a_1)
        timesteps = env.step([actions])
        if timesteps[0].last():
            i = timesteps[0]
            score = i.observation['score_cumulative'][0]
            sum_score += score
            episodes += 1

            print("episode %d: score = %f" % (episodes, score))

if __name__=='__main__':
    main()