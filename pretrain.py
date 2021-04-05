import torch
from torch.distributions.categorical import Categorical
from rule_base import RuleBase2,RuleBase4,RuleBase7,RuleBase6
from net import CNN,xCNN
import argparse
parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')
mapdict=dict()

mapdict['CollectMineralShards']=RuleBase2
mapdict['DefeatRoaches']=RuleBase4
mapdict['CollectMineralsAndGas']=RuleBase6
mapdict['BuildMarines']=RuleBase7
parser.add_argument('--map', type=str, default='CollectMineralShards',
                    help='name of SC2 map')


parser.add_argument('--process_screen',action='store_true',
                    help='process screen and minimap')

args = parser.parse_args()
if args.process_screen:
        net = CNN(348,1985).cuda()
else:net=CNN().cuda()
mapdict[args.map](net,args.map,args.process_screen)
