import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.lib.actions import TYPES
import numpy as np

from pre_processing import is_spatial_action, NUM_FUNCTIONS, FLAT_FEATURES
A=348
B=1985
class MyLinear(nn.Module):
    def __init__(self,out_channel):
        super(MyLinear,self).__init__()
        self.linear=nn.Linear(256,out_channel)

    def forward(self,x):
        x=self.linear(x)
        return x

class CNN(nn.Module):
    def __init__(self,A=27,B=11):
        super(CNN,self).__init__()
        self.conv1_screen=nn.Conv2d(A,16,kernel_size=5,stride=1,padding=2)
        self.conv1_minimap=nn.Conv2d(B,16,kernel_size=5,stride=1,padding=2)
        self.conv2=nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
        self.linear_1=nn.Linear(75*32*32,256)
        self.linear_2=nn.Linear(256,1)
        self.linear_3=nn.Linear(256,NUM_FUNCTIONS)
        self.conv3=nn.Conv2d(75,1,kernel_size=1,stride=1)
        self.mylinear=dict()
        for k in actions.TYPES:
            if not is_spatial_action[k]:
                self.mylinear[k]=MyLinear(k.sizes[0]).cuda()
    def forward(self,screen,minimap,flat):
        screen=self.relu(self.conv1_screen(screen))
        screen=self.relu(self.conv2(screen)).cuda()
        minimap=self.relu(self.conv1_minimap(minimap))
        minimap=self.relu(self.conv2(minimap))
        flat=torch.log(1.+flat)
        flat=flat.unsqueeze(2).unsqueeze(3)
        flat=flat.repeat(1,1,screen.shape[2],screen.shape[3])
        state=torch.cat((screen,minimap,flat),dim=1)
        flatten=state.contiguous().view(-1,75*32*32)   # 要不要先 tanspose
        flatten=self.relu(self.linear_1(flatten))
        value=self.linear_2(flatten)
        value=torch.reshape(value,(-1,))
        act_prob=self.linear_3(flatten)
        act_prob=F.softmax(act_prob,dim=1)
        args_out=dict()
        for i in actions.TYPES:
            if is_spatial_action[i]:
                arg_out=self.conv3(state)
                arg_out=arg_out.flatten(1,-1)
                arg_out=F.softmax(arg_out,dim=1)
            else:
                arg_out=self.mylinear[i](flatten)
                arg_out=F.softmax(arg_out,dim=1)
            args_out[i]=arg_out
        policy=(act_prob,args_out)

        return policy,value

#84*84 screen 和 64*64 minimap版本
class xCNN(nn.Module):
    def __init__(self):
        super(xCNN,self).__init__()
        self.conv1_screen=nn.Conv2d(27,16,kernel_size=11,stride=1)
        self.conv1_screen1=nn.Conv2d(16,16,kernel_size=11,stride=1)
        self.conv1_minimap=nn.Conv2d(11,16,kernel_size=5,stride=1,padding=2)
        self.conv2=nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
        self.linear_1=nn.Linear(75*64*64,256)
        self.linear_2=nn.Linear(256,1)
        self.linear_3=nn.Linear(256,NUM_FUNCTIONS)
        self.conv3=nn.Conv2d(75,1,kernel_size=1,stride=1)
        self.mylinear=dict()
        for k in actions.TYPES:
            if not is_spatial_action[k]:
                self.mylinear[k]=MyLinear(k.sizes[0]).cuda()
    def forward(self,screen,minimap,flat):
        screen=self.relu(self.conv1_screen(screen))
        screen=self.relu(self.conv1_screen1(screen))
        screen = self.relu(self.conv2(screen))
        minimap=self.relu(self.conv1_minimap(minimap))
        minimap=self.relu(self.conv2(minimap))
        flat=torch.log(1.+flat)
        flat=flat.unsqueeze(2).unsqueeze(3)
        flat=flat.repeat(1,1,screen.shape[2],screen.shape[3])
        state=torch.cat((screen,minimap,flat),dim=1)
        flatten=state.contiguous().view(-1,75*64*64)   # 要不要先 tanspose
        flatten=self.relu(self.linear_1(flatten))
        value=self.linear_2(flatten)
        value=torch.reshape(value,(-1,))
        act_prob=self.linear_3(flatten)
        act_prob=F.softmax(act_prob,dim=1)
        args_out=dict()
        for i in actions.TYPES:
            if is_spatial_action[i]:
                arg_out=self.conv3(state)
                arg_out=arg_out.flatten(1,-1)
                arg_out=F.softmax(arg_out,dim=1)
            else:
                arg_out=self.mylinear[i](flatten)
                arg_out=F.softmax(arg_out,dim=1)
            args_out[i]=arg_out
        policy=(act_prob,args_out)

        return policy,value
