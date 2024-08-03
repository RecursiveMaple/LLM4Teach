#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   student_net.py
@Time    :   2023/07/14 16:34:11
@Author  :   Zhou Zihao 
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import numpy as np


class NNBase(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        # Define embedding
        self.ln0 = nn.LayerNorm(obs_space)
        self.fc1 = nn.Linear(obs_space, obs_space)
        self.silu1 = nn.SiLU()
        self.ln1 = nn.LayerNorm(obs_space)
        self.fc2 = nn.Linear(obs_space, obs_space)
        self.silu2 = nn.SiLU()
        self.ln2 = nn.LayerNorm(2*obs_space)
        self.fc3 = nn.Linear(2*obs_space, 2*obs_space)
        self.silu3 = nn.SiLU()
        self.ln3 = nn.LayerNorm(4*obs_space)
        self.fc4 = nn.Linear(4*obs_space, 4*obs_space)
        self.silu4 = nn.SiLU()
        self.ln4 = nn.LayerNorm(4*obs_space)
        
        embedding_size = 4 * obs_space
        return embedding_size, action_space
    
    def embed(self, x):
        x = self.ln0(x)
        x = self.ln1(self.silu1(self.fc1(x)) + x)
        x = self.ln2(torch.cat((self.silu2(self.fc2(x)), x), dim=-1))
        x = self.ln3(torch.cat((self.silu3(self.fc3(x)), x), dim=-1))
        x = self.ln4(self.silu4(self.fc4(x)))
        return x

    def forward(self, obs, masks=None, states=None):
        raise NotImplementedError

        
class MLPBase(NNBase):
    def __init__(self, obs_space, action_space):
        embedding_size, action_space = super().__init__(obs_space, action_space)
        
        # self.fc = nn.Sequential(
        #     nn.Linear(embedding_size, 64),
        #     nn.Tanh()
        # )
        
        # # Define actor's model
        # self.actor = nn.Linear(64, action_space)
        # # Define critic's model
        # self.critic = nn.Linear(64, 1)
        
        # # Define actor's model
        # self.actor = nn.Sequential(
        #     nn.Linear(64, 16),
        #     nn.Tanh(),
        #     nn.Linear(16, action_space)
        # )
        # # Define critic's model
        # self.critic = nn.Sequential(
        #     nn.Linear(64, 16),
        #     nn.Tanh(),
        #     nn.Linear(16, 1)
        # )
        
        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )
        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def init_states(self, device=None, num_trajs=1):
        return None

    def forward(self, obs, masks=None, states=None):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        assert obs.dim() == 2, "observation dimension expected to be 2, but got {}.".format(obs.dim())
        
        # feature extractor
        embedding = self.embed(obs)

        # actor-critic
        value = self.critic(embedding).squeeze(1)
        action_logits = self.actor(embedding)
        dist = Categorical(logits=action_logits)

        return dist, value, embedding
        
# TODO: debug LSTM model
class LSTMBase(NNBase):
    def __init__(self, obs_space, action_space):
        embedding_size, action_space = super().__init__(obs_space, action_space)
    
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.ReLU()
        )
        self.core = nn.LSTM(256, 256, 2)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )
        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def init_states(self, device, num_trajs=1):
        return (torch.zeros(self.core.num_layers, num_trajs, self.core.hidden_size).to(device),
                torch.zeros(self.core.num_layers, num_trajs, self.core.hidden_size).to(device))

    def forward(self, obs, masks, states):
        input_dim = obs.dim()
        if input_dim == 2:
            unroll_length = obs.size(0)
            num_trajs = 1
        elif input_dim == 3:
            unroll_length, num_trajs, _ = obs.size()
            obs = torch.flatten(obs, 0, 1) # [unroll_length * num_trajs, width, height, channels]
        else:
            assert False, "observation dimension expected to be 4 or 5, but got {}.".format(input_dim)

        # feature extractor
        x = self.embed(x)
        x = self.fc(x)
        
        # LSTM
        core_input = x.view(unroll_length, num_trajs, -1) # [unroll_length, num_trajs, -1] 
        masks = masks.view(unroll_length, 1, num_trajs, 1) # [unroll_length, 1, num_trajs, 1]
        core_output_list = []
        for inp, mask in zip(core_input.unbind(), masks.unbind()):
            states = tuple(mask * s for s in states)
            output, states = self.core(inp.unsqueeze(0), states)
            core_output_list.append(output)
        core_output = torch.cat(core_output_list) # [unroll_length, num_trajs, -1]
        core_output = core_output.view(unroll_length * num_trajs, -1) # [unroll_length * num_trajs, -1]

        # actor-critic
        action_logits = self.actor(core_output)
        dist = Categorical(logits=action_logits)
        value = self.critic(core_output).squeeze(1)

        return dist, value, states
