from __future__ import absolute_import, division, print_function
import torch as th
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from tkinter import Variable
import numpy as np
import math
import matplotlib.pyplot as plt

import os
import logging
import argparse

logging.getLogger().setLevel(logging.DEBUG)


from layers import ConvexQuadratic


class QMixer(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    # def __init__(
    #     self, in_dim, 
    #     hidden_layer_sizes=[32, 32, 32],
    #     rank=1, activation='celu', dropout=0.03,
    #     strong_convexity=1e-6
    # ):
    def __init__(self, args):
        super(QMixer, self).__init__()
        
        self.strong_convexity = 1e-6    #strong_convexity
        self.hidden_layer_sizes = [32, 32, 32]  #hidden_layer_sizes
        self.droput = 0.03  #dropout
        self.activation = 'celu'    #activation
        self.rank = 1   #rank
        self.in_dim = args.n_agents
        self.n_agents = args.n_agents
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(self.in_dim, out_features, rank=self.rank, bias=True),
                nn.Dropout(self.dropout)
            )
            for out_features in self.hidden_layer_sizes
        ])
        
        sizes = zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=True),
                nn.Dropout(self.dropout)
            )
            for (in_features, out_features) in sizes
        ])

        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=True)
        
        hypernet_embed = 64     # QMIX的hypernet参数的中间channel数是64         self.args.hypernet_embed
        self.hyper_w_1 = nn.Sequential(nn.Linear(64, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, 32 * self.hidden_layer_sizes[1]))
        self.hyper_w_2 = nn.Sequential(nn.Linear(64, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.hidden_layer_sizes[1] * self.hidden_layer_sizes[2]))
   
        self.hyper_w_final = nn.Sequential(nn.Linear(64, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.hidden_layer_sizes[2] * 1))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(64, 32 * 2000)
        self.hyper_b_2 = nn.Linear(64, 32 * 2000)
        self.hyper_b_final = nn.Linear(64, 2000)
                                        

    def forward(self, input, states):
        output = self.quadratic_layers[0](input)

        # output_for_hyper = output.reshape(-1, 32, 1)
        
        # for quadratic_layer in self.quadratic_layers[1:]:

        quad1 = self.quadratic_layers[1]
        a1 = quad1(input)

        # First Layer, weights positive


        w1 = self.hyper_w_1(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(32, 32)
        b1 = b1.view(-1, 32)
        hidden1 = F.elu(th.mm(output, w1) + b1) 
        quad1 = self.quadratic_layers[1]
        a1 = quad1(input)
        output = hidden1 + quad1(input)
        if self.activation == 'celu':
            output = th.celu(output)
        elif self.activation == 'softplus':
            output = F.softplus(output)
        else:
            raise Exception('Activation is not specified or unknown.')


        # Second Layer, weights positive
        w2 = self.hyper_w_2(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b2 = self.hyper_b_1(states)
        w2 = w2.view(32, 32)
        b2 = b2.view(-1, 32)
        hidden2 = F.elu(th.mm(output, w2) + b2) 
        quad2 = self.quadratic_layers[2]
        output = hidden2 + quad2(input)
        if self.activation == 'celu':
            output = th.celu(output)

        
        # Final, weights negative
        w_final = -self.hyper_w_final(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b_final = self.hyper_b_final(states)
        w_final = w_final.view(32, -1)
        b_final = b_final.view(-1, 1)
        hidden_final = th.mm(output, w_final) + b_final
        output = hidden_final + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)
        return output
   
    def push(self, input):
        output = autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=th.ones((input.size()[0], 1)).cuda().float()
        )[0]
        return output    
    
    def convexify(self):
        pass
        # for layer in self.convex_layers:
        #     for sublayer in layer:
        #         if (isinstance(sublayer, nn.Linear)):
        #             sublayer.weight.data.clamp_(0)
        # self.final_layer.weight.data.clamp_(0)

    def neg_convexify(self):
        pass
        # for layer in self.convex_layers:
        #     for sublayer in layer:
        #         if (isinstance(sublayer, nn.Linear)):
        #             sublayer.weight.data.clamp_(0)
        # self.final_layer.weight.data.clamp_(max = 0)

    def neg_convexify_1(self):
        pass
        # for layer in self.convex_layers:
        #     for sublayer in layer:
        #         if (isinstance(sublayer, nn.Linear)):
        #             sublayer.weight.data.clamp_(0)
        # self.final_layer.weight.data.clamp_(max = 0)
