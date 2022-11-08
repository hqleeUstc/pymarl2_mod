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


from .layers import ConvexQuadratic


class QMixer_WGAN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    # def __init__(
    #     self, in_dim, 
    #     hidden_layer_sizes=[32, 32, 32],
    #     rank=1, activation='celu', dropout=0.03,
    #     strong_convexity=1e-6
    # ):
    def __init__(self, args):
        super(QMixer_WGAN, self).__init__()
        
        self.strong_convexity = 1e-6    #strong_convexity
        self.hidden_layer_sizes = [32, 32, 32]  #hidden_layer_sizes
        self.dropout = 0.03  #dropout
        self.activation = 'celu'    #activation
        self.rank = 1   #rank
        self.in_dim = args.n_agents
        self.n_agents = args.n_agents

        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(self.in_dim, out_features, rank=self.rank, bias=True),
                nn.Dropout(self.dropout)
            )
            for out_features in self.hidden_layer_sizes
        ])
        
        sizes = zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])
        # self.convex_layers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(in_features, out_features, bias=True),
        #         nn.Dropout(self.dropout)
        #     )
        #     for (in_features, out_features) in sizes
        # ])

        # self.final_layer = nn.Linear(self.hidden_layer_sizes[-1], 1, bias=True)
        
        hypernet_embed = args.hypernet_embed     # QMIX的hypernet参数的中间channel数是64         self.args.hypernet_embed
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.hidden_layer_sizes[0]))
        self.hyper_w_2 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.embed_dim))
   
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_2 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_final = nn.Linear(self.state_dim, 1)

        # # V(s) instead of a bias for the last layers
        # self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
        #                        nn.ReLU(inplace=True),
        #                        nn.Linear(self.embed_dim, 1))
                                        

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)

        output = self.quadratic_layers[0](agent_qs)

        # First Layer, weights positive
        w1 = self.hyper_w_1(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.hidden_layer_sizes[0], self.embed_dim)    # 32 is the channel of the quadratic_layer
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden1 = F.elu(th.bmm(output, w1) + b1)
        # hidden1 = th.bmm(output, w1) + b1
        quad1 = self.quadratic_layers[1](agent_qs)
        output = hidden1 + quad1
        if self.activation == 'celu':
            output = th.celu(output)
        elif self.activation == 'softplus':
            output = F.softplus(output)
        else:
            raise Exception('Activation is not specified or unknown.')

        # Second Layer, weights positive
        w2 = self.hyper_w_2(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b2 = self.hyper_b_2(states)
        w2 = w2.view(-1, self.embed_dim, self.embed_dim)
        b2 = b2.view(-1, 1, self.embed_dim)
        hidden2 = F.elu(th.bmm(output, w2) + b2) 
        # hidden2 = th.bmm(output, w2) + b2
        quad2 = self.quadratic_layers[2](agent_qs)
        output = hidden2 + quad2
        if self.activation == 'celu':
            output = th.celu(output)

        
        # Final, weights negative
        w_final = -self.hyper_w_final(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b_final = self.hyper_b_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        b_final = b_final.view(-1, 1, 1)
        hidden_final = th.bmm(output, w_final) + b_final
        output = hidden_final + .5 * self.strong_convexity * (agent_qs ** 2).sum(dim=2).reshape(-1, 1, 1)
        q_tot = output.view(bs, -1, 1)
        return q_tot
   
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

class QMixer_WGAN_v2(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    # def __init__(
    #     self, in_dim, 
    #     hidden_layer_sizes=[32, 32, 32],
    #     rank=1, activation='celu', dropout=0.03,
    #     strong_convexity=1e-6
    # ):
    def __init__(self, args):
        super(QMixer_WGAN_v2, self).__init__()
        
        self.strong_convexity = 1e-6    #strong_convexity
        self.hidden_layer_sizes = [32, 32, 32]  #hidden_layer_sizes
        self.dropout = 0.03  #dropout
        self.activation = 'celu'    #activation
        self.rank = 1   #rank
        self.in_dim = args.n_agents
        self.n_agents = args.n_agents

        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(self.in_dim, out_features, rank=self.rank, bias=True),
                nn.Dropout(self.dropout)
            )
            for out_features in self.hidden_layer_sizes
        ])
        
        sizes = zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])
        # self.convex_layers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(in_features, out_features, bias=True),
        #         nn.Dropout(self.dropout)
        #     )
        #     for (in_features, out_features) in sizes
        # ])

        # self.final_layer = nn.Linear(self.hidden_layer_sizes[-1], 1, bias=True)
        
        hypernet_embed = args.hypernet_embed     # QMIX的hypernet参数的中间channel数是64         self.args.hypernet_embed
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.hidden_layer_sizes[0]))
        self.hyper_w_2 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.embed_dim))
   
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_2 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_final = nn.Linear(self.state_dim, 1)

        # # V(s) instead of a bias for the last layers
        # self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
        #                        nn.ReLU(inplace=True),
        #                        nn.Linear(self.embed_dim, 1))
                                        

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)

        output = self.quadratic_layers[0](agent_qs)

        # First Layer, weights positive
        w1 = self.hyper_w_1(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.hidden_layer_sizes[0], self.embed_dim)    # 32 is the channel of the quadratic_layer
        b1 = b1.view(-1, 1, self.embed_dim)
        # hidden1 = F.elu(th.bmm(output, w1) + b1)
        hidden1 = th.bmm(output, w1) + b1
        quad1 = self.quadratic_layers[1](agent_qs)
        output = hidden1 + quad1
        if self.activation == 'celu':
            output = th.celu(output)
        elif self.activation == 'softplus':
            output = F.softplus(output)
        else:
            raise Exception('Activation is not specified or unknown.')

        # Second Layer, weights positive
        w2 = self.hyper_w_2(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b2 = self.hyper_b_2(states)
        w2 = w2.view(-1, self.embed_dim, self.embed_dim)
        b2 = b2.view(-1, 1, self.embed_dim)
        # hidden2 = F.elu(th.bmm(output, w2) + b2) 
        hidden2 = th.bmm(output, w2) + b2
        quad2 = self.quadratic_layers[2](agent_qs)
        output = hidden2 + quad2
        if self.activation == 'celu':
            output = th.celu(output)

        
        # Final, weights negative
        w_final = -self.hyper_w_final(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b_final = self.hyper_b_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        b_final = b_final.view(-1, 1, 1)
        hidden_final = th.bmm(output, w_final) + b_final
        output = hidden_final + .5 * self.strong_convexity * (agent_qs ** 2).sum(dim=2).reshape(-1, 1, 1)
        q_tot = output.view(bs, -1, 1)
        return q_tot
   
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

class QMixer_WGAN_v3(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    # def __init__(
    #     self, in_dim, 
    #     hidden_layer_sizes=[32, 32, 32],
    #     rank=1, activation='celu', dropout=0.03,
    #     strong_convexity=1e-6
    # ):
    def __init__(self, args):
        super(QMixer_WGAN_v2, self).__init__()
        
        self.strong_convexity = 1e-6    #strong_convexity
        self.hidden_layer_sizes = [32, 32, 32]  #hidden_layer_sizes
        self.dropout = 0.03  #dropout
        self.activation = 'celu'    #activation
        self.rank = 1   #rank
        self.in_dim = args.n_agents
        self.n_agents = args.n_agents

        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(self.in_dim, out_features, rank=self.rank, bias=True),
                nn.Dropout(self.dropout)
            )
            for out_features in self.hidden_layer_sizes
        ])
        
        sizes = zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])
        # self.convex_layers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(in_features, out_features, bias=True),
        #         nn.Dropout(self.dropout)
        #     )
        #     for (in_features, out_features) in sizes
        # ])

        # self.final_layer = nn.Linear(self.hidden_layer_sizes[-1], 1, bias=True)
        
        hypernet_embed = args.hypernet_embed     # QMIX的hypernet参数的中间channel数是64         self.args.hypernet_embed
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.hidden_layer_sizes[0]))
        self.hyper_w_2 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.embed_dim))
   
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_2 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_final = nn.Linear(self.state_dim, 1)

        # # V(s) instead of a bias for the last layers
        # self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
        #                        nn.ReLU(inplace=True),
        #                        nn.Linear(self.embed_dim, 1))
                                        

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)

        output = self.quadratic_layers[0](agent_qs)

        # First Layer, weights positive
        w1 = self.hyper_w_1(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.hidden_layer_sizes[0], self.embed_dim)    # 32 is the channel of the quadratic_layer
        b1 = b1.view(-1, 1, self.embed_dim)
        # hidden1 = F.elu(th.bmm(output, w1) + b1)
        hidden1 = th.bmm(output, w1) + b1
        quad1 = self.quadratic_layers[1](agent_qs)
        output = hidden1 + quad1
        if self.activation == 'celu':
            output = th.celu(output)
        elif self.activation == 'softplus':
            output = F.softplus(output)
        else:
            raise Exception('Activation is not specified or unknown.')
      
        # Final, weights negative
        w_final = -self.hyper_w_final(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b_final = self.hyper_b_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        b_final = b_final.view(-1, 1, 1)
        hidden_final = th.bmm(output, w_final) + b_final
        output = hidden_final + .5 * self.strong_convexity * (agent_qs ** 2).sum(dim=2).reshape(-1, 1, 1)
        q_tot = output.view(bs, -1, 1)
        return q_tot
   
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

    def neg_convexify(self):
        pass

    def neg_convexify_1(self):
        pass


class QMixer_ICNN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    # def __init__(
    #     self, in_dim, 
    #     hidden_layer_sizes=[32, 32, 32],
    #     rank=1, activation='celu', dropout=0.03,
    #     strong_convexity=1e-6
    # ):
    def __init__(self, args):
        super(QMixer_ICNN, self).__init__()
        
        self.strong_convexity = 1e-6    #strong_convexity
        self.hidden_layer_sizes = [32, 32, 32]  #hidden_layer_sizes
        self.dropout = 0.03  #dropout
        self.activation = 'celu'    #activation
        self.rank = 1   #rank
        self.in_dim = args.n_agents
        self.n_agents = args.n_agents

        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        
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

        self.final_layer = nn.Linear(self.hidden_layer_sizes[-1], 1, bias=True)
        
        hypernet_embed = args.hypernet_embed     # QMIX的hypernet参数的中间channel数是64         self.args.hypernet_embed
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.hidden_layer_sizes[0]))
        self.hyper_w_2 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.embed_dim))
   
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_2 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_final = nn.Linear(self.state_dim, 1)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))

        self.linear_y1 = nn.Sequential(
                nn.Linear(3, self.hidden_layer_sizes[1], bias=True),
                nn.Dropout(self.dropout)
            )
        self.linear_y2 = nn.Sequential(
                nn.Linear(3, self.hidden_layer_sizes[2], bias=True),
                nn.Dropout(self.dropout)
            )                                

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        
        agent_qs_reshape = agent_qs.reshape(-1, 1, self.n_agents)
        agent_qs = agent_qs.reshape(-1, 3)

        output = self.quadratic_layers[0](agent_qs_reshape)

        # First Layer, weights positive
        w1 = self.hyper_w_1(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.hidden_layer_sizes[0], self.embed_dim)    # 32 is the channel of the quadratic_layer
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden1 = F.elu(th.bmm(output, w1) + b1)
        output = hidden1 + self.linear_y1(agent_qs)
        if self.activation == 'celu':
            output = th.celu(output)
        elif self.activation == 'softplus':
            output = F.softplus(output)
        else:
            raise Exception('Activation is not specified or unknown.')

        # Second Layer, weights positive
        w2 = self.hyper_w_2(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b2 = self.hyper_b_2(states)
        w2 = w2.view(-1, self.embed_dim, self.embed_dim)
        b2 = b2.view(-1, 1, self.embed_dim)
        hidden2 = F.elu(th.bmm(output, w2) + b2) 
        quad2 = self.quadratic_layers[2](agent_qs)
        output = hidden2 + self.linear_y2(agent_qs)
        if self.activation == 'celu':
            output = th.celu(output)

        
        # Final, weights negative
        w_final = -self.hyper_w_final(states).abs() # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b_final = self.hyper_b_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        b_final = b_final.view(-1, 1, 1)
        hidden_final = th.bmm(output, w_final) + b_final
        output = hidden_final + .5 * self.strong_convexity * (agent_qs ** 2).sum(dim=2).reshape(-1, 1, 1)
        q_tot = output.view(bs, -1, 1)
        return q_tot
   
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

class QMixer_bak(nn.Module):
    def __init__(self, args):
        super(QMixer_bak, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.abs = getattr(self.args, 'abs', True)

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))
        

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        
        return q_tot

    def k(self, states):
        bs = states.size(0)
        w1 = th.abs(self.hyper_w_1(states))
        w_final = th.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)
        k = th.bmm(w1,w_final).view(bs, -1, self.n_agents)
        k = k / th.sum(k, dim=2, keepdim=True)
        return k

    def b(self, states):
        bs = states.size(0)
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        v = self.V(states).view(-1, 1, 1)
        b = th.bmm(b1, w_final) + v
        return b