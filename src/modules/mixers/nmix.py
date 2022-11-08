import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
# from torch.nn import LayerNorm

from .layers import ConvexQuadratic


class Mixer_bak(nn.Module):
    def __init__(self, args, abs=True):
        super(Mixer_bak, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape)) 

        self.abs = abs # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        
        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))
        
        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1))

        if getattr(args, "use_orthogonal", False):
            for m in self.modules():
                orthogonal_init_(m)

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()
        
        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = self.hyper_w1(states).view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        
        # Second layer
        w2 = self.hyper_w2(states).view(-1, self.embed_dim, 1) # b * t, emb, 1
        b2= self.hyper_b2(states).view(-1, 1, 1)
        
        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)
            
        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1) # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2 # b * t, 1, 1
        
        return y.view(b, t, -1)

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)
        
class Mixer_wgan(nn.Module):
    def __init__(self, args, abs=True):
        super(Mixer_wgan, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape)) 

        self.abs = abs # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        

        self.strong_convexity = 1e-6    #strong_convexity
        self.dropout = 0.03  #dropout
        self.activation = 'celu'    #activation
        self.rank = 1   #rank

        self.hidden_layer_sizes = [32, 32] #, 32]
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(self.n_agents, out_features, rank=self.rank, bias=True),
                nn.Dropout(self.dropout)
            )
            for out_features in self.hidden_layer_sizes
        ])

        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))
        
        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.embed_dim * self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))

        # hyper w_final b_final
        self.hyper_w_final = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b_final = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1))

        # if getattr(args, "use_orthogonal", False):
        #     for m in self.modules():
        #         orthogonal_init_(m)

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()
        
        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = self.hyper_w1(states).abs().view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)

        hidden = F.elu(th.matmul(qvals, w1) + b1) # b * t, 1, emb
        quad = self.quadratic_layers[0](qvals)
        output = hidden + quad
        if self.activation == 'celu':
            output = th.celu(output)
        
        # Second layer
        w2 = self.hyper_w2(states).abs().view(-1, self.embed_dim,  self.embed_dim) # b * t, emb, 1
        b2= self.hyper_b2(states).view(-1, 1,  self.embed_dim)
        hidden = F.elu(th.matmul(output, w2) + b2) # b * t, 1, emb
        quad = self.quadratic_layers[1](qvals)
        output = hidden + quad
        if self.activation == 'celu':
            output = th.celu(output)

        # Final, weights negative
        w_final = -self.hyper_w_final(states).abs().view(-1, self.embed_dim, 1) # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b_final = self.hyper_b_final(states).view(-1, 1, 1)
        hidden = th.bmm(output, w_final) + b_final
        output = hidden + .5 * self.strong_convexity * (qvals ** 2).sum(dim=2).reshape(-1, 1, 1)
        q_tot = output.view(b, -1, 1)
        return q_tot


    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)

class Mixer_wgan_v2(nn.Module):
    def __init__(self, args, abs=True):
        super(Mixer_wgan_v2, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape)) 

        self.abs = abs # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        

        self.strong_convexity = 1e-6    #strong_convexity
        self.dropout = 0.03  #dropout
        self.activation = 'celu'    #activation
        self.rank = 1   #rank

        self.hidden_layer_sizes = [32, 32] #, 32]
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(self.n_agents, out_features, rank=self.rank, bias=True),
                nn.Dropout(self.dropout)
            )
            for out_features in self.hidden_layer_sizes
        ])

        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))
        
        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.embed_dim * self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))


        # hyper w_final b_final
        self.hyper_w_final = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b_final = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1))

        # if getattr(args, "use_orthogonal", False):
        #     for m in self.modules():
        #         orthogonal_init_(m)

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()
        
        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = self.hyper_w1(states).abs().view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)

        hidden = th.matmul(qvals, w1) + b1 # b * t, 1, emb
        quad = self.quadratic_layers[0](qvals)
        output = hidden + quad
        if self.activation == 'celu':
            output = th.celu(output)
        
        # Second layer
        w2 = self.hyper_w2(states).abs().view(-1, self.embed_dim,  self.embed_dim) # b * t, emb, 1
        b2= self.hyper_b2(states).view(-1, 1,  self.embed_dim)
        hidden = th.matmul(output, w2) + b2 # b * t, 1, emb
        quad = self.quadratic_layers[1](qvals)
        output = hidden + quad
        if self.activation == 'celu':
            output = th.celu(output)

        # Final, weights negative
        w_final = -self.hyper_w_final(states).abs().view(-1, self.embed_dim, 1) # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b_final = self.hyper_b_final(states).view(-1, 1, 1)
        hidden = th.bmm(output, w_final) + b_final
        output = hidden + .5 * self.strong_convexity * (qvals ** 2).sum(dim=2).reshape(-1, 1, 1)
        q_tot = output.view(b, -1, 1)
        return q_tot


    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)
        
class Mixer_wgan_v3(nn.Module):
    def __init__(self, args, abs=True):
        super(Mixer_wgan_v3, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape)) 

        self.abs = abs # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        

        self.strong_convexity = 1e-6    #strong_convexity
        self.dropout = 0.03  #dropout
        self.activation = 'celu'    #activation
        self.rank = 1   #rank

        self.hidden_layer_sizes = [32]
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(self.n_agents, out_features, rank=self.rank, bias=True),
                nn.Dropout(self.dropout)
            )
            for out_features in self.hidden_layer_sizes
        ])

        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))
        
        # # hyper w2 b2
        # self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(args.hypernet_embed, self.embed_dim * self.embed_dim))
        # self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))

        # hyper w_final b_final
        self.hyper_w_final = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b_final = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1))

        if getattr(args, "use_orthogonal", False):
            for m in self.modules():
                orthogonal_init_(m)

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()
        
        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = self.hyper_w1(states).abs().view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)

        hidden = th.matmul(qvals, w1) + b1 # b * t, 1, emb
        quad = self.quadratic_layers[0](qvals)
        output = hidden + quad
        if self.activation == 'celu':
            output = F.elu(output)
        
        # # Second layer
        # w2 = self.hyper_w2(states).abs().view(-1, self.embed_dim,  self.embed_dim) # b * t, emb, 1
        # b2= self.hyper_b2(states).view(-1, 1,  self.embed_dim)
        # hidden = F.elu(th.matmul(output, w2) + b2) # b * t, 1, emb
        # quad = self.quadratic_layers[1](qvals)
        # output = hidden + quad
        # if self.activation == 'celu':
        #     output = th.celu(output)

        # Final, weights negative
        w_final = -self.hyper_w_final(states).abs().view(-1, self.embed_dim, 1) # 暂时默认需要正的weights     if self.abs else self.hyper_w_1(states)
        b_final = self.hyper_b_final(states).view(-1, 1, 1)
        hidden = th.bmm(output, w_final) + b_final
        output = hidden# + .5 * self.strong_convexity * (qvals ** 2).sum(dim=2).reshape(-1, 1, 1)
        q_tot = output.view(b, -1, 1)
        return q_tot


    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)