import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer_bak
from modules.mixers.nmix import Mixer_wgan_v3
from modules.mixers.nmix import Mixer_wgan_v2
from modules.mixers.nmix import Mixer_wgan
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
from torch.distributions import Categorical

class NQLearnerNew:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix_bak":
            self.mixer = Mixer_bak(args)
        elif args.mixer == "qmix_wgan":
            self.mixer = Mixer_wgan(args)
        elif args.mixer == "qmix_wgan_v2":
            self.mixer = Mixer_wgan_v2(args)
        elif args.mixer == "qmix_wgan_v3":
            self.mixer = Mixer_wgan_v3(args)
        else:
            raise "mixer error"

        # initialize    
        for p in self.mixer.parameters():
            p.data = th.randn(p.shape, dtype=th.float32) / 20.


        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        
        # Mixer
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # chosen_action_qvals_ = chosen_action_qvals
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        
        # target_max_qvals_cMax = target_max_qvals.clone().detach()
        # cur_max_actions_cGlobalMax = cur_max_actions.clone().detach()
        
        # # Calculate n-step Q-Learning targets
        # target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])


        # Pick the Q-Values for the actions taken by each agent

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach_bak = mac_out.clone().detach()
            mac_out_detach_bak[avail_actions == 0] = -9999999
            mac_out_detach = mac_out.clone().detach()
            
            # current max qvals and actions
            # Mixer
            # device = th.device('cuda')
            cur_max_actions_for_target = mac_out_detach.max(dim=3, keepdim=True)[1]
            # actions_target = th.randint(0, 10, cur_max_actions_for_target.shape)#.to(device)   #.to(device)
            # chosen_action_qvals_for_target2 = th.gather(mac_out_detach[:,:], dim=3, index=cur_max_actions_for_target).squeeze(3) 
            # try:
            chosen_action_qvals_for_target = th.gather(mac_out_detach[:,:], dim=3, index=cur_max_actions_for_target).squeeze(3)  # Remove the last dim
                # print("1")
            # except Exception as e:
                # print("exception")

            chosen_action_qvals_for_target = self.mixer(chosen_action_qvals_for_target, batch["state"])
            chosen_action_qvals_for_target_cMax = chosen_action_qvals_for_target.clone().detach()
            cur_max_actions_cGlobalMax = cur_max_actions_for_target.clone().detach()
            cur_max_actions_target_cg = cur_max_actions_for_target.clone().detach()
            cur_max_actions_target = cur_max_actions_for_target.clone()
            # 接下来选max Q_target的actions，求cur_max_actions

            for agentn in range(self.args.n_agents):
                # cur_max_actions_c = cur_max_actions_cg.clone() #cur_max_actions_cg.clone()
                # avail_actions_c = avail_actions_cg.clone()

                for actionn in range(self.args.n_actions):
                          # available actions
                    cur_max_actions_target[:, :, agentn] = actionn   
                    chosen_action_qvals_for_target = th.gather(mac_out, dim=3, index=cur_max_actions_target).squeeze(3)  # Remove the last dim
                    # chosen_qmix_action_qvals = self.mixer(qmix_action_qvals, batch["state"][:, :-1])

                    # central_target_max_agent_qvals_c = th.gather(target_mac_out[:, :], 3,
                    # cur_max_actions_c[:,:].unsqueeze(4).repeat(1,1,1,1,self.args.n_actions)).squeeze(3)
                    # central_target_max_agent_qvals_c = th.gather(target_mac_out[:, :], 3, cur_max_actions_c).squeeze(3)
                    chosen_action_qvals_for_target = self.target_mixer(chosen_action_qvals_for_target, batch["state"])
                    condition = (chosen_action_qvals_for_target > chosen_action_qvals_for_target_cMax)
                    
                    chosen_action_qvals_for_target_cMax = th.where(condition, chosen_action_qvals_for_target, chosen_action_qvals_for_target_cMax)
                    cur_max_actions_cGlobalMax[:, :, agentn] = th.where(condition,
                        cur_max_actions_target[:, :, agentn], cur_max_actions_cGlobalMax[:, :, agentn])
                    
                    print("qmix_max_qvals_cMax[2:5, 2:5, 0]: ", chosen_action_qvals_for_target_cMax[2:5, 2:5, 0])
                        
                        
                    cur_max_actions_target = cur_max_actions_cGlobalMax


 
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions_target).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])
           
            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)


        # current max qvals and actions
        chosen_action_qvals_cMax = chosen_action_qvals.clone().detach()
        chosen_max_actions_cGlobalMax = actions.clone().detach()
        chosen_max_actions_cg = actions.clone().detach()
        # 接下来选max Q_target的actions，求cur_max_actions
        chosen_max_actions_c = chosen_max_actions_cg.clone().detach()

    # 先把mac_out_detach不等于-999999的拎出来
        for agentn in range(self.args.n_agents):
            for actionn in range(self.args.n_actions):
                # if mac_out_detach[0, 0, agentn, actionn] != -9999999:         # available actions
                # update the actionn of agentn
                chosen_max_actions_c[:, :, agentn] = actionn

                chosen_max_agent_qvals_c = th.gather(mac_out[:, :-1], 3, chosen_max_actions_c).squeeze(3)     # chosen_max_actions_c
                chosen_max_qvals_c = self.mixer(chosen_max_agent_qvals_c, batch["state"][:, :-1])
                # if higher q_val, then replace the max-q-val-action and the corresponding max q_val
                condition = (chosen_max_qvals_c > chosen_action_qvals_cMax)
                chosen_action_qvals_cMax = th.where(condition, chosen_max_qvals_c, chosen_action_qvals_cMax)
                chosen_max_actions_cGlobalMax[:, :, agentn] = th.where(condition,
                        chosen_max_actions_c[:, :, agentn], chosen_max_actions_cGlobalMax[:, :, agentn])

                chosen_max_actions_c = chosen_max_actions_cGlobalMax
                    

        # terminated = [False for _ in range(batch.batch_size)]
        # envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        # bs=envs_not_terminated
        # avail_actions_used = avail_actions[bs]


        # random_numbers = th.rand_like(mac_out[:, :, 0])
        # pick_random = (random_numbers < self.mac.get_epsilon()).long()
        # random_actions = Categorical(avail_actions_used.float()).sample().long()

        # picked_actions = pick_random * random_actions + (1 - pick_random) * chosen_max_actions_c


        # chosen_max_actions_f_qtot_Max = picked_actions
        chosen_max_actions_f_qtot_Max = chosen_max_actions_c
        chosen_max_qtot_f = th.gather(mac_out[:, :], 3, chosen_max_actions_f_qtot_Max).squeeze(3)     # chosen_max_actions_c
        chosen_max_qtot_f = self.mixer(chosen_max_qtot_f, batch["state"][:, :-1])

        # td_error = (chosen_action_qvals - targets.detach())
        td_error = (chosen_max_qtot_f - targets.detach())        # chosen_action_qvals_cMax
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        loss = L_td = masked_td_error.sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_max_qtot_f * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
