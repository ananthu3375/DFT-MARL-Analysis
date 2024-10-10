import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from src.env_creator import env_creator
from src.batchify import batchify,batchify_obs,unbatchify
gpu = False
device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    
    return

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

class Agent(nn.Module):
    def __init__(self, env, name):
        super(Agent, self).__init__()
        self.env = env
        self.name = name
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, env.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, invalid_action_masks=None):
        logits = self.actor(x)
        split_logits = torch.split(logits,  1)
        #print(split_logits)
        if invalid_action_masks is not None:
            split_invalid_action_masks = torch.split(torch.tensor(invalid_action_masks), 1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
        else:
            multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])

        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, logprob.sum(0), entropy.sum(0), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    total_episodes = 2

    """ ENV SETUP """
    env = env_creator.create_env()
    num_agents = len(env.possible_agents)
    num_actions = env.action_space.n
    observation_size = env.observation_space.shape
    max_cycles = env.game.get_max_steps() + 4
    stack_size = 1
    depth = env.observation_space.shape
    observation = env.reset()
    

    """ LEARNER SETUP """
    red_agent = Agent(env, "red_agent").to(device)
    blue_agent = Agent(env, "blue_agent").to(device)
    agents = [red_agent,blue_agent]
    optimizer = {}
    for i in range(len(agents)):
        optimizer[agents[i].name] = optim.Adam(agents[i].parameters(), lr=0.001, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    
    rb_obs = {}
    rb_actions = {}
    rb_logprobs = {}
    rb_rewards = {}
    rb_terms = {}
    rb_values = {}
    rb_invalid_action_masks = {}
    rb_episodic_return = {"red_agent" : 0, "blue_agent" : 0}
    
    for i in range(len(agents)):
        index = agents[i].name
        rb_obs[index] = torch.zeros((max_cycles//2,) + observation_size).to(device)
        rb_actions[index] = torch.zeros((max_cycles//2, )).to(device) 
        rb_logprobs[index] = torch.zeros((max_cycles//2, )).to(device)    
        rb_rewards[index] = torch.zeros((max_cycles//2,)).to(device)   
        rb_terms[index] = torch.zeros((max_cycles//2,)).to(device)     
        rb_values[index] = torch.zeros((max_cycles//2,)).to(device)
        agent = env.game.players[i]
        #rb_invalid_action_masks[index] = torch.zeros((max_cycles, env.game.get_num_players_strategy_type(index)) + (agent.get_num_valid_actions(),)).to(device)
        rb_invalid_action_masks[index] = torch.zeros((max_cycles//2,)).to(device)
        #print(rb_obs,rb_actions,rb_logprobs,rb_rewards,rb_terms,rb_values)
        #print(env.game.num_observation())
        
    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        # collect an episode
        with torch.no_grad():
            obs = env.reset(seed=None) 
            # reset the episodic return
            total_episodic_return = {}
            for i in range(len(agents)):
                index = agents[i].name
                total_episodic_return[index] = 0
            agent_nn = red_agent
            red_step, blue_step = 0, 0
            for step in range(0, max_cycles):
                # rollover the observation
                actions = {}
                logprobs= {}
                values  = {}
                rewards = {}
                terminations = {}
                invalid_action_masks = {}
                for i in range(len(agents)):
                        index = agents[i].name
                        actions[index] = {}
                        logprobs[index] = {}
                        values[index] = {}
                        rewards[index] = {}
                        terminations[index] = {}
                        invalid_action_masks[index] = {}
                observation, reward, termination, truncation, info = env.last()
                agent = env.agent_selection
                if termination or truncation:
                       action = None
                else:
                    #print(observation)
                    invalid_action_masks[agent_nn.name] = env.game.get_mask(agent)
                    obs = batchify_obs(observation, device)
                    action_mask = env.game.get_mask(agent)
                    action_mask = batchify(action_mask,device)
                    action, logprob, _, value = agent_nn.get_action_and_value(torch.Tensor([observation]), invalid_action_masks = action_mask)
                    action = unbatchify(action)[0]
                    logprob = unbatchify(logprob)
                    value = unbatchify(value)
                    
                    actions[agent_nn.name] = action
                    logprobs[agent_nn.name] = logprob
                    values[agent_nn.name] = value
                    #print(actions, logprobs, values)
                observation, reward, termination, truncation, info = env.step(action)
                mat_step = red_step if (agent_nn.name == "red_agent") else blue_step
                if action != None:
                    rewards[agent_nn.name] = reward
                    terminations[agent_nn.name] = termination
                    rb_obs[agent_nn.name][mat_step] = batchify_obs([env.observation], device)
                    rb_rewards[agent_nn.name][mat_step] = batchify([rewards[agent_nn.name]], device)
                    rb_terms[agent_nn.name][mat_step] = batchify([terminations[agent_nn.name]], device)
                    rb_actions[agent_nn.name][mat_step] = batchify([actions[agent_nn.name]],device)
                    rb_logprobs[agent_nn.name][mat_step] = batchify([logprobs[agent_nn.name]],device)
                    rb_values[agent_nn.name][mat_step] = batchify([values[agent_nn.name]],device).flatten()
                    rb_invalid_action_masks[agent_nn.name][mat_step] = batchify_obs([invalid_action_masks[agent_nn.name]],device)
                total_episodic_return[agent_nn.name] += rb_rewards[agent_nn.name][mat_step].cpu().numpy()
                # if we reach termination or truncation, end
                if all([env.terminations[a] for a in env.terminations]) or all([env.truncations[a] for a in env.truncations]):
                    end_step = step
                    break
                agent_nn = blue_agent if (agent_nn.name == "red_agent") else red_agent
            print(rb_actions,rb_episodic_return,rb_invalid_action_masks,rb_logprobs,rb_obs,rb_rewards,rb_terms,rb_values)
                    
    # for _ in range(200):
       
    #     action = env.action_space.sample()
        
    #     agent= env.agent_selection
    #     observation, reward, terminations,truncations, info = env.step(action)
    #     #print(reward,observation)
    #     episodic_return[agent.name] = episodic_return[agent.name] + reward
    #     if terminations or truncations:
    #         observation = env.reset()
    #         print(f"episodic return: {episodic_return}")
    #         episodic_return = {"red_agent" : 0, "blue_agent" : 0}
    # env.close()
    
    
    #print(env.game.players[1].valid_actions_mask)
    # print(red_agent.get_action_and_value(torch.Tensor([observation]),invalid_action_masks=env.game.players[1].valid_actions_mask))