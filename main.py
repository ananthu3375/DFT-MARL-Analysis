import gym
import torch
import torch.nn as nn
import numpy as np
from src.env_creator import env_creator
from src.batchify import batchify_obs, unbatchify,batchify
from torch.distributions.categorical import Categorical
import os
import torch.nn.functional as F
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

gpu = False
device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")
env = env_creator.create_env()
game_plays = 50
max_cycles = env.game.get_max_steps() + 4

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
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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
            nn.Linear(1024, 1),
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
            nn.Linear(1024, env.action_space.n),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, invalid_action_masks=None):
        logits = self.actor(x)
        #print(invalid_action_masks)
        #split_logits = torch.split(logits, 1, dim=1)
        if invalid_action_masks is not None:
            #split_invalid_action_masks = torch.split(torch.tensor(invalid_action_masks), 1)
            multi_categoricals = CategoricalMasked(logits=logits, masks=invalid_action_masks)
            #multi_categoricals = [CategoricalMasked(logits=split_logits,masks=split_invalid_action_masks)]
        else:
            multi_categoricals = Categorical(logits=logits)
        if action is None:
            action = multi_categoricals.sample()
        #print("action:",action)
        logprob = multi_categoricals.log_prob(action)
        entropy = multi_categoricals.entropy()
        return action, logprob, entropy




red_agent_actor = Agent(env, name="red_agent")
blue_agent_actor = Agent(env, name="blue_agent")
red_agent_actor.actor.load_state_dict(torch.load("PPO_Agent/25000ppo_actor_red_agent.pth"))
blue_agent_actor.actor.load_state_dict(torch.load("PPO_Agent/25000ppo_actor_blue_agent.pth"))
# red_agent_critic = torch.load('PPO_Agent/ppo_critic_red_agent.pth')
# blue_agent_critic = torch.load('PPO_Agent/ppo_critic_blue_agent.pth')

agents = {"red_agent":red_agent_actor,
          "blue_agent":blue_agent_actor}
wins = {"red":0,"blue":0}
red_win_actions = []
for i in range(game_plays):
    obs = env.reset(seed=None)
    agent_nn = "red_agent"
    total_episodic_return = {"red_agent":0, "blue_agent":0}
    actions = {}
    rewards = {}
    invalid_action_masks = {}
    for key in agents.keys():
            actions[key] = []
            rewards[key] = 0
            invalid_action_masks[key] = []
    #print(actions,rewards,invalid_action_masks)
    
    for step in range(0, max_cycles):
        observation, reward, termination, truncation, info = env.last()
        agent = env.agent_selection
        if termination or truncation:
            break
        else:
            invalid_action_masks[agent_nn] = env.game.get_mask(agent)[:len(env.game.actions)]
            #print(observation.shape)
            obs = batchify_obs(observation, device)
            #print(obs.shape)
            action_mask = env.game.get_mask(agent)[:len(env.game.actions)]
            #print(action_mask)
            action_mask = batchify_obs(action_mask[:len(env.game.actions)], device)
            action, logprob, _ = agents[agent_nn].get_action_and_value(torch.Tensor([observation]), invalid_action_masks=action_mask)
            action = unbatchify(action)
            #actions[agent_nn] = action
            observation, reward, termination, truncation, info = env.step(action)
            #print(type(actions[agent_nn]))
            actions[agent_nn].append(env.game.actions[action])
            rewards[agent_nn] += reward
            invalid_action_masks[agent_nn].append(agent.valid_actions_mask[:len(env.game.actions)])
        #agent = env.agent_selection
        agent_nn = "blue_agent" if (agent_nn == "red_agent") else "red_agent"
    print(actions,rewards)
    if env.system_state == 1:
        wins["blue"] += 1
        
    else:
        wins["red"] += 1
         #print(actions["red_agent"])
    #print(actions)
print(wins)


        