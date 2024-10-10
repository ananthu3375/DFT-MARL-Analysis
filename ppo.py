from datetime import datetime
import argparse
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

from src.env_creator import env_creator
from src.batchify import batchify_obs, batchify, unbatchify
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

gpu = False
device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")


def parse_args():
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
        split_logits = torch.split(logits,  1)
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
    PATH = 'PPO_Agent/'
    args = parse_args()
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    total_episodes = 25000
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
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
    agents = [red_agent, blue_agent]
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
    rb_episodic_return = {"red_agent": 0, "blue_agent": 0}

    for i in range(len(agents)):
        index = agents[i].name
        rb_obs[index] = torch.zeros((max_cycles // 2,) + observation_size).to(device)
        rb_actions[index] = torch.zeros((max_cycles // 2,)).to(device)
        rb_logprobs[index] = torch.zeros((max_cycles // 2,)).to(device)
        rb_rewards[index] = torch.zeros((max_cycles // 2,)).to(device)
        rb_terms[index] = torch.zeros((max_cycles // 2,)).to(device)
        rb_values[index] = torch.zeros((max_cycles // 2,)).to(device)
        agent = env.game.players[i]
        rb_invalid_action_masks[index] = torch.zeros(max_cycles // 2, num_actions).to(device)
    
    # Initialize lists to store metrics for plotting
    value_losses_red = []
    policy_losses_red = []
    old_approx_kls_red = []
    approx_kls_red = []
    clip_fractions_red = []
    explained_variances_red = []

    value_losses_blue = []
    policy_losses_blue = []
    old_approx_kls_blue = []
    approx_kls_blue = []
    clip_fractions_blue = []
    explained_variances_blue = []

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        action_sequence = {"red_agent":[],"blue_agent":[]}
        # collect an episode
        with torch.no_grad():
            obs = env.reset(seed=None)
            # reset the episodic return
            #print(obs)
            total_episodic_return = {}
            for i in range(len(agents)):
                index = agents[i].name
                total_episodic_return[index] = 0
            agent_nn = red_agent
            red_step, blue_step = 0, 0
            
            for step in range(0, max_cycles):
                # rollover the observation
                actions = {}
                logprobs = {}
                values = {}
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
                #print(actions,logprobs,values,rewards,terminations,invalid_action_masks)
                observation, reward, termination, truncation, info = env.last()
                agent = env.agent_selection
                #print(agent.name)
                if termination or truncation:
                    action = None
                else:
                    invalid_action_masks[agent_nn.name] = env.game.get_mask(agent)
                    obs = batchify_obs(observation, device)
                    #print(obs.shape)
                    action_mask = env.game.get_mask(agent)
                    action_mask = batchify(action_mask, device)
                    #print(action_mask.shape)
                    action, logprob, _, value = agent_nn.get_action_and_value(torch.Tensor([observation]), invalid_action_masks=action_mask)
                    action = unbatchify(action)[0]
                    logprob = unbatchify(logprob)
                    value = unbatchify(value)

                    actions[agent_nn.name] = action
                    logprobs[agent_nn.name] = logprob
                    values[agent_nn.name] = value
                observation, reward, termination, truncation, info = env.step(action)
                #print(reward)
                # for agent in env.agents:
                #     print (agent.valid_actions_mask)
                # action_sequence[agent_nn.name].append(env.game.actions[action]) 
                #print(agent.name,reward)
                mat_step = red_step if (agent_nn.name == "red_agent") else blue_step
                if action is not None:
                    rewards[agent_nn.name] = reward
                    terminations[agent_nn.name] = termination
                    rb_obs[agent_nn.name][mat_step] = batchify_obs([env.observation], device)
                    rb_rewards[agent_nn.name][mat_step] = batchify([rewards[agent_nn.name]], device)
                    rb_terms[agent_nn.name][mat_step] = batchify([terminations[agent_nn.name]], device)
                    rb_actions[agent_nn.name][mat_step] = batchify([actions[agent_nn.name]], device)
                    rb_logprobs[agent_nn.name][mat_step] = batchify([logprobs[agent_nn.name]], device)
                    rb_values[agent_nn.name][mat_step] = batchify([values[agent_nn.name]], device).flatten()
                    rb_invalid_action_masks[agent_nn.name][mat_step] = batchify([invalid_action_masks[agent_nn.name]], device)
                total_episodic_return[agent_nn.name] += rb_rewards[agent_nn.name][mat_step].cpu().numpy()
                # if we reach termination or truncation, end
                if all([env.terminations[a] for a in env.terminations]) or all([env.truncations[a] for a in env.truncations]):
                    end_step = step
                    break
                if agent_nn.name == "red_agent":
                    red_step += 1
                else:
                    blue_step += 1
                agent_nn = blue_agent if (agent_nn.name == "red_agent") else red_agent
                
        if env.system.state == 0:
            print("red_wins")
        else:
            print("blue_wins")            
            # print(rb_rewards)
            # print(env.system.state)
            # print(env.observation)
            # print(total_episodic_return)
        # if episode>100:
        #     print(rb_actions)
        with torch.no_grad():
                rb_advantages = {}
                rb_returns = {}
                for i in range(len(agents)):
                   index = agents[i].name
                   rb_advantages[index] = torch.zeros_like(rb_rewards[index]).to(device)
                   mat_steps = red_step if index == "red_agent" else blue_step
                   for t in reversed(range(mat_steps)):
                       delta = (
                           rb_rewards[index][t]
                           + gamma * rb_values[index][t + 1] * rb_terms[index][t + 1]
                           - rb_values[index][t]
                       )
                       rb_advantages[index][t] = delta + gamma * gamma * rb_advantages[index][t + 1]
                   rb_returns[index] = rb_advantages[index] + rb_values[index] 
        # print(rb_advantages)
        # print(rb_returns)
        # convert our episodes to batch of individual transitions
        b_obs = {}
        b_logprobs = {}
        b_actions = {}
        b_returns = {}
        b_values = {}
        b_advantages = {}
        b_invalid_action_masks = {}
        for i in range(len(agents)):
            index = agents[i].name
            end_step = red_step if index == "red_agent" else blue_step
            b_obs[index] = torch.flatten(rb_obs[index][:end_step], start_dim=1)
            b_logprobs[index] = rb_logprobs[index][:end_step].view(-1, 1)
            b_actions[index] = rb_actions[index][:end_step].view(-1, 1)
            b_returns[index] = rb_returns[index][:end_step].view(-1, 1)
            b_values[index] = rb_values[index][:end_step].view(-1, 1)
            b_advantages[index] = rb_advantages[index][:end_step].view(-1, 1)
            b_invalid_action_masks[index] = torch.flatten(rb_invalid_action_masks[index][:end_step], start_dim=1)
        #print(len(b_obs["red_agent"]),len(b_logprobs["red_agent"]),len(b_actions["red_agent"]),len(b_returns["red_agent"]),len(b_values["red_agent"]),len(b_advantages["red_agent"]),len(b_invalid_action_masks["red_agent"]))
        # Optimizing the policy and value network
        b_index = {}
        batch_index = {}
        for i in range(len(agents)):
           index = agents[i].name
           b_index[index] = np.arange(len(b_actions[index]))
           clip_fracs = []
           for repeat in range(3):
               # shuffle the indices we use to access the data
               np.random.shuffle(b_index[index])
               for start in range(0, len(b_actions[index]), batch_size):
                   # select the indices we want to train on
                   end = start + batch_size
                   batch_index[index] = b_index[index][start:end]   
                   _, newlogprob, entropy, value = agents[i].get_action_and_value(
                       b_obs[index][batch_index[index]],
                       action = b_actions[index].long()[batch_index[index]], 
                       invalid_action_masks = b_invalid_action_masks[index][batch_index[index]]
                   )
                   logratio = newlogprob - b_logprobs[index][batch_index[index]]
                   #print(logratio.shape,newlogprob.shape,b_logprobs[index][batch_index[index]].shape)
                   ratio = logratio.exp()
        # print(ratio)
                   with torch.no_grad():
                      # calculate approx_kl http://joschu.net/blog/kl-approx.html
                       old_approx_kl = (-logratio).mean()
                       approx_kl = ((ratio - 1) - logratio).mean()
                       clip_fracs += [
                           ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                       ]
                    # normalize advantaegs
                   advantages = b_advantages[index][batch_index[index]]
                   advantages = (advantages - advantages.mean()) / (
                       advantages.std() + 1e-8
                   )
                   # Policy loss
                   pg_loss1 = -b_advantages[index][batch_index[index]] * ratio
                   pg_loss2 = -b_advantages[index][batch_index[index]] * torch.clamp(
                       ratio, 1 - clip_coef, 1 + clip_coef
                   )
                   pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                   # Value loss
                   value = value.flatten()
                   v_loss_unclipped = (value - b_returns[index][batch_index[index]]) ** 2
                   v_clipped = b_values[index][batch_index[index]] + torch.clamp(
                       value - b_values[index][batch_index[index]],
                       -clip_coef,
                       clip_coef,
                   )
                   v_loss_clipped = (v_clipped - b_returns[index][batch_index[index]]) ** 2
                   v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                   v_loss = 0.5 * v_loss_max.mean()

                   entropy_loss = entropy.mean()
                   loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                   optimizer[index].zero_grad()
                   loss.backward()
                   optimizer[index].step()
           y_pred, y_true = b_values[index].cpu().numpy(), b_returns[index].cpu().numpy()
           var_y = np.var(y_true)
           explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                     
           print("**" + index + " Team**")
           print(f"Training episode {episode}")
           print(f"Episodic Return: {np.mean(total_episodic_return[index])}")
           print(f"Episode Length: {end_step}")
           print("")
           print(f"Value Loss: {v_loss.item()}")
           print(f"Policy Loss: {pg_loss.item()}")
           print(f"Old Approx KL: {old_approx_kl.item()}")
           print(f"Approx KL: {approx_kl.item()}")
           print(f"Clip Fraction: {np.mean(clip_fracs)}")
           print(f"Explained Variance: {explained_var.item()}")
           print("\n***************************************\n")

           # Append metrics for plotting
           if index == "red_agent":
               value_losses_red.append(v_loss.item())
               policy_losses_red.append(pg_loss.item())
               old_approx_kls_red.append(old_approx_kl.item())
               approx_kls_red.append(approx_kl.item())
               clip_fractions_red.append(np.mean(clip_fracs))
               explained_variances_red.append(explained_var.item())
           else:
               value_losses_blue.append(v_loss.item())
               policy_losses_blue.append(pg_loss.item())
               old_approx_kls_blue.append(old_approx_kl.item())
               approx_kls_blue.append(approx_kl.item())
               clip_fractions_blue.append(np.mean(clip_fracs))
               explained_variances_blue.append(explained_var.item())

        print("\n-------------------------------------------\n")
    #print(len(value_losses_red),len(policy_losses_red),len(old_approx_kls_red),len(approx_kls_red),len(clip_fractions_red),len(explained_variances_red))
        #print("actions:",action_sequence)
    torch.save(red_agent.actor.state_dict(), PATH + str(total_episodes) + 'ppo_actor_red_agent.pth')
    torch.save(red_agent.critic.state_dict(), PATH + str(total_episodes) +'ppo_critic__red_agent.pth')
    torch.save(blue_agent.actor.state_dict(), PATH + str(total_episodes) + 'ppo_actor_blue_agent.pth')
    torch.save(blue_agent.critic.state_dict(), PATH + str(total_episodes) +'ppo_critic__blue_agent.pth')
# Plotting
skip_episodes = 100
episodes = range(0, total_episodes, skip_episodes)

# Plot metrics for red agent
plt.figure(figsize=(15, 10))  # Adjust figure size as needed
plt.subplot(3, 2, 1)
plt.plot(episodes, value_losses_red[::skip_episodes], label='Value Loss')
plt.xlabel('Episodes')
plt.ylabel('Value Loss')
plt.title('Value Loss for Red Agent')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(episodes, policy_losses_red[::skip_episodes], label='Policy Loss')
plt.xlabel('Episodes')
plt.ylabel('Policy Loss')
plt.title('Policy Loss for Red Agent')
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(episodes, old_approx_kls_red[::skip_episodes], label='Old Approx KL')
plt.xlabel('Episodes')
plt.ylabel('Old Approx KL')
plt.title('Old Approx KL for Red Agent')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(episodes, approx_kls_red[::skip_episodes], label='Approx KL')
plt.xlabel('Episodes')
plt.ylabel('Approx KL')
plt.title('Approx KL for Red Agent')
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(episodes, clip_fractions_red[::skip_episodes], label='Clip Fraction')
plt.xlabel('Episodes')
plt.ylabel('Clip Fraction')
plt.title('Clip Fraction for Red Agent')
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(episodes, explained_variances_red[::skip_episodes], label='Explained Variance')
plt.xlabel('Episodes')
plt.ylabel('Explained Variance')
plt.title('Explained Variance for Red Agent')
plt.legend()

# Save the plot for red agent
plt.tight_layout()
plt.savefig('plots/red_agent_metrics_'+str(total_episodes)+'_'+str(time)+'.png')

# Plot metrics for blue agent
plt.figure(figsize=(15, 10))  # Adjust figure size as needed
plt.subplot(3, 2, 1)
plt.plot(episodes, value_losses_blue[::skip_episodes], label='Value Loss')
plt.xlabel('Episodes')
plt.ylabel('Value Loss')
plt.title('Value Loss for Blue Agent')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(episodes, policy_losses_blue[::skip_episodes], label='Policy Loss')
plt.xlabel('Episodes')
plt.ylabel('Policy Loss')
plt.title('Policy Loss for Blue Agent')
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(episodes, old_approx_kls_blue[::skip_episodes], label='Old Approx KL')
plt.xlabel('Episodes')
plt.ylabel('Old Approx KL')
plt.title('Old Approx KL for Blue Agent')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(episodes, approx_kls_blue[::skip_episodes], label='Approx KL')
plt.xlabel('Episodes')
plt.ylabel('Approx KL')
plt.title('Approx KL for Blue Agent')
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(episodes, clip_fractions_blue[::skip_episodes], label='Clip Fraction')
plt.xlabel('Episodes')
plt.ylabel('Clip Fraction')
plt.title('Clip Fraction for Blue Agent')
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(episodes, explained_variances_blue[::skip_episodes], label='Explained Variance')
plt.xlabel('Episodes')
plt.ylabel('Explained Variance')
plt.title('Explained Variance for Blue Agent')
plt.legend()

# Save the plot for blue agent
plt.tight_layout()
plt.savefig('plots/blue_agent_metrics_'+str(total_episodes)+'_'+str(time)+'.png')
