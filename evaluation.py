"""
File:     evaluation.py
Author:   Ananthu Ramesh S
Purpose:  To evaluate the outcomes of the trained model in the actual game
"""


import os
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from src.env_creator import env_creator
import matplotlib.pyplot as plt
import gym


# to store memory
class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        # input shape = shape of observations
        self.mem_size = max_size
        self.mem_cntr = 0  # index of last stored memory
        # print("input shape",input_shape)
        self.state_memory = np.zeros((self.mem_size, input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)  # for done flags

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        # print(self.mem_cntr)
        # print(state_)

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        # print(done)
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    # for using form sampling of memory when agent learns
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)  # so that non filled values zeros are not chosen
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        # print("states''''''''''''''",states)
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class Agent():
    def __init__(self, name, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=1e-3,
                 replace=1000, chkpt_dir='tmp/double_dqn'):
        # epsilon is the fraction of time it spends on taking random actions
        # how often to replace agent network
        self.name = name                           
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.learn_step_counter = 0
        self.action_space = [i for i in range(self.n_actions)]
        self.steps_per_episode = []                                                                   
        self.rewards_per_episode = []                                                                    
        self.loss_history = []  # To store loss values                                                                                                                                 
        self.episode_loss = []  # List to store losses for the current episode                                  
        self.episode_losses = []  # List to store average loss per episode
        self.epsilon_history = []
        # print("input dimensions,,,,,,,,,,,,,,,,,,",input_dims)
        self.memory = ReplayBuffer(mem_size, len(input_dims))
        self.q_eval = DoubleDeepQNetwork(self.lr, self.n_actions,
                                         input_dims=self.input_dims,
                                         name=name + 'DoubleDeepQNetwork_q_eval',
                                         chkpt_dir=self.chkpt_dir)
        self.q_next = DoubleDeepQNetwork(self.lr, self.n_actions,
                                         input_dims=self.input_dims,
                                         name=name + 'DoubleDeepQNetwork_q_next',
                                         chkpt_dir=self.chkpt_dir)
        self.training_actions = []  # store actions during training             

    def choose_action(self, observation, action_mask):
        if action_mask is not None:
            valid_actions = np.where(action_mask)[0]  # Get indices of valid actions
            if len(valid_actions) == 0:
                return 0
            if np.random.random() > self.epsilon:
                state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                _, advantage = self.q_eval.forward(state)
                masked_advantage = advantage[:, valid_actions]  # Consider only valid actions
                action = valid_actions[T.argmax(masked_advantage).item()]
            else:
                action = np.random.choice(valid_actions)
        else:
            if np.random.random() > self.epsilon:
                state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                _, advantage = self.q_eval.forward(state)
                action = T.argmax(advantage).item()
            else:
                action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
        self.training_actions.append(action)  # Track action in training                                                         
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    # to reduce epsilon over time
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self, i):
        # if the model hasn't filled the batch_size of memory
        if self.memory.mem_cntr < self.batch_size:
            return
        # print(i)
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        # sampling of memory
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
        # print("state........................",state)
        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        dones = T.tensor(done).bool().to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        indices = np.arange(self.batch_size)
        # print("states........................",states)
        if (states.numel() != 0):
            V_s, A_s = self.q_eval.forward(states)
            V_s_, A_s_ = self.q_next.forward(states_)

            V_s_eval, A_s_eval = self.q_eval.forward(states_)

            q_pred = T.add(V_s,
                           (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
            q_next = T.add(V_s_,
                           (A_s_ - A_s_.mean(dim=1, keepdim=True)))
            q_eval = T.add(V_s_eval,
                           (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
            max_actions = T.argmax(q_eval, dim=1)

            q_next[dones] = 0.0
            q_target = rewards + self.gamma * q_next[indices, max_actions]

            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
            
            # print(f"Training step {self.learn_step_counter}: Loss = {loss.item()}") # To print loss for each step
            
            self.episode_loss.append(loss.item())  # Append the loss value for the current episode                         
            
            self.loss_history.append(loss.item())  # Append the loss value                                                 
            self.epsilon_history.append(self.epsilon)  # Track epsilon                                                     
            
            loss.backward()
            
            max_grad_norm = 3.0  # gradient clipping                                                          
            utils.clip_grad_norm_(self.q_eval.parameters(), max_grad_norm)                                         
            
            self.q_eval.optimizer.step()
            self.learn_step_counter += 1
            self.decrement_epsilon()
            # print(f"Loss: {loss.item()}")
            
    def end_of_episode(self):                                                                                 
        # Print average loss for the episode
        if self.episode_loss:
            avg_loss = np.mean(self.episode_loss)
            self.episode_losses.append(avg_loss)  # Store the average loss for plotting
            print(f"Episode Loss for {self.name}: {avg_loss:.4f}")
            self.episode_loss = []  # Clear the list for the next episode


class DoubleDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DoubleDeepQNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name)
        # input_dims =
        # print("..............",len(input_dims))
        print("Input dimensions length:", len(input_dims))
        self.fcl = nn.Linear(len(input_dims), 512)                                                     
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization layer                                         
        self.fc2 = nn.Linear(512, 512)  # New hidden layer                                                      
        self.bn2 = nn.BatchNorm1d(512)                                                                                    
        self.dropout = nn.Dropout(p=0.2)  # Dropout layer with 20% dropout rate                                             
        self.V = nn.Linear(512, 1)  # the value
        self.A = nn.Linear(512, n_actions)  # the advantage of actions, relative value of each action
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)   # weight decay
        self.loss = nn.SmoothL1Loss()  # Huber loss : Replaced MSELoss with HuberLoss                                       
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # print(state[0].size())
        # print("state...................",state)
        # flat1 = F.relu(self.fcl(state))
        flat1 = F.leaky_relu(self.fcl(state), negative_slope=0.01)                                                    
        flat1 = self.dropout(flat1)  # Applying dropout                                                                 
        if flat1.size(0) > 1:                                                                                           
            flat1 = self.bn1(flat1)                                                                                     
        
        # Second hidden layer
        flat2 = F.leaky_relu(self.fc2(flat1), negative_slope=0.01)                                                                                  
        flat2 = self.dropout(flat2)                                                                                      
        if flat2.size(0) > 1:                                                                                            
            flat2 = self.bn2(flat2)                                                                                      
            
        V = self.V(flat2)                                                                                                
        A = self.A(flat2)                                                                                                

        return V, A

    def save_checkpoint(self):
        print('.......saving a checkpoint....')
        # print(self.state_dict())
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        # print('... loading checkpoint ....')
        print(f'... loading checkpoint from {self.checkpoint_file} ...')                                                
        self.load_state_dict(T.load(self.checkpoint_file, map_location=T.device('cpu')))


def calculate_and_plot_event_usage(actions):
    # Calculate event counts for both agents
    red_agent_event_count = {}
    for game_actions in actions["red_agent"]:
        for event in game_actions:
            if event in red_agent_event_count:
                red_agent_event_count[event] += 1
            else:
                red_agent_event_count[event] = 1
    print("\n\nRed Agent Event Usage:")
    for event in sorted(red_agent_event_count.keys()):
        count = red_agent_event_count[event]
        print(f"Event {event}: {count} times")

    blue_agent_event_count = {}
    for game_actions in actions["blue_agent"]:
        for event in game_actions:
            if event in blue_agent_event_count:
                blue_agent_event_count[event] += 1
            else:
                blue_agent_event_count[event] = 1
    print("\nBlue Agent Event Usage:")
    for event in sorted(blue_agent_event_count.keys()):
        count = blue_agent_event_count[event]
        print(f"Event {event}: {count} times")

    # Combine the events to create a unified list of all events
    all_events = sorted(set(red_agent_event_count.keys()).union(blue_agent_event_count.keys()))
    red_counts = [red_agent_event_count.get(event, 0) for event in all_events]
    blue_counts = [blue_agent_event_count.get(event, 0) for event in all_events]
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    index = range(len(all_events))
    plt.bar(index, red_counts, width=bar_width, color='red', label='Red Agent')
    plt.bar([i + bar_width for i in index], blue_counts, width=bar_width, color='blue', label='Blue Agent')
    plt.xlabel('Events')
    plt.ylabel('Count')
    plt.title('Red vs Blue Agent Event Usage')
    plt.xticks([i + bar_width / 2 for i in index], all_events, rotation=45)
    plt.legend()
    plt.tight_layout()
    save_dir = '01_100K_DFT_MARL-ddqn_analysisGraphs'
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 'game_event_usage.png')
    plt.savefig(file_path, format='png')
    plt.close()
    print("\nRed vs Blue Agent Event Usage plot saved at:", file_path)


def actual_game():
    # Initialize the environment
    env = env_creator.create_env()
    num_agents = len(env.possible_agents)
    num_actions = env.action_space.n
    observation_size = env.observation_space.shape
    observation = env.game.observations
    max_cycles = env.game.get_max_steps() + 4

    # Set the number of evaluation games
    num_games = 50

    # Load the saved models for both agents (red and blue)
    red_agent = Agent(name="red_agent_01_100K", gamma=0.99, epsilon=1.0, lr=5e-6,
                      input_dims=observation, n_actions=num_actions, mem_size=1000000, eps_min=0.01,
                      batch_size=32, eps_dec=1e-3, replace=100)
    blue_agent = Agent(name="blue_agent_01_100K", gamma=0.99, epsilon=1.0, lr=5e-6,
                       input_dims=observation, n_actions=num_actions, mem_size=1000000, eps_min=0.01,
                       batch_size=32, eps_dec=1e-3, replace=100)

    agents = {"red_agent": red_agent, "blue_agent": blue_agent}
    red_agent.load_models()
    blue_agent.load_models()
    
    red_agent.epsilon = 0.00                                                                        
    blue_agent.epsilon = 0.00 

    # To store results
    scores = {"red_agent": [], "blue_agent": []}
    actions = {"red_agent": [], "blue_agent": []}
    wins = {"red_agent": 0, "blue_agent": 0}
    red_system_states = []  # To store system state when the red agent takes an action
    blue_system_states = []  # To store system state when the blue agent takes an action
    action_sequence_counts = {"red_agent": {}, "blue_agent": {}}  # Dictionary to store all/unique sequences and their counts
    alternating_action_sequence = []  # New list to store alternating actions sequence

    # Run the evaluation games
    for i in range(num_games):
        print(f"Evaluation Game {i + 1}")
        done = False
        observation = env.reset()
        score = {}
        for k, v in agents.items():
            score[k] = 0
        agent_nn = "red_agent"  # Red agent starts first
        action_taken = {"red_agent": [], "blue_agent": []}
        game_action_sequence = []  # List to store the action sequence for this game

        while not done:
            observation, reward, termination, truncation, info = env.last()  # Get the last observation and action mask
            if termination or truncation:
                action = None
                done = True
                break

            env_agent = env.agent_selection
            action_mask = env.game.get_mask(env_agent)
            action = agents[agent_nn].choose_action(observation, action_mask)
            action_taken[agent_nn].append(env.game.actions[action])
            new_observation, reward, termination, truncation, info = env.step(action)  # Step the environment with the chosen action
            game_action_sequence.append((agent_nn, env.game.actions[action]))

            if agent_nn == "red_agent":
                system_state = env.system.state
                red_system_states.append((env.game.actions[action], system_state))  # Store action and system state
                # print(f"Red Agent Action: {env.game.actions[action]}, System State: {system_state}")
            else:
                system_state = env.system.state
                blue_system_states.append((env.game.actions[action], system_state))

            if termination or truncation:
                done = True
            score[agent_nn] += reward  # Update scores

            agent_nn = "blue_agent" if agent_nn == "red_agent" else "red_agent"  # Switch agents after each step

        alternating_action_sequence.append(game_action_sequence)

        for k in scores.keys(): # After each game, update the scores and wins
            # scores[k].append(score[k])
            if env.system.state == 1:  # 1 indicates the blue agent win
                if k == "blue_agent":
                    scores[k].append(f'{score[k]} - won')
                else:
                    scores[k].append(f'{score[k]} - lost')
            else:  # Red agent win
                if k == "red_agent":
                    scores[k].append(f'{score[k]} - won')
                else:
                    scores[k].append(f'{score[k]} - lost')
        
        for k in actions.keys():
            actions[k].append(action_taken[k])
            
        # Checks and updates action sequences and its counts
        for agent in action_sequence_counts.keys():
            action_sequence = tuple(action_taken[agent])  # Tuple for immutability
            if action_sequence in action_sequence_counts[agent]:
                action_sequence_counts[agent][action_sequence] += 1
            else:
                action_sequence_counts[agent][action_sequence] = 1
            
        if env.system.state == 1:  # 1 indicates the blue agent win
            wins["blue_agent"] += 1
            # print(f"Game {i + 1}: Blue Agent wins!")
        else:
            wins["red_agent"] += 1
            # print(f"Game {i + 1}: Red Agent wins!")

        # Print the alternating action sequence for all games
    # for i, game_sequence in enumerate(alternating_action_sequence):
    #     formatted_sequence = []
    #     red_system_state_index = 0  # Track the index for red system state
    #     blue_system_state_index = 0  # Track the index for blue system state
    #
    #     for idx, (agent, action) in enumerate(game_sequence):
    #         if agent == "red_agent":
    #             state = red_system_states[red_system_state_index][1]  # Fetch system state for red agent
    #             red_system_state_index += 1
    #             formatted_sequence.append(f'"{agent}: {action}, State: {state}"')
    #         else:
    #             state = blue_system_states[blue_system_state_index][1]  # Fetch system state for blue agent
    #             blue_system_state_index += 1
    #             formatted_sequence.append(f'"{agent}: {action}, State: {state}"')
    #
    #     print(f"\nGame {i + 1} Action Sequence: [{', '.join(formatted_sequence)}]")

    for i, game_sequence in enumerate(alternating_action_sequence):
        formatted_sequence = []
        for idx, (agent, action) in enumerate(game_sequence):
            formatted_sequence.append(f'"{agent}: {action}"')
        print(f"\nGame {i + 1} Action Sequence: [{', '.join(formatted_sequence)}]")

    print(f"\n\nScores: {scores}")
    print(f"\n\nWins: {wins}")
    # print(f"\n\nActions: {actions}")
    print(f"\n\nSystem States for Red Agent: {red_system_states}")
    print(f"\n\nSystem States for Blue Agent: {blue_system_states}")

    # To print action sequence counts
    for agent in action_sequence_counts.keys():
        print(f"\nAction Sequence Counts for {agent}:")
        sequence_occurrences = {}  # Dictionary to store sequences and their number of times occurrence
        for sequence in actions[agent]:
            score_result = scores[agent][actions[agent].index(sequence)]
            sequence_key = tuple(sequence)  # Convert sequence to tuple to use as a dictionary key
            if sequence_key in sequence_occurrences:  # Checks if the sequence already exists in the dictionary
                sequence_occurrences[sequence_key]['occurrence'] += 1
            else:
                sequence_occurrences[sequence_key] = {'result': score_result,
                                                      'occurrence': 1}  # Initialized occurrence count to 1
        for sequence_key, data in sequence_occurrences.items():
            print(f"{sequence_key} = {data['result']} - Occurrence= {data['occurrence']}")
    
    calculate_and_plot_event_usage(actions)
    print(f"\n\nWins: {wins}")
    print("Finished executing the games. Returning actions and wins.")
    return {"actions": actions, "wins": wins}


if __name__ == '__main__':
    actual_game()
