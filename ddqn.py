import os
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.env_creator import env_creator
import matplotlib.pyplot as plt
import gym


def set_seed(seed):  ####
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed(seed)
        T.cuda.manual_seed_all(seed)
    T.backends.cudnn.deterministic = True  # Makes computations deterministic, but can slow down training
    T.backends.cudnn.benchmark = False  # Ensures reproducibility


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

    # for usinform sampling of memory when agent learns
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)  # so that non filled valyes zeros are not chosen
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
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, chkpt_dir='tmp/double_dqn'):
        # epsilon is the fraction of timeit spends on taking random actions
        # how often to replace agent network
        self.name = name  ###
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
        self.loss_history = []  # To store loss values                                                             ###
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

        self.training_actions = []  # store actions during training                                                 ###

        # random number less than epsilon it takes a random action
        # if the random number is greater than epsilon it takes a greedy action

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
        self.training_actions.append(
            action)  # Track action in training                                                           ###

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
            loss.backward()
            self.q_eval.optimizer.step()
            self.learn_step_counter += 1
            self.decrement_epsilon()
            self.loss_history.append(
                loss.item())  # Append the loss value                                                          ###
            # print(f"Loss: {loss.item()}")                                                                                           ###


class DoubleDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DoubleDeepQNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name)
        # input_dims =
        # print("..............",len(input_dims))
        self.fcl = nn.Linear(len(input_dims), 512)
        self.V = nn.Linear(512, 1)  # the value
        self.A = nn.Linear(512, n_actions)  # the advantage of actions, relative value of each action

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # print(state[0].size())
        # print("state...................",state)
        flat1 = F.relu(self.fcl(state))
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

    def save_checkpoint(self):
        print('.......saving a checkpoint....')
        # print(self.state_dict())
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ....')
        self.load_state_dict(T.load(self.checkpoint_file, map_location=T.device('cpu')))


def training_event_mapping(n_actions):  ####
    """Create a mapping from action indices to event names."""
    events = ["No Action"]  # Start with "No Action" for index 0
    for i in range(1, n_actions):
        events.append(chr(64 + i))  # 65 is ASCII for 'A'
    return {i: event for i, event in enumerate(events)}


def print_training_event_usage(agent):  ####
    num_actions = agent.n_actions
    event_mapping = training_event_mapping(num_actions)

    event_count = {}
    for action in agent.training_actions:
        event_name = event_mapping.get(action, f"Unknown Event {action}")  # Use the mapping
        if event_name in event_count:
            event_count[event_name] += 1
        else:
            event_count[event_name] = 1

    print(f"{agent.name} Training Event Usage:")
    for event, count in sorted(event_count.items()):
        print(f"{event}: {count} times")


def print_event_usage(actions):  ####
    event_count = {}
    for game_actions in actions["red_agent"]:
        for event in game_actions:
            if event in event_count:
                event_count[event] += 1
            else:
                event_count[event] = 1

    print("Red Agent Event Usage:")
    for event in sorted(event_count.keys()):
        count = event_count[event]
        print(f"Event {event}: {count} times")


def plot_event_usage(event_count, title, plot_filename):  ####
    events = sorted(list(event_count.keys()))
    counts = [event_count[event] for event in events]

    max_count = max(counts)
    colors = ['green' if count == max_count else 'blue' for count in counts]

    plt.figure(figsize=(10, 6))
    plt.bar(events, counts, color=colors)
    plt.xlabel('Events')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    save_dir = 'C:\\Users\\HP\\PycharmProjects\\DFT_MARL\\ddqn_analysisGraphs_25K'
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, plot_filename)
    plt.savefig(file_path, format='png')
    plt.close()


def plot_loss(agent, save_dir, agent_name):  ####
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.plot(agent.loss_history)
    plt.title(f"Training Loss for {agent_name} Over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    save_path = os.path.join(save_dir, f"{agent_name}_loss.png")
    plt.savefig(save_path)
    plt.clf()


def play():
    env = env_creator.create_env()
    num_agents = len(env.possible_agents)
    num_actions = env.action_space.n
    observation_size = env.observation_space.shape
    observation = env.game.observations
    max_cycles = env.game.get_max_steps() + 4

    num_games = 50
    red_agent = Agent(name="red_agent_25000", gamma=0.99, epsilon=1.0, lr=5e-4,
                      input_dims=observation, n_actions=num_actions, mem_size=1000000, eps_min=0.01,
                      batch_size=32, eps_dec=1e-3, replace=100)
    blue_agent = Agent(name="blue_agent_25000", gamma=0.99, epsilon=1.0, lr=5e-4,
                       input_dims=observation, n_actions=num_actions, mem_size=1000000, eps_min=0.01,
                       batch_size=32, eps_dec=1e-3, replace=100)
    agents = {"red_agent": red_agent, "blue_agent": blue_agent}
    for k, v in agents.items():
        v.load_models()

    # Fine-tuning epsilon for exploitation after training
    red_agent.epsilon = 0.01  ###
    blue_agent.epsilon = 0.01  ###

    scores = {"red_agent": [], "blue_agent": []}
    actions = {"red_agent": [], "blue_agent": []}
    wins = {"red_agent": 0, "blue_agent": 0}
    for i in range(num_games):
        print("episode ", i)
        done = False
        observation = env.reset()
        score = {}
        for k, v in agents.items():
            score[k] = 0
        agent_nn = "red_agent"
        action_taken = {"red_agent": [], "blue_agent": []}
        while not done:
            observation, reward, termination, truncation, info = env.last()
            # print(observation)
            if termination or truncation:
                action = None
                break
            env_agent = env.agent_selection
            action_mask = env.game.get_mask(env_agent)
            action = agents[agent_nn].choose_action(observation, action_mask)
            action_taken[agent_nn].append(env.game.actions[action])
            new_observation, reward, termination, truncation, info = env.step(action)
            if termination or truncation:
                done = True
            score[agent_nn] += reward

            agent_nn = "blue_agent" if agent_nn == "red_agent" else "red_agent"
        for k in scores.keys():
            scores[k].append(score[k])
        for k in actions.keys():
            actions[k].append(action_taken[k])
        if env.system.state == 1:
            wins["blue_agent"] += 1
        else:
            wins["red_agent"] += 1

    print(scores)
    print(wins)
    # print(actions)

    # Calculate and plot event usage for Red Agent                                                                  #####
    red_agent_event_count = {}
    for game_actions in actions["red_agent"]:
        for event in game_actions:
            if event in red_agent_event_count:
                red_agent_event_count[event] += 1
            else:
                red_agent_event_count[event] = 1

    print("Red Agent Event Usage:")
    for event in sorted(red_agent_event_count.keys()):
        count = red_agent_event_count[event]
        print(f"Event {event}: {count} times")

    # Plot the event usage for Red Agent
    plot_event_usage(red_agent_event_count, 'Red Agent Event Usage', 'red_agent_event_usage.png')

    # Calculate and plot event usage for Blue Agent (optional, if desired)
    blue_agent_event_count = {}
    for game_actions in actions["blue_agent"]:
        for event in game_actions:
            if event in blue_agent_event_count:
                blue_agent_event_count[event] += 1
            else:
                blue_agent_event_count[event] = 1

    print("Blue Agent Event Usage:")
    for event in sorted(blue_agent_event_count.keys()):
        count = blue_agent_event_count[event]
        print(f"Event {event}: {count} times")

    # Plot the event usage for Blue Agent
    plot_event_usage(blue_agent_event_count, 'Blue Agent Event Usage', 'blue_agent_event_usage.png')  ######

    # Print event usage by the red agent
    # print_event_usage(actions)                                                                                       ###


if __name__ == '__main__':
    set_seed(
        42)  # Set a fixed seed value for reproducibility                                                              ###
    env = env_creator.create_env()
    num_agents = len(env.possible_agents)
    num_actions = env.action_space.n
    observation_size = env.observation_space.shape
    observation = env.game.observations
    max_cycles = env.game.get_max_steps() + 4

    num_games = 1000
    load_checkpoint = False

    red_agent = Agent(name="red_agent_25000", gamma=0.99, epsilon=0.5, lr=5e-4,
                      input_dims=observation, n_actions=num_actions, mem_size=1000000, eps_min=0.01,
                      batch_size=32, eps_dec=1e-3, replace=100)
    blue_agent = Agent(name="blue_agent_25000", gamma=0.99, epsilon=0.5, lr=5e-4,
                       input_dims=observation, n_actions=num_actions, mem_size=1000000, eps_min=0.01,
                       batch_size=32, eps_dec=1e-3, replace=100)
    agents = {"red_agent": red_agent, "blue_agent": blue_agent}
    if load_checkpoint:
        for k, v in agents.items():
            v.load_models()

    filename = 'DFT-DDQN.png'
    scores, eps_history = {}, {}
    wins = {"red_agent": 0, "blue_agent": 0}
    actions_taken = {"red_agent": [],
                     "blue_agent": []}  # Store actions taken by each agent                             ###
    for k in agents.keys():
        scores[k] = []
        eps_history[k] = []

    for i in range(num_games):
        print("episode ", i)
        done = False
        observation = env.reset()
        score = {}
        for k, v in agents.items():
            score[k] = 0
            actions_taken[k].append([])  # Start a new list for this episode            ###
        agent_nn = "red_agent"
        while not done:
            observation, reward, termination, truncation, info = env.last()
            # print(observation)
            if termination or truncation:
                action = None
                break
            env_agent = env.agent_selection
            action_mask = env.game.get_mask(env_agent)
            action = agents[agent_nn].choose_action(observation, action_mask)
            new_observation, reward, termination, truncation, info = env.step(action)

            # Storing the action taken
            actions_taken[agent_nn][-1].append(action)  # Append action to the last episode's list          ###

            if termination or truncation:
                done = True
            score[agent_nn] += reward
            agents[agent_nn].store_transition(observation, action, reward,
                                              new_observation, done)
            agents[agent_nn].learn(i)
            agent_nn = "blue_agent" if agent_nn == "red_agent" else "red_agent"
        for k in scores.keys():
            scores[k].append(score[k])
        if env.system.state == 1:
            wins["blue_agent"] += 1
        else:
            wins["red_agent"] += 1

    save_dir = 'C:\\Users\\HP\\PycharmProjects\\DFT_MARL\\ddqn_analysisGraphs_25K'  ###
    plot_loss(red_agent, save_dir, "red_agent")  ###
    plot_loss(blue_agent, save_dir, "blue_agent")  ###
    print(wins)

    for k, v in agents.items():
        v.save_models()

        print_training_event_usage(
            v)  # Print event usage after training                                                             ###

        # save_dir = 'C:\\Users\\HP\\PycharmProjects\\DFT_MARL\\ddqn_analysisGraphs'
        # plot_training_metrics(v, save_dir) # Plot metrics after training                                                             ###

        # Create and plot event usage for training                                                                                    ####
        training_event_count = {}
        for action in v.training_actions:
            event_name = training_event_mapping(v.n_actions).get(action, f"Unknown Event {action}")
            if event_name in training_event_count:
                training_event_count[event_name] += 1
            else:
                training_event_count[event_name] = 1

        plot_event_usage(training_event_count, f'{v.name} Training Event Usage', f'{v.name}_training_event_usage.png')

#     play()

play()