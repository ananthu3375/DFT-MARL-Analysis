import numpy as np
import torch


def batchify_obs(obs,device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    # print(obs)
    # obs = {agent: obs}
    # print(obs)
    # obs = np.stack([obs[a] for a in obs], axis=0)
    # print(obs)
    # transpose to be (batch, channel, height, width)
    # obs = obs.transpose(0, -1, 1, 2)
    # print(obs)
    # convert to torch
    obs = torch.Tensor(obs)
    
    return obs

def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    if isinstance(x, list) or isinstance(x, tuple):
        x = torch.Tensor(x).to(device)
    elif isinstance(x, torch.Tensor):
        x = x.to(device)
    else:
        raise ValueError("Unsupported input type. Must be list, tuple, or torch.Tensor.")
    return x


def unbatchify(x):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    # x = {a: x[i] for i, a in enumerate(env.possible_agents)}
    return x[0]  