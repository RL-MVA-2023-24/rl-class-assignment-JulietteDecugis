import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.device = device
    
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append((s, a, r, s_, d))

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    
    def __len__(self):
        return len(self.data)