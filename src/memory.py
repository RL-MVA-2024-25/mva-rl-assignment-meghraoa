import random
import torch
import numpy as np
from segment_tree import SumSegmentTree, MinSegmentTree

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.data = []
        self.index = 0
        self.device = device
    
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    
    def __len__(self):
        return len(self.data)
    
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha, beta, beta_increment_per_sampling, device):
        super().__init__(capacity, device)
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        
        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def append(self, s, a, r, s_, d):
        super().append(s, a, r, s_, d)
        priority = self.max_priority ** self.alpha
        self.sum_tree[self.tree_ptr] = priority
        self.min_tree[self.tree_ptr] = priority
        self.tree_ptr = (self.tree_ptr + 1) % self.capacity

    def sample(self, batch_size):
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])
        indices = self._sample_proportional(batch_size)
        batch = [self.data[idx] for idx in indices]
        s, a, r, s_, d = list(zip(*batch))
        weights = self._calculate_weights(indices, self.beta)

        return (
            torch.Tensor(np.array(s)).to(self.device),
            torch.Tensor(np.array(a)).to(self.device),
            torch.Tensor(np.array(r)).to(self.device),
            torch.Tensor(np.array(s_)).to(self.device),
            torch.Tensor(np.array(d)).to(self.device),
            torch.Tensor(weights).to(self.device),
            indices
        )
        
    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            new_priority = priority ** self.alpha
            self.sum_tree[idx] = new_priority
            self.min_tree[idx] = new_priority
        self.max_priority = max(self.max_priority, np.max(priorities))
            
    def _sample_proportional(self, batch_size):
        indices = np.zeros(batch_size, dtype=np.int32) 
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            indices[i] = self.sum_tree.retrieve(upperbound)

        return indices
    
    def _calculate_weights(self, indices, beta):
        weights = np.zeros(len(indices), dtype=np.float32)
        p_min = self.min_tree.query() / self.sum_tree.query()
        max_weight = (p_min * len(self)) ** (-beta)

        for i, idx in enumerate(indices):
            p_sample = self.sum_tree[idx] / self.sum_tree.query()
            weight = (p_sample * len(self)) ** (-beta) / max_weight        
            weights[i] = weight

        return weights
