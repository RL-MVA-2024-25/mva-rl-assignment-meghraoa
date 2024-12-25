import numpy as np

INF = float('inf')

class SegmentTree:
    def __init__(self, capacity, operation, neutral_value):
        self.capacity = int(capacity)
        self.operation = operation
        self.neutral_value = neutral_value
        self.tree = np.full(2 * capacity, neutral_value, dtype=np.float32)

    def _query(self, start, end, node, node_start, node_end):
        if start > node_end or end < node_start:
            return self.neutral_value
        if start <= node_start and end >= node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        left_result = self._query(start, end, 2 * node, node_start, mid)
        right_result = self._query(start, end, 2 * node + 1, mid + 1, node_end)
        return self.operation(left_result, right_result)

    def query(self, start=0, end=None):
        if end is None:
            end = self.capacity
        return self._query(start, end - 1, 1, 0, self.capacity - 1)

    def update(self, idx, val):
        idx += self.capacity
        self.tree[idx] = val
        while idx > 1:
            idx //= 2
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])

    def __setitem__(self, idx, val):
        self.update(idx, val)

    def __getitem__(self, idx):
        return self.tree[self.capacity + idx]
    
    def __len__(self):
        return len(self.tree)

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        sum = lambda x, y: x + y 
        super().__init__(capacity, operation=sum, neutral_value=0.0)

    def retrieve(self, upperbound):
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] > upperbound:
                idx = left
            else:
                upperbound -= self.tree[left]
                idx = left + 1
        return idx - self.capacity

class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, operation=min, neutral_value=INF)