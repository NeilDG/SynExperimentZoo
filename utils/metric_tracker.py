import numpy as np
from collections import deque

class CircularBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def append(self, value):
        self.buffer.append(value)

    def get_data(self):
        return list(self.buffer)

    def __len__(self):
        return len(self.buffer)

class MetricTracker:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.metrics = {}

    def add_metric(self, key, value):
        if key not in self.metrics:
            self.metrics[key] = CircularBuffer(self.max_size)
        
        if isinstance(value, (int, float)):
            self.metrics[key].append(value)
        elif hasattr(value, 'item'): # Handle tensors
            self.metrics[key].append(value.item())
        else:
            self.metrics[key].append(value)

    def get_metric(self, key):
        if key in self.metrics:
            return self.metrics[key].get_data()
        return []

    def get_all_metrics(self):
        return {key: buffer.get_data() for key, buffer in self.metrics.items()}

    def clear(self):
        for key in self.metrics:
            self.metrics[key] = CircularBuffer(self.max_size)
