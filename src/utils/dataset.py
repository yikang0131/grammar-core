from collections import defaultdict
from torch.utils.data import Sampler
import torch


def get_sampler(data, batch_size, seed):
    class InterventionSampler(Sampler):
        def __init__(self, data_source, batch_size, seed):
            self.data_source = data_source
            self.batch_size = batch_size
            self.seed = seed
            self.grouped_indices = self._group_by_intervention()
            self.batches = self._create_batches()
        
        def _group_by_intervention(self):
            grouped = defaultdict(list)
            for idx, item in enumerate(self.data_source):
                key = tuple(item["intervention_variables"])  # Convert list to tuple for hashing
                grouped[key].append(idx)
            return grouped
        
        def _create_batches(self):
            all_batches = []
            for indices in self.grouped_indices.values():
                for i in range(0, len(indices), self.batch_size):
                    batch = indices[i:i + self.batch_size]
                    if len(batch) == self.batch_size:
                        all_batches.append(batch)
            return all_batches
        
        def __len__(self):
            return len(self.batches)
        
        def __iter__(self):
            g = torch.Generator()
            g.manual_seed(self.seed)
            shuffled_batches = self.batches.copy()
            shuffled_indices = torch.randperm(len(shuffled_batches), generator=g).tolist()
            for i in shuffled_indices:
                yield from shuffled_batches[i]

    return InterventionSampler(data, batch_size, seed)