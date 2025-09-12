import torch
import pandas as pd
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizer
from typing import Tuple
from collections import defaultdict


class InterventionDataset(Dataset):
    
    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, seq_length: int, concept_config = None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.df = df
        self.concepts = concept_config.input_concepts
        self.data = self._generate_data()
        for i, item in enumerate(self.data):
            base_ids, base_mask = self.tokenize(item["base_inputs"])
            source_ids, source_mask = self.tokenize(item["source_inputs"])
            item["base_input_ids"] = base_ids
            item["base_attention_mask"] = base_mask
            item["source_input_ids"] = source_ids
            item["source_attention_mask"] = source_mask

        self.labels = tokenizer.convert_tokens_to_ids(concept_config["label"].values)

    def _generate_data(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def tokenize(self, sentence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and left-pad/truncate to fixed length"""
        tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
        attention_mask = [1] * len(tokens)
        if len(tokens) > self.seq_length:
            tokens = tokens[-self.seq_length:]  # Truncate from the left
            attention_mask = attention_mask[-self.seq_length:]
        else:
            tokens = [self.tokenizer.eos_token_id] * (self.seq_length - len(tokens)) + tokens  # Left-pad with eos
            attention_mask = [0] * (self.seq_length - len(attention_mask)) + attention_mask
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return input_ids, attention_mask  # Return as tuple, not dict
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_sampler(self, batch_size, seed):
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

        return InterventionSampler(self, batch_size, seed)

    def collate_fn(self, batch):
        batch_out = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if isinstance(values[0], (int, float, torch.Tensor)):
                if isinstance(values[0], torch.Tensor):
                    batch_out[key] = torch.stack(values)
                else:
                    batch_out[key] = torch.tensor(values)
            else:
                batch_out[key] = values
        # Ensure all items in the batch have the same intervention_variables
        intervention_vars = batch_out["intervention_variables"]
        if any(v != intervention_vars[0] for v in intervention_vars):
            raise ValueError("All items in the batch must have the same intervention_variables.")
        batch_out["intervention_variables"] = batch_out["intervention_variables"][0]
        return batch_out