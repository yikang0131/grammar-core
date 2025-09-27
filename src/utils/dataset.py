import torch
from torch.utils.data import Sampler, DataLoader, Dataset

from collections import defaultdict


def tokenize_function(texts, tokenizer, max_length, **kwargs):
    chat = kwargs.pop("chat", False)
    if chat:
        texts = [
            tokenizer.apply_chat_template(
                t,
                add_generation_prompt=True,
                tokenize=False,
                **kwargs
            ) for t in texts
        ]
    inputs = tokenizer(texts, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    return inputs


def get_inputs(data, tokenizer, max_length, chat=True, **kwargs):
    base_inputs = tokenize_function(
        data["base_question"],
        tokenizer=tokenizer,
        max_length=max_length,
        chat=chat,
        **kwargs
    )
    source_inputs = tokenize_function(
        data["source_question"],
        tokenizer=tokenizer,
        max_length=max_length,
        chat=chat,
        **kwargs
    )
    outputs = ({"base_input_ids": base_inputs["input_ids"],
                "base_attention_mask": base_inputs["attention_mask"],
                "source_input_ids": source_inputs["input_ids"],
                "source_attention_mask": source_inputs["attention_mask"],
                "intervention_variables": data["intervention_variables"]})
    return outputs


def get_outputs(data, tokenizer, output_space, mutli_token=False):
    if mutli_token:
        raise NotImplementedError("Multi-token answers are not implemented yet.")
    else:
        base_answers = tokenizer.convert_tokens_to_ids(data["base_answer"])
        source_answers = tokenizer.convert_tokens_to_ids(data["source_answer"])
        base_labels = []
        for ans in base_answers:
            labels = [0.0] * len(output_space)
            labels[output_space.index(ans)] = 1.0
            base_labels.append(labels)
        source_labels = []
        for ans in source_answers:
            labels = [0.0] * len(output_space)
            labels[output_space.index(ans)] = 1.0
            source_labels.append(labels)
        base_labels = torch.tensor(base_labels, dtype=torch.float)
        source_labels = torch.tensor(source_labels, dtype=torch.float)
        outputs = {"base_labels": base_labels, 
                   "source_labels": source_labels}
    return outputs


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
            return len(self.data_source)
        
        def __iter__(self):
            g = torch.Generator()
            g.manual_seed(self.seed)
            shuffled_batches = self.batches.copy()
            shuffled_indices = torch.randperm(len(shuffled_batches), generator=g).tolist()
            for i in shuffled_indices:
                yield from shuffled_batches[i]

    return InterventionSampler(data, batch_size, seed)


def collate_fn(batch):
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


def get_dataloader(data, batch_size, seed, tokenizer, max_length, output_space, **kwargs):

    class InterventionDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]

    all_inputs = get_inputs(data, tokenizer, max_length, **kwargs)
    all_outputs = get_outputs(data, tokenizer, output_space)
    all_data = {**all_inputs, **all_outputs}
    # to list of dicts
    data = [dict(zip(all_data, t)) for t in zip(*all_data.values())]
    sampler = get_sampler(data, batch_size, seed)
    dataset = InterventionDataset(data)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
