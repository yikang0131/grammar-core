import torch
import pandas as pd
from src.utils.dataset import tokenize_function
from src.utils.models import load_from_checkpoint

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

loaded = load_from_checkpoint(
    "results/qwen3-8b_coreference_rotated-space",
    device_map="auto"
)

task = loaded["task"]
# data = task.generate_data("data/coreference.jsonl", sample=False, train_test_split=False)
data = pd.read_json("data/coreference.jsonl", lines=True)
examples = []
for i, row in data.iterrows():
    example = task.format_example(row.to_dict())
    examples.append(example)

# print(examples)
all_activations = {
    i["name"]: [] for i in task.interv_configs
}
model = loaded["model"]
intervention_module = loaded["intervention_module"]
model.eval()
intervention_module.eval()

top_k = intervention_module.hidden_size // 10
from tqdm import tqdm

for example in tqdm(examples):
    inputs = tokenize_function([example["question"]], loaded["tokenizer"], max_length=256, chat=True, enable_thinking=False)
    with torch.no_grad():
        hidden_states = model.intervenable_forward(
            **inputs, 
            intervened_modules=["model.layers[20]"]
        ).hidden_states["model.layers[20]"][:, -1, :]
        
    for i in range(task.num_variables):
        activations = intervention_module(hidden_states, hidden_states, [i], top_k=top_k, rotated_back=False)
        # Drop dim which is all zero
        activations = activations[0, (activations != 0).any(dim=0)]
        all_activations[task.id2var(i)].append(activations.detach().cpu().numpy())

# Save activations to pickle
import pickle
with open("coreference_qwen3-8b_rotated-space_activations.pkl", "wb") as f:
    pickle.dump(all_activations, f)

