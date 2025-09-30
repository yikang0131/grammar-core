import torch
import pandas as pd
from transformers import AutoTokenizer
from src.utils.dataset import tokenize_function, get_dataloader
from src.utils.models import load_from_checkpoint
from src.tasks import load_task

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"



task = load_task("results/qwen3-8b_coreference_rotated-space_256/task_config.json")
data = task.generate_data("data/coreference.jsonl", sample=True, train_test_split=False)
tokenizer = AutoTokenizer.from_pretrained("/data/ykliu/models/qwen3-8b")


dataloader = get_dataloader(
    data, 
    tokenizer=tokenizer, 
    batch_size=8, 
    seed=10, 
    max_length=256, 
    chat=task.chat, 
    output_space=tokenizer.convert_tokens_to_ids(task.output_space)
)

# cnt = 0
# for batch in dataloader:
#     # print(batch)
#     base_input_ids = batch["base_input_ids"]
#     base_sentences = tokenizer.batch_decode(base_input_ids, skip_special_tokens=True)
#     source_input_ids = batch["source_input_ids"]
#     source_sentences = tokenizer.batch_decode(source_input_ids, skip_special_tokens=True)
#     intervention_var = batch["intervention_variables"][0]
#     # print("Intervention variable:", task.id2var(intervention_var))
#     # for b, s in zip(base_sentences, source_sentences):
#     #     print("Base sentence:", b)
#     #     print("Source sentence:", s)
#     #     print()
#     print(batch["source_labels"])
#     print()
#     cnt += 1
#     if cnt == 100:
#         break


loaded = load_from_checkpoint(
    "results/qwen3-8b_coreference_rotated-space_256",
    device_map="auto"
)

task = loaded["task"]
# task = load_task("results/qwen3-8b_coreference_rotated-space_256/task_config.json")
# data = task.generate_data("data/coreference.jsonl", sample=False, train_test_split=False)
data = pd.read_json("data/coreference.jsonl", lines=True)
examples = []
for i, row in data.iterrows():
    example = task.format_example(row.to_dict())
    examples.append(example)

data = task.generate_data("data/coreference.jsonl", sample=False, train_test_split=False)
data.intervention_variables = data.intervention_variables.apply(lambda x: x[0])

# for interv_var, data_ in data.groupby("intervention_variables"):
#     print(interv_var, data_.base_question.nunique())

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
        activations = intervention_module(hidden_states, hidden_states, [i], top_k=top_k, rotated_back=True)
        # Drop dim which is all zero
        activations = activations[0, (activations != 0).any(dim=0)]
        all_activations[task.id2var(i)].append(activations.detach().cpu().numpy())

# Save activations to pickle
import pickle
with open("coreference_qwen3-8b_rotated-space_activations.pkl", "wb") as f:
    pickle.dump(all_activations, f)

