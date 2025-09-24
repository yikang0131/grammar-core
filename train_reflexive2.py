import random
import pandas as pd
from src.api import (
    InterventionDataset, 
    ConceptConfig, 
    DASTrainer,
    DistributedAlignment,
    IntervenableQwen3ForCausalLM,
    TopKScheduler
)
from transformers import AutoTokenizer

concept_config = ConceptConfig.from_dict(concept_dict = {
    "syn_s": {"id": 0, "values": ["male", "female"], "at": "model.layers[20]"},
    "syn_o": {"id": 1, "values": ["male", "female"], "at": "model.layers[20]"},
    "prag_svs": {"id": 2, "values": ["reflexive", "non-reflexive"], "at": "model.layers[25]"},
    "prag_ovs": {"id": 3, "values": ["reflexive", "non-reflexive"], "at": "model.layers[25]"},
    "label": {"id": 4, "values": ["A", "B"], "at": ""}
})


class ReflexiveDataset(InterventionDataset):

    def __init__(self, df, tokenizer, seq_len, concept_config=None):
        super().__init__(df, tokenizer, seq_len, concept_config)

    def _generate_data(self):
        
        import re
        def format_prompt(row, key, choices=None):
            sentence = row[key]
            # Replace the last "himself/herself/her/him" in the sentence with a blank
            answer = re.findall(r'\b(himself|herself|him|her)\b', sentence)[-1]
            sentence = re.sub(r'\b(himself|herself|him|her)\b(?!.*\b(himself|herself|him|her)\b)', '_____', sentence)
            
            if choices is None:
                choices = ["him", "her", "himself", "herself"]
                random.shuffle(["him", "her", "himself", "herself"])

            prompt_template = f"Read the following sentence and answer the quesetion.\n\nSentence: {sentence}\n\nQuestion: What should be filled in the blank?\n\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nReply the answer with A, B, C or D only."
            prompt = prompt_template.format(sentence=sentence)
            prompt = self.tokenizer.apply_chat_template([
                {"role": "user", "content": prompt}
            ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            label = ["A", "B", "C", "D"][choices.index(answer)]
            return prompt, label

        data = []
        cols = self.df.columns.tolist()[1:6]

        for col in cols:
            self.df = self.df[~self.df[f"{col}_obj"].isin(["him", "her"])]

        for i, row in self.df.iterrows():
            choices = ["him", "her", "himself", "herself"]
            random.shuffle(["him", "her", "himself", "herself"])
            base_prompt, base_label = format_prompt(row, "baseline", choices)
            for key in cols:
                prompt, label = format_prompt(row, key, choices)
                # prompt, label = format_prompt(row, key)
                if key == "syn_so":
                    intervention_variables = [concept_config["syn_s"].id, concept_config["syn_o"].id]
                else:
                    intervention_variables = [concept_config[key].id]
                data.append({
                    "base_inputs": base_prompt,
                    "source_inputs": prompt,
                    "intervention_variables": intervention_variables,
                    "base_labels": self.tokenizer.convert_tokens_to_ids(base_label),
                    "targets": self.tokenizer.convert_tokens_to_ids(label),
                })
                data.append({
                    "base_inputs": prompt,
                    "source_inputs": base_prompt,
                    "intervention_variables": intervention_variables,
                    "base_labels": self.tokenizer.convert_tokens_to_ids(label),
                    "targets": self.tokenizer.convert_tokens_to_ids(base_label),
                })

        return data


df = pd.read_json("data/reflexive.jsonl", lines=True)
train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)


# tokenizer = AutoTokenizer.from_pretrained("models/pythia-160m")
tokenizer = AutoTokenizer.from_pretrained("/mnt/models/qwen3-8b")
train_dataset = ReflexiveDataset(train_df, tokenizer, seq_len=128, concept_config=concept_config)
eval_dataset = ReflexiveDataset(eval_df, tokenizer, seq_len=128, concept_config=concept_config)
model = IntervenableQwen3ForCausalLM.from_pretrained("/mnt/models/qwen3-8b", device_map="auto")

from tqdm import tqdm
choices = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D"])
acc = 0
total = 0

data = train_dataset.data

# drop duplicates in data
sents = []
unique_data = []
for d in data:
    identifier = d["base_inputs"]
    if identifier not in sents:
        sents.append(identifier)
        unique_data.append(d)

for d in tqdm(unique_data):
    inputs = tokenizer(d["base_inputs"], return_tensors="pt").to(model.device)
    label = d["base_labels"]
    
    outputs = model(**inputs)
    
    pred_logits = outputs.logits[0, -1, choices].detach()  # Get logits for A,B,C,D only
    pred_idx = pred_logits.argmax().item()  # Index in [0,1,2,3]
    pred_token_id = choices[pred_idx]  # Actual token ID
    
    # Fix the comparison logic
    if pred_token_id == label:  # Compare token IDs directly
        acc += 1
    else:
        print("Prompt:", d["base_inputs"])
        print("Ground Truth:", tokenizer.convert_ids_to_tokens([label])[0])
        print("Prediction:", tokenizer.convert_ids_to_tokens([pred_token_id])[0])
        print("Pred logits:", pred_logits.tolist())  # Debug: see the actual logits
        print("==============================")
    
    total += 1

print("Accuracy:", acc / total)