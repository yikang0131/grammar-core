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

        def format_prompt(row, key, answer_index=None):
            sentence = row[key]
            refl = row[f"{key}_refl"]
            subj = row[f"{key}_subj"]
            obj = row[f"{key}_obj"]
            if refl in ["himself", "herself"]:
                answer = obj
                non_answer = subj
            else:
                answer = subj
                non_answer = obj
            if answer_index is not None:
                answer_idx = random.choice([0, 1])
            if answer_idx == 0:
                A, B = answer, non_answer
            else:
                A, B = non_answer, answer
            prompt_template = f"Read the following sentence and answer the quesetion.\n\nSentence: {sentence}\n\nQuestion: \"{refl}\" refers to:\n\nA. {A}\nB. {B}\n\nAnswer: "
            return prompt_template.format(sentence=sentence, A=A, B=B, refl=refl), ["A", "B"][answer_idx]

        data = []
        cols = self.df.columns.tolist()[1:6]

        for col in cols:
            self.df = self.df[~self.df[f"{col}_obj"].isin(["him", "her"])]


        for i, row in self.df.iterrows():
            answer_idx = random.choice([0, 1])
            base_prompt, base_label = format_prompt(row, "baseline", answer_index=answer_idx)
            for key in cols:
                if "syn" in key:
                    prompt, label = format_prompt(row, key, answer_index=answer_idx)
                elif key == "prag_svs":
                    prompt, label = format_prompt(row, key, answer_index=answer_idx)
                elif key == "prag_ovs":
                    prompt, label = format_prompt(row, key, answer_index=1 - answer_idx)
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

        return data


df = pd.read_json("data/reflexive.jsonl", lines=True)
train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)


# tokenizer = AutoTokenizer.from_pretrained("models/pythia-160m")
tokenizer = AutoTokenizer.from_pretrained("models/qwen3-4b")
train_dataset = ReflexiveDataset(train_df, tokenizer, seq_len=128, concept_config=concept_config)
eval_dataset = ReflexiveDataset(eval_df, tokenizer, seq_len=16, concept_config=concept_config)


model = IntervenableQwen3ForCausalLM.from_pretrained("models/qwen3-4b").to("cuda:0")
das = DistributedAlignment(model.config.hidden_size, 4)


max_steps = 1000
top_k_config = {
    "initial_top_k": model.config.hidden_size,
    "final_top_k": 200,
    "total_steps": max_steps,
    "schedule_type": "cosine",  # or "linear", "exponential", "step"
    "warmup_steps": 100,  # Keep initial value for first 100 steps
}

top_k_scheduler = TopKScheduler(**top_k_config)

trainer = DASTrainer(
    model,
    das,
    train_dataset,
    eval_dataset=eval_dataset,
    batch_size=8,
    seed=20,
    device="cuda:0",
    learning_rate=1e-3,
    # output_dir="svagree_das",
    output_dir="results/reflexive_das",
    top_k_scheduler=top_k_scheduler,
)


trainer.train(max_steps=max_steps, eval_steps=100, logging_steps=10)