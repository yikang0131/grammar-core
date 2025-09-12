import pandas as pd
from src.api import (
    InterventionDataset, 
    ConceptConfig, 
    BNCTokenizer, 
    DASTrainer,
    DistributedAlignment,
    IntervenableQwen2ForCausalLM,
    TopKScheduler
)


concept_config = ConceptConfig.from_dict(concept_dict = {
    "target_num": {"id": 0, "values": ["singular", "plural"], "at": "model.layers[4]"},
    "distractor_num": {"id": 1, "values": ["singular", "plural"], "at": "model.layers[4]"},
    "agree_with": {"id": 2, "values": ["target", "distractor"], "at": "model.layers[5]"},
    "label": {"id": 3, "values": ["is", "are"], "at": ""}
})


class SVDataset(InterventionDataset):
    def __init__(self, df, tokenizer, seq_len=16, concept_config=concept_config):
        super().__init__(df, tokenizer, seq_len, concept_config)

    def _generate_data(self):

        def solve_sv_agreement(target_num, distractor_num, agree_with):
            assert agree_with in ["target", "distractor"]
            if agree_with == "target":
                return target_num
            elif agree_with == "distractor":
                return distractor_num
            
        id2label = {0: self.tokenizer.convert_tokens_to_ids(["is"])[0], 
                   1: self.tokenizer.convert_tokens_to_ids(["are"])[0]}

        data = []

        for i, row in self.df.iterrows():
            for k1 in ["sentence00", "sentence01", "sentence10", "sentence11"]:
                for k2 in ["sentence00", "sentence01", "sentence10", "sentence11"]:
                    # if k1 == k2:
                    #     continue
                    base_sentence = row[k1]
                    source_sentence = row[k2]
                    base_target_num, base_distractor_num = [int(i) for i in k1[-2:]]
                    source_target_num, source_distractor_num = [int(i) for i in k2[-2:]]
                    base_verb_num = solve_sv_agreement(base_target_num, base_distractor_num, "target")

                    intervention_concept = []

                    def get_expected_verb_num(intervention_concept):
                        target_num, distractor_num = base_target_num, base_distractor_num
                        agree_with = "target"
                        if "target_num" in intervention_concept:
                            target_num = source_target_num
                        if "distractor_num" in intervention_concept:
                            distractor_num = source_distractor_num
                        if "agree_with" in intervention_concept:
                            agree_with = "distractor"
                        return solve_sv_agreement(target_num, distractor_num, agree_with)

                    if base_target_num != source_target_num:
                        intervention_concept.append("target_num")
                        expected_verb_num = get_expected_verb_num(intervention_concept)
                        data.append({
                            "base_inputs": base_sentence,
                            "source_inputs": source_sentence,
                            "intervention_variables": [self.concepts[c].id for c in intervention_concept],
                            "base_labels": id2label[base_verb_num],
                            "targets": id2label[expected_verb_num],
                        })
                    if base_distractor_num != source_distractor_num:
                        intervention_concept.append("distractor_num")
                        expected_verb_num = get_expected_verb_num(intervention_concept)
                        data.append({
                            "base_inputs": base_sentence,
                            "source_inputs": source_sentence,
                            "intervention_variables": [self.concepts[c].id for c in intervention_concept],
                            "base_labels": id2label[base_verb_num],
                            "targets": id2label[expected_verb_num],
                        })

                    # expected_verb_num = solve_sv_agreement(source_target_num, source_distractor_num, "target")

                    # data.append({
                    #     "base_inputs": base_sentence,
                    #     "source_inputs": source_sentence,
                    #     "intervention_variables": [self.concepts[c].id for c in intervention_concept],
                    #     "base_labels": id2label[base_verb_num],
                    #     "targets": id2label[expected_verb_num],
                    # })

                    intervention_concept.append("agree_with")
                    # expected_verb_num = solve_sv_agreement(source_target_num, source_distractor_num, "distractor")
                    expected_verb_num = get_expected_verb_num(intervention_concept)
                    data.append({
                        "base_inputs": base_sentence,
                        "source_inputs": source_sentence,
                        "intervention_variables": [self.concepts[c].id for c in intervention_concept],
                        "base_labels": id2label[base_verb_num],
                        "targets": id2label[expected_verb_num],
                    })

        return data


train_df = pd.read_json("data/svagree/solved.train.jsonl", lines=True)
eval_df = pd.read_json("data/svagree/solved.dev.jsonl", lines=True)
tokenizer = BNCTokenizer.from_pretrained("results/checkpoint-97790/bnc_word2c5.json")
train_dataset = SVDataset(train_df, tokenizer, seq_len=16, concept_config=concept_config)
eval_dataset = SVDataset(eval_df, tokenizer, seq_len=16, concept_config=concept_config)


model = IntervenableQwen2ForCausalLM.from_pretrained("results/checkpoint-97790").to("cuda:0")
das = DistributedAlignment(model.config.hidden_size, 3)


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
    eval_dataset,
    8,
    20,
    "cuda:0",
    learning_rate=1e-3,
    # output_dir="svagree_das",
    output_dir="svagree_das2",
    top_k_scheduler=top_k_scheduler,
)


trainer.train(max_steps=max_steps, eval_steps=100, logging_steps=10)