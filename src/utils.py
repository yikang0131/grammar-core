import math
import torch
from src.api.model import DistributedAlignment
from src.api.model import IntervenableQwen2ForCausalLM
from src.api.tokenizer import BNCTokenizer


def load_model_tokenizer(model_path, **model_kwargs):
    model = IntervenableQwen2ForCausalLM.from_pretrained(model_path, **model_kwargs)
    tokenizer = BNCTokenizer.from_pretrained(f"{model_path}/bnc_word2c5.json")
    return model, tokenizer

def load_alignment(alignment_path, hidden_size, proj_num, device):
    alignment = DistributedAlignment(hidden_size=hidden_size, proj_num=proj_num)
    alignment.load_state_dict(torch.load(alignment_path))
    return alignment.to(device)

def get_concept_representations(
    sentence, 
    tokenizer, 
    model, 
    alignment,
    top_k,
    concept_config
):
    """Get interpretable concept representations for a given sentence"""
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    hidden_states = model.intervenable_forward(**inputs).hidden_states
    outputs = {}
    for idx, concept_module in concept_config.items():
        concept_reps = alignment.extract_interpretable_representations(
            hidden_states[concept_module][:, -1, :], idx, top_k, rotated_back=True
        )
        outputs[idx] = concept_reps.squeeze(0) # Remove batch dimension
    
    return outputs
