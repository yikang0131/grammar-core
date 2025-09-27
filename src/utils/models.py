import json
import torch
from transformers import AutoTokenizer

from src.tasks import load_task
from src.models import load_intervenable_model


def load_from_checkpoint(folder_path, **kwargs):
    with open(f"{folder_path}/intervention_config.json", "r") as f:
        intervention_config = json.load(f)
    
    model = load_intervenable_model(
        intervention_config["model_name_or_path"],
        **kwargs
    )
    intervention_module = torch.load(
        f"{folder_path}/intervention_module.pt",
        weights_only=False
    ).to(model.device)
    
    tokenizer = AutoTokenizer.from_pretrained(intervention_config["model_name_or_path"])

    task = load_task(f"{folder_path}/task_config.json")
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "intervention_module": intervention_module,
        "task": task
    }

