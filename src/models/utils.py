import torch
from typing import Dict
from dataclasses import dataclass


@dataclass
class IntervenableOutput:
    logits: torch.FloatTensor = None
    hidden_states: Dict[str, torch.FloatTensor] = None


@dataclass
class IntervenedOutput:
    loss: torch.FloatTensor = None
    original_logits: torch.FloatTensor = None
    modified_logits: torch.FloatTensor = None
    original_hidden_states: Dict[str, torch.FloatTensor] = None
    modified_hidden_states: Dict[str, torch.FloatTensor] = None