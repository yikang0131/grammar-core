import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from transformers.cache_utils import Cache, DynamicCache


from dataclasses import dataclass
from typing import Optional, Dict
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class IntervenableOutput:
    logits: torch.FloatTensor = None
    hidden_states: Dict[str, torch.FloatTensor] = None


class IntervenableQwen2ForCausalLM(Qwen2ForCausalLM):
    """An intervenable version of Qwen2ForCausalLM.
    intervention_kwargs:
    {
        "model.embed_tokens": torch.FloatTensor,
        "model.layers[0]": torch.FloatTensor,
        "model.layers[1]": torch.FloatTensor,
        ...
        "model.norm": torch.FloatTensor,
        "lm_head": torch.FloatTensor,
    }
    """

    def __init__(self, config):
        super().__init__(config)
        # disable gradient for the original model parameters
        for param in self.parameters():
            param.requires_grad = False

    def intervenable_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        intervention_position: int = None,
        **intervention_kwargs,
    ):
        """Forward pass with interventions applied at specified layers."""
        if intervention_position is None:
            intervention_position = -1
        
        use_cache = use_cache if use_cache is not None else self.model.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.model.gradient_checkpointing and self.model.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        all_hidden_states = {}

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # Apply intervention to embeddings if specified
        if "model.embed_tokens" in intervention_kwargs:
            inputs_embeds[:, intervention_position, :] = intervention_kwargs["model.embed_tokens"]
            # inputs_embeds = intervention_kwargs["model.embed_tokens"]

        all_hidden_states["model.embed_tokens"] = inputs_embeds

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self.model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values
        )

        hidden_states = inputs_embeds

        # Create position embeddings
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        
        # Apply intervention to position embeddings if specified
        if "model.rotary_emb" in intervention_kwargs:
            position_embeddings[:, intervention_position, :] = intervention_kwargs["model.rotary_emb"]
            # position_embeddings = intervention_kwargs["model.rotary_emb"]

        all_hidden_states["model.rotary_emb"] = position_embeddings

        # Process through decoder layers
        for i, decoder_layer in enumerate(self.model.layers[: self.model.config.num_hidden_layers]):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]
            
            # Apply intervention to this layer's output if specified
            intervention_key = f"model.layers[{i}]"
            if intervention_key in intervention_kwargs:
                hidden_states[:, intervention_position, :] = intervention_kwargs[intervention_key]
                # hidden_states = intervention_kwargs[intervention_key]
            all_hidden_states[f"model.layers[{i}]"] = hidden_states

        # Apply layer norm
        hidden_states = self.model.norm(hidden_states)
        
        # Apply intervention to normalized hidden states if specified
        if "model.norm" in intervention_kwargs:
            hidden_states[:, intervention_position, :] = intervention_kwargs["model.norm"]
            # hidden_states = intervention_kwargs["model.norm"]

        all_hidden_states["model.norm"] = hidden_states

        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Apply intervention to logits if specified
        if "lm_head" in intervention_kwargs:
            logits[:, intervention_position, :] = intervention_kwargs["lm_head"]
            # logits = intervention_kwargs["lm_head"]

        all_hidden_states["lm_head"] = logits

        return IntervenableOutput(logits=logits, hidden_states=all_hidden_states)


class DistributedAlignment(nn.Module):
    
    def __init__(self, hidden_size, proj_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_num = proj_num
        
        rotation_matrix = torch.empty(hidden_size, hidden_size)
        nn.init.orthogonal_(rotation_matrix)
        self.rotation_matrix = nn.Parameter(rotation_matrix, requires_grad=True)
        
        var_proj = torch.empty(proj_num, hidden_size)
        nn.init.kaiming_uniform_(var_proj)
        self.var_proj = nn.Parameter(var_proj, requires_grad=True)

    def get_projection_weights(self):
        return torch.sigmoid(self.var_proj)
        
    def extract_interpretable_representations(self, hidden_states, intervention_variable, top_k, rotated_back=False):
        """Extract interpretable representations by projecting to the learned basis"""
        rotated_states = torch.matmul(hidden_states, self.rotation_matrix)
        projection_weights = self.get_projection_weights()
        
        # Select top_k dimensions for this variable
        top_indices = self.select_subspace(top_k, intervention_variable)
        
        # Create a mask for the selected dimensions
        mask = torch.zeros_like(projection_weights[intervention_variable])
        mask[top_indices] = 1.0
        
        output = rotated_states * (projection_weights[intervention_variable] * mask)

        if rotated_back:
            output = torch.matmul(output, self.rotation_matrix.T)

        return output

    def select_subspace(self, top_k, var_idx):
        """Select top_k dimensions based on projection weights for variable var_idx"""
        _, top_indices = torch.topk(self.var_proj[var_idx], top_k)
        return top_indices
    
    def forward(self, base, source, intervention_variables, top_k):
        """
        Samples in a batch should have the same intervention positions
        
        Args:
            base: base hidden states [batch, seq, hidden_size]
            source: source hidden states [batch, seq, hidden_size] 
            intervention_variables: list of variable indices to intervene on
            top_k: number of top dimensions to select for intervention
        """
        # Rotate to learned basis
        rotated_base = torch.matmul(base, self.rotation_matrix)
        rotated_source = torch.matmul(source, self.rotation_matrix)
        
        # Start with base representation
        # rotated_base_intervened = rotated_base.clone()
        rotated_base_intervened = torch.zeros_like(rotated_base)
        
        # Apply interventions only on selected top_k dimensions for each variable
        for var_idx in intervention_variables:
            # Select top_k dimensions for this variable
            top_indices = self.select_subspace(top_k, var_idx)
            
            # Create intervention mask - only intervene on top_k dimensions
            intervention_mask = torch.zeros(self.hidden_size, device=base.device)
            intervention_mask[top_indices] = 1.0
            
            # Apply intervention: replace base with source only for selected dimensions
            rotated_base_intervened = (
                rotated_base_intervened * (1 - intervention_mask) +  # Keep non-selected dims from base
                rotated_source * intervention_mask  # Replace selected dims with source
            )
        
        # Rotate back to original basis
        return torch.matmul(rotated_base_intervened, self.rotation_matrix.T)




class TopKScheduler:
    """
    Scheduler for dynamically adjusting top_k parameter during training.
    Supports various scheduling strategies: linear, exponential, cosine, step-wise.
    """
    
    def __init__(
        self,
        initial_top_k: int,
        final_top_k: int,
        total_steps: int,
        schedule_type: str = "linear",
        warmup_steps: int = 0,
        step_size: int = None,
        gamma: float = 0.1
    ):
        """
        Args:
            initial_top_k: Starting value of top_k
            final_top_k: Final value of top_k
            total_steps: Total training steps
            schedule_type: "linear", "exponential", "cosine", "step"
            warmup_steps: Number of steps to keep initial_top_k (warmup period)
            step_size: For step scheduler, steps between reductions
            gamma: For exponential/step scheduler, decay factor
        """
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.step_size = step_size or (total_steps // 4)  # Default to 4 reductions
        self.gamma = gamma
        
        self.current_step = 0
        self.current_top_k = initial_top_k
        
        # Validate inputs
        assert initial_top_k >= final_top_k, "initial_top_k must be >= final_top_k"
        assert schedule_type in ["linear", "exponential", "cosine", "step"], \
            "schedule_type must be one of: linear, exponential, cosine, step"
    
    def step(self):
        """Update the current step and recalculate top_k"""
        self.current_step += 1
        self.current_top_k = self.get_top_k()
        return self.current_top_k
    
    def get_top_k(self) -> int:
        """Calculate current top_k based on schedule"""
        # Warmup period - keep initial value
        if self.current_step <= self.warmup_steps:
            return self.initial_top_k
        
        # Adjust step for warmup
        adjusted_step = self.current_step - self.warmup_steps
        adjusted_total = self.total_steps - self.warmup_steps
        
        if adjusted_step >= adjusted_total:
            return self.final_top_k
        
        # Calculate progress ratio
        progress = adjusted_step / adjusted_total
        
        if self.schedule_type == "linear":
            top_k = self.initial_top_k - (self.initial_top_k - self.final_top_k) * progress
            
        elif self.schedule_type == "exponential":
            # Exponential decay: top_k = initial * (final/initial)^progress
            decay_factor = (self.final_top_k / self.initial_top_k) ** progress
            top_k = self.initial_top_k * decay_factor
            
        elif self.schedule_type == "cosine":
            # Cosine annealing
            top_k = self.final_top_k + (self.initial_top_k - self.final_top_k) * \
                     (1 + math.cos(math.pi * progress)) / 2
                     
        elif self.schedule_type == "step":
            # Step-wise reduction
            num_reductions = adjusted_step // self.step_size
            top_k = self.initial_top_k * (self.gamma ** num_reductions)
            top_k = max(top_k, self.final_top_k)  # Don't go below final_top_k
        
        return max(int(round(top_k)), self.final_top_k)
    
    def state_dict(self):
        """Return scheduler state for checkpointing"""
        return {
            'current_step': self.current_step,
            'current_top_k': self.current_top_k
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint"""
        self.current_step = state_dict['current_step']
        self.current_top_k = state_dict['current_top_k']