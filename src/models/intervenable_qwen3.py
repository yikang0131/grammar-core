from transformers import Qwen3ForCausalLM
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

import torch
from typing import Optional, List

from src.models.utils import IntervenableOutput
from src.models.intervenable_base import IntervenableBase


logger = logging.get_logger(__name__)


class IntervenableQwen3ForCausalLM(Qwen3ForCausalLM, IntervenableBase):
    """An intervenable version of Qwen3ForCausalLM.
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

    def intervenable_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        intervention_position: slice = None,
        intervened_modules: Optional[List[str]] = None,
        **intervention_kwargs,
    ):
        """Forward pass with interventions applied at specified layers."""
        def to_collect(module_name):
            return (module_name in intervention_kwargs) or (module_name in intervened_modules)

        if intervened_modules is None:
            intervened_modules = []

        all_hidden_states = {}
        if intervention_position is None:
            intervention_position = slice(-1, None)
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # Apply intervention to embeddings if specified
        if "model.embed_tokens" in intervention_kwargs:
            inputs_embeds[:, intervention_position, :] = intervention_kwargs["model.embed_tokens"]
            # inputs_embeds = intervention_kwargs["model.embed_tokens"]

        if to_collect("model.embed_tokens"):
            all_hidden_states["model.embed_tokens"] = inputs_embeds

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.model.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            
            # Apply intervention to this layer's output if specified
            intervention_key = f"model.layers[{i}]"
            if intervention_key in intervention_kwargs:
                hidden_states[:, intervention_position, :] = intervention_kwargs[intervention_key]
                # hidden_states = intervention_kwargs[intervention_key]

            if to_collect(intervention_key):
                all_hidden_states[f"model.layers[{i}]"] = hidden_states

        hidden_states = self.model.norm(hidden_states)
        
        # Apply intervention to normalized hidden states if specified
        if "model.norm" in intervention_kwargs:
            hidden_states[:, intervention_position, :] = intervention_kwargs["model.norm"]
            # hidden_states = intervention_kwargs["model.norm"]

        if to_collect("model.norm"):
            all_hidden_states["model.norm"] = hidden_states

        # Get logits
        logits = self.lm_head(hidden_states)

        return IntervenableOutput(logits=logits, hidden_states=all_hidden_states)