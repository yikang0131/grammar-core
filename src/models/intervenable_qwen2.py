from transformers import Qwen2ForCausalLM
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache

import torch
from typing import Optional

from src.models.utils import IntervenableOutput
from src.models.intervenable_base import IntervenableBase


logger = logging.get_logger(__name__)


class IntervenableQwen2ForCausalLM(Qwen2ForCausalLM, IntervenableBase):
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
        **intervention_kwargs,
    ):
        """Forward pass with interventions applied at specified layers."""
        if intervention_position is None:
            intervention_position = slice(-1, None)
        
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
