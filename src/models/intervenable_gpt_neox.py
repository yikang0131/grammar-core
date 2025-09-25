import torch
from transformers import GPTNeoXForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask

from transformers.utils import logging

from typing import Optional, Union

logger = logging.get_logger(__name__)


from src.models.intervenable_base import IntervenableBase
from src.models.utils import IntervenableOutput


class IntervenableGPTNeoXForCausalLM(GPTNeoXForCausalLM, IntervenableBase):
    """An intervenable version of GPTNeoXForCausalLM.
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
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, tuple[tuple[torch.FloatTensor]]]] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        intervention_position: slice = None,
        **intervention_kwargs,
    ):
        if intervention_position is None:
            intervention_position = slice(-1, None)

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        
        all_hidden_states = {}

        if inputs_embeds is None:
            inputs_embeds = self.gpt_neox.embed_in(input_ids)

        if "model.embed_tokens" in intervention_kwargs:
            inputs_embeds[:, intervention_position, :] = intervention_kwargs["model.embed_tokens"]
        
        all_hidden_states["model.embed_tokens"] = inputs_embeds

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = self.gpt_neox.emb_dropout(inputs_embeds)

        position_embeddings = self.gpt_neox.rotary_emb(hidden_states, position_ids)
        if "model.rotary_emb" in intervention_kwargs:
            position_embeddings[:, intervention_position, :] = intervention_kwargs["model.rotary_emb"]
        all_hidden_states["model.rotary_emb"] = position_embeddings

        for i, layer in enumerate(self.gpt_neox.layers):
            outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                head_mask=None,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = outputs[0]
            
            intervention_key = f"model.layers[{i}]"
            if intervention_key in intervention_kwargs:
                hidden_states[:, intervention_position, :] = intervention_kwargs[intervention_key]
            all_hidden_states[intervention_key] = hidden_states

        hidden_states = self.gpt_neox.final_layer_norm(hidden_states)
        if "model.norm" in intervention_kwargs:
            hidden_states[:, intervention_position, :] = intervention_kwargs["model.norm"]
        all_hidden_states["model.norm"] = hidden_states

        logits = self.embed_out(hidden_states)
        if "lm_head" in intervention_kwargs:
            logits[:, intervention_position, :] = intervention_kwargs["lm_head"]
        all_hidden_states["lm_head"] = logits

        return IntervenableOutput(logits=logits, hidden_states=all_hidden_states)