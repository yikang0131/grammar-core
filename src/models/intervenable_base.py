from src.models.utils import IntervenedOutput


class IntervenableBase:

    def __init__(self, **kwargs):
        for param in self.parameters():
            param.requires_grad = False

    def intervenable_forward(self, **intervention_kwargs):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def run_intervention(
        self,
        base_input_ids,
        base_attention_mask,
        source_input_ids,
        source_attention_mask,
        intervention_module,
        intervention_variables,
        intervention_position=None,
        **kwargs
    ):
        if intervention_position is None:
            intervention_position = slice(-1, None)

        base_inputs = {
            "input_ids": base_input_ids.to(self.device),
            "attention_mask": base_attention_mask.to(self.device),
        }
        source_inputs = {
            "input_ids": source_input_ids.to(self.device),
            "attention_mask": source_attention_mask.to(self.device),
        }

        interv_map = {}
        for interv_id, interv_config in enumerate(intervention_variables):
            interv_at = interv_config["interv_at"]
            if interv_at not in interv_map:
                interv_map[interv_at] = []
            interv_map[interv_at].append(interv_id)
        
        original_outputs = self.intervenable_forward(
            intervention_position=intervention_position,
            **base_inputs
        )
        original_hidden_states = original_outputs.hidden_states
        original_logits = original_outputs.logits

        source_hidden_states = self.intervenable_forward(
            intervention_position=intervention_position,
            **source_inputs
        ).hidden_states

        base_hidden_states = {k: v.clone() for k, v in original_hidden_states.items()}
        
        for interv_at, vars in interv_map.items():
            intervention_kwargs = {}
            base_activations = base_hidden_states[interv_at][:, intervention_position, :]
            source_activations = source_hidden_states[interv_at][:, intervention_position, :]
            modified_activations = intervention_module(base_activations, source_activations, vars, **kwargs)

            intervention_kwargs[interv_at] = modified_activations

            outputs = self.model.intervenable_forward(
                intervention_position=intervention_position,
                **base_inputs,
                **intervention_kwargs
            )
            base_hidden_states = outputs.hidden_states
            modified_logits = outputs.logits

        return IntervenedOutput(
            original_logits=original_logits,
            modified_logits=modified_logits,
            original_hidden_states=original_hidden_states,
            modified_hidden_states=base_hidden_states
        )