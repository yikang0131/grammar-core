from src.models.intervenable_qwen2 import IntervenableQwen2ForCausalLM
from src.models.intervenable_qwen3 import IntervenableQwen3ForCausalLM
from src.models.intervenable_gpt_neox import IntervenableGPTNeoXForCausalLM


AUTO_INTERVENABLE_MODELS = {
    "Qwen2ForCausalLM": IntervenableQwen2ForCausalLM,
    "Qwen3ForCausalLM": IntervenableQwen3ForCausalLM,
    "GPTNeoXForCausalLM": IntervenableGPTNeoXForCausalLM,
}


def load_intervenable_model(model_name_or_path: str, **kwargs):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path)
    pretrained_model_class = config.architectures[0]
    if pretrained_model_class in AUTO_INTERVENABLE_MODELS:
        model_class = AUTO_INTERVENABLE_MODELS[pretrained_model_class]
    else:
         raise ValueError(f"Model {pretrained_model_class} not supported for intervention.")
    return model_class.from_pretrained(model_name_or_path, **kwargs)
