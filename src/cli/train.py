import torch
from transformers import AutoTokenizer

from src.tasks import init_task
from src.models import load_intervenable_model
from src.interventions import INTERVENTIONS
from src.utils.dataset import get_dataloader


class Trainer:

    def __init__(
        self, 
        model_name_or_path, 
        task_name, 
        intervention_name,
        hidden_size=None,
        num_variables=None,
        **model_kwargs
    ):
        self.task = init_task(task_name=task_name)
        self.model = load_intervenable_model(model_name_or_path, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        if hidden_size is None:
            hidden_size = self.model.config.hidden_size
        if num_variables is None:
            num_variables = self.task.num_variables
        
        intervention_class = INTERVENTIONS.get(intervention_name, None)
        if intervention_class is None:
            raise ValueError(f"Intervention {intervention_name} not found in available interventions: {list(INTERVENTIONS.keys())}")
        
        self.intervention_module = intervention_class(hidden_size, num_variables)
        self.intervention_module.to(self.model.device)

    def save(self, save_path):
        self.task.save_task_config(f"{save_path}/task_config.json")
        torch.save(self.intervention_module, f"{save_path}/intervention_module.pt")

    def prepare_dataloader(self, data_path, batch_size, seed, max_length, **kwargs):
        data = self.task.generate_data(data_path)
        train_data = data
        eval_data = None
        if isinstance(data, dict):
            train_data = data["train"]
            eval_data = data["validation"]
        train_dataloader = get_dataloader(
            train_data, batch_size, seed, self.tokenizer, max_length, **kwargs
        )
        if eval_data is not None:
            eval_dataloader = get_dataloader(
                eval_data, batch_size, seed, self.tokenizer, max_length, **kwargs
            )
        else:
            eval_dataloader = None
        return train_dataloader, eval_dataloader
    
    def train(self, data_path, output_dir, max_steps, learning_rate, batch_size, eval_steps, seed, max_length, **kwargs):
        train_dataloader, eval_dataloader = self.prepare_dataloader(
            data_path, batch_size, seed, max_length, **kwargs
        )
        # Training loop here
        # For each batch in train_dataloader:
        #   1. Get inputs and labels
        #   2. Forward pass through the model with interventions
        #   3. Compute loss
        #   4. Backward pass and optimizer step
        #   5. Every eval_steps, evaluate on eval_dataloader if provided
        pass  # Placeholder for training logic
