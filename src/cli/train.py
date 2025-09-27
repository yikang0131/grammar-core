import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import argparse
import os
import json
from collections import defaultdict

# Add project root to sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models import load_intervenable_model
from src.tasks import init_task
from src.utils.dataset import get_dataloader
from src.utils.topk_scheduler import TopKScheduler
from src.interventions import INTERVENTIONS


class Trainer:

    def __init__(
        self, 
        model_name_or_path, 
        task_name, 
        intervention_name,
        hidden_size=None,
        num_variables=None,
        interv_at=None,
        **model_kwargs
    ):
        self.task = init_task(task_name=task_name)
        if interv_at:
            self.task.update_interv_configs(interv_at)

        self.model_name_or_path = model_name_or_path
        self.model = load_intervenable_model(model_name_or_path, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.output_space = self.tokenizer.convert_tokens_to_ids(self.task.output_space)
        
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
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.task.save_task_config(f"{save_path}/task_config.json")
        torch.save(self.intervention_module, f"{save_path}/intervention_module.pt")
        intervention_config = {
            "model_name_or_path": self.model_name_or_path,
            "intervenable_name": self.model.__class__.__name__,
            "intervention_name": self.intervention_module.__class__.__name__,
        }
        with open(f"{save_path}/intervention_config.json", "w") as f:
            json.dump(intervention_config, f, indent=4)

    def prepare_dataloader(self, data_path, batch_size, seed, max_length, **kwargs):
        data = self.task.generate_data(data_path)
        train_data = data
        eval_data = None
        if isinstance(data, dict):
            train_data = data["train"]
            eval_data = data["validation"]
        train_dataloader = get_dataloader(
            train_data, batch_size, seed, self.tokenizer, max_length, self.output_space, **kwargs
        )
        if eval_data is not None:
            eval_dataloader = get_dataloader(
                eval_data, batch_size, seed, self.tokenizer, max_length, self.output_space, **kwargs
            )
        else:
            eval_dataloader = None
        return train_dataloader, eval_dataloader
    
    def train(self, data_path, output_dir, max_steps, learning_rate, batch_size, eval_steps, seed, max_length, warmup_steps=0, **kwargs):
        writer = SummaryWriter(log_dir=f"{output_dir}/logs")
        train_dataloader, eval_dataloader = self.prepare_dataloader(
            data_path, batch_size, seed, max_length, **kwargs
        )
        
        optimizer = Adam(self.intervention_module.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
        loss_fn = CrossEntropyLoss()
        
        self.model.eval() # Freeze the intervenable model
        self.intervention_module.train()

        # Add TopK scheduler if needed
        if self.intervention_module.__class__.__name__ == "RotatedSpaceIntervention":
            topk_scheduler = TopKScheduler(
                initial_top_k=self.intervention_module.hidden_size,
                final_top_k=max(1, self.intervention_module.hidden_size // 10),
                total_steps=max_steps,
                warmup_steps=warmup_steps,
                schedule_type="linear"
            )
        else:
            topk_scheduler = None

        
        pbar = tqdm(total=max_steps, desc="Training")
        total_steps = 0
        
        while total_steps < max_steps:
            for batch in train_dataloader:
                if total_steps >= max_steps:
                    break

                kwargs = {}
                if topk_scheduler is not None:
                    current_top_k = topk_scheduler.get_top_k()
                    kwargs.update({"top_k": current_top_k})
                
                intervention_variables = [self.task.interv_configs[var] for var in batch["intervention_variables"]]
                intervened_outputs = self.model.run_intervention(
                    base_input_ids=batch["base_input_ids"],
                    base_attention_mask=batch["base_attention_mask"],
                    source_input_ids=batch["source_input_ids"],
                    source_attention_mask=batch["source_attention_mask"],
                    intervention_variables=intervention_variables,
                    intervention_module=self.intervention_module,
                    output_space=self.output_space,
                    labels=batch["source_labels"],
                    criterion=loss_fn,
                    **kwargs
                )
                
                loss = intervened_outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if topk_scheduler is not None:
                    topk_scheduler.step()
                    writer.add_scalar("TopK", topk_scheduler.get_top_k(), total_steps)
                
                writer.add_scalar("Loss/train", loss.item(), total_steps)
                writer.add_scalar("LR", scheduler.get_last_lr()[0], total_steps)
                
                if (total_steps + 1) % eval_steps == 0 and eval_dataloader is not None:
                    eval_loss, success_rates = self.evaluate(eval_dataloader, loss_fn, self.output_space, **kwargs)
                    writer.add_scalar("Loss/eval", eval_loss, total_steps)
                    for interv_vars, rate in success_rates.items():
                        writer.add_scalar(f"Accuracy/interv_{interv_vars}", rate, total_steps)
                    self.intervention_module.train()

                pbar.update(1)
                total_steps += 1

        pbar.close()
        self.save(output_dir)

    def evaluate(self, eval_dataloader, loss_fn, output_space_tokens, **kwargs):
        self.intervention_module.eval()
        total_loss = 0
        correct_predictions = defaultdict(int)
        total_predictions = defaultdict(int)
        pbar = tqdm(total=len(eval_dataloader), desc="Evaluating", position=1, leave=False)

        with torch.no_grad():
            for batch in eval_dataloader:

                intervention_variables = [self.task.interv_configs[var] for var in batch["intervention_variables"]]
                intervened_outputs = self.model.run_intervention(
                    base_input_ids=batch["base_input_ids"],
                    base_attention_mask=batch["base_attention_mask"],
                    source_input_ids=batch["source_input_ids"],
                    source_attention_mask=batch["source_attention_mask"],
                    intervention_variables=intervention_variables,
                    intervention_module=self.intervention_module,
                    output_space=output_space_tokens,
                    labels=batch["source_labels"],
                    criterion=loss_fn,
                    **kwargs
                )
                
                modified_logits = intervened_outputs.modified_logits
            
                loss = intervened_outputs.loss
                total_loss += loss.item()
                
                preds = torch.argmax(modified_logits, dim=-1)
                truths = batch["source_labels"].argmax(dim=-1).to(preds.device)

                interv_vars_tuple = tuple(batch["intervention_variables"])
                correct_predictions[interv_vars_tuple] += (preds == truths).sum().item()
                total_predictions[interv_vars_tuple] += truths.size(0)

                pbar.update(1)
        pbar.close()

        avg_loss = total_loss / len(eval_dataloader)
        success_rates = {k: v / total_predictions[k] for k, v in correct_predictions.items()}
        return avg_loss, success_rates

if __name__ == "__main__":
    class KVAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            if values:
                for value in values:
                    key, val = value.split("=", 1)
                    getattr(namespace, self.dest)[key] = val

    parser = argparse.ArgumentParser(description="Train an intervention module.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--model_kwargs", nargs='*', action=KVAction, help="Key-value pairs for model kwargs. E.g., device_map=auto")
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task to train on.")
    parser.add_argument("--intervention_name", type=str, required=True, help="Name of the intervention to use.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained intervention module and logs.")
    parser.add_argument("--interv_at", type=str, nargs='+', help="Module names to apply interventions to.")
    
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Total number of training steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Linear warmup over warmup_steps.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every eval_steps during training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    
    model_kwargs = args.model_kwargs or {}

    trainer = Trainer(
        model_name_or_path=args.model_name_or_path,
        task_name=args.task_name,
        intervention_name=args.intervention_name,
        interv_at=args.interv_at,
        **model_kwargs,
    )

    trainer.train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        seed=args.seed,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
    )
