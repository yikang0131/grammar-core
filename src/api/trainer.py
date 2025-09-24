import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import math
import logging
from tqdm import tqdm
from typing import Dict, Tuple
import os


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DASTrainer:
    def __init__(
        self,
        model,
        alignment,
        train_dataset,
        eval_dataset,
        batch_size,
        seed,
        device,
        learning_rate,
        output_dir,
        top_k_scheduler
    ):
        
        self.device = device
        self.concept_config = train_dataset.concepts
        self.labels = train_dataset.labels
        self.label2id = {v: i for i, v in enumerate(self.labels)}
        self.id2label = {i: v for i, v in enumerate(self.labels)}

        # Load frozen base model
        self.model = model.to(device)
        self.alignment = alignment.to(device)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_dataset.get_sampler(batch_size, seed),
            collate_fn=train_dataset.collate_fn
        )
        self.train_dataloader = train_dataloader

        if eval_dataset is not None:
            eval_dataloader = DataLoader(
                eval_dataset, 
                batch_size=batch_size, 
                sampler=eval_dataset.get_sampler(batch_size, seed),
                collate_fn=eval_dataset.collate_fn
            )
            self.eval_dataloader = eval_dataloader
        else:
            self.eval_dataloader = None

        self.output_dir = output_dir
        self.top_k_scheduler = top_k_scheduler
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))

        # Optimizer (only train DAS parameters)
        self.optimizer = optim.Adam(self.alignment.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.labels = train_dataset.labels
        
        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f" Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)} (frozen)")
        logger.info(f" Alignment parameters: {sum(p.numel() for p in alignment.parameters() if p.requires_grad)} (trainable)")
        logger.info(f" TensorBoard logs will be saved to: {os.path.join(output_dir, 'tensorboard')}")
        
    def compute_intervention_loss(
            self,
            base_input_ids,
            base_attention_mask,
            source_input_ids,
            source_attention_mask,
            intervention_variables,
            targets,
            top_k,
            intervention_position=None,
            **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss for a batch of intervention examples with same intervention_positions"""

        if intervention_position is None:
            intervention_position = -1

        base_inputs = {
            "input_ids": base_input_ids.to(self.device),
            "attention_mask": base_attention_mask.to(self.device)
        }
        source_inputs = {
            "input_ids": source_input_ids.to(self.device),
            "attention_mask": source_attention_mask.to(self.device)
        }
        targets = torch.tensor([
            [0, 1] if self.label2id[t.item()] else [1, 0] for t in targets
        ], dtype=torch.float).to(self.device)  # [batch_size, num_labels]
            
        interv_map = {}
        for interv_id in intervention_variables:
            interv_at = self.concept_config[interv_id].at
            if interv_at not in interv_map:
                interv_map[interv_at] = []
            interv_map[interv_at].append(interv_id)

        base_hidden_states = self.model.intervenable_forward(**base_inputs).hidden_states
        source_hidden_states = self.model.intervenable_forward(**source_inputs).hidden_states

        for interv_at, vars in interv_map.items():
            intervention_kwargs = {}
            base_repr = base_hidden_states[interv_at][:, intervention_position, :]
            source_repr = source_hidden_states[interv_at][:, intervention_position, :]
            modified_repr = self.alignment(base_repr, source_repr, vars, top_k)

            intervention_kwargs[interv_at] = modified_repr

            outputs = self.model.intervenable_forward(
                **base_inputs,
                **intervention_kwargs
            )
            base_hidden_states = outputs.hidden_states
            modified_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

        label_logits = modified_logits[:, self.labels]  # [batch_size, num_labels]
        
        # Compute loss
        loss = self.criterion(label_logits, targets)
        # Compute predictions and accuracy
        model_predictions = torch.argmax(label_logits, dim=1)  # [batch_size]
        targets = torch.argmax(targets, dim=1)  # [batch_size]
        accuracy = (model_predictions == targets).float().mean().item()
        
        return modified_logits, loss, accuracy

    def save_alignment(self, step: int = None):
        """Save the alignment model state"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        if step is not None:
            save_path = os.path.join(self.output_dir, f"alignment_step_{step}.pt")
        else:
            save_path = os.path.join(self.output_dir, "alignment_final.pt")
        
        torch.save(self.alignment.state_dict(), save_path)
        
        logger.info(f"Alignment model saved to: {save_path}")
        return save_path

    def train(
            self, 
            max_steps: int = None, 
            epochs: int = 1, 
            eval_steps: int = 100,
            logging_steps: int = 10,
            save_steps: int = None
        ):

        self.alignment.train()
        self.model.eval()  # Keep base model frozen
        
        # Calculate how many epochs we need based on max_steps
        steps_per_epoch = len(self.train_dataloader)
        if max_steps is not None:
            epochs_needed = math.ceil(max_steps / steps_per_epoch)
            total_steps = max_steps
        else:
            epochs_needed = epochs
            total_steps = epochs * steps_per_epoch

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-5
        )
        
        current_step = 0
        
        # Create progress bar for total steps across all epochs
        pbar = tqdm(total=total_steps, 
                    desc="Training", 
                    unit="step")
        
        for epoch in range(epochs_needed):
            
            for batch in self.train_dataloader:
                if max_steps is not None and current_step >= max_steps:
                    break
                    
                self.optimizer.zero_grad()
                top_k = self.top_k_scheduler.get_top_k()
                logits, loss, batch_acc = self.compute_intervention_loss(**batch, top_k=top_k)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                current_step += 1
                
                # Update progress bar with metrics (always)
                pbar.set_postfix({
                    "epoch": f"{epoch+1}/{epochs_needed}",
                    "step": current_step,
                    "loss": f"{loss.item():.4f}",
                    "interv_acc": f"{batch_acc:.4f}",
                })
                pbar.update()
                scheduler.step()
                self.top_k_scheduler.step()

                if current_step % logging_steps == 0:
                    self.writer.add_scalar("train/loss", loss.item(), current_step)
                    self.writer.add_scalar("train/intervention_accuracy", batch_acc, current_step)
                    self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], current_step)
                    self.writer.add_scalar("train/top_k", top_k, current_step)

                if current_step % eval_steps == 0 and self.eval_dataloader is not None:
                    eval_metrics = self.evaluate()
                    
                    # Log to TensorBoard
                    self.writer.add_scalar("eval/loss", eval_metrics["avg_loss"], current_step)
                    for var, acc in eval_metrics["accuracy"].items():
                        self.writer.add_scalar(f'eval/accuracy_{var}', acc, current_step)
                    
                    # Log to terminal
                    evaluation_log = (
                        f"***** Eval results *****\n" + 
                        f"{'step':<12} = {current_step}\n" +
                        f"{'eval_loss':<12} = {eval_metrics['avg_loss']:.6f}\n"
                    )
                    for var, acc in eval_metrics['accuracy'].items():
                        evaluation_log += f"{'eval_interv_acc_' + var:<12} = {acc:.6f}\n"
                    logger.info(evaluation_log)
                
                # Save alignment model at specified intervals
                if save_steps is not None and current_step % save_steps == 0:
                    self.save_alignment(step=current_step)

            # Break if we've reached max_steps
            if max_steps is not None and current_step >= max_steps:
                break
        
        pbar.close()
        
        # Final save and cleanup
        self.save_alignment(step=current_step)
        self.writer.close()
        
        logger.info(f"***** Training completed *****")
        logger.info(f"{'total_steps':<12} = {current_step}")
    
    def evaluate(self, dataloader: DataLoader = None, top_k: int = None) -> Dict[str, float]:
        """Evaluate the model"""

        if dataloader is None:
            dataloader = self.eval_dataloader
        if top_k is None:
            top_k = self.top_k_scheduler.get_top_k()

        self.alignment.eval()
        self.model.eval()
        
        global_acc = {}
        global_loss = 0
        total_batches = 0

        pbar = tqdm(total=len(dataloader), 
                    desc="Evaluating", 
                    unit="batch")
        
        for batch in dataloader:
            intervention_vars = batch["intervention_variables"]
            intervention_vars_key = "-".join(map(str, intervention_vars))

            logits, loss, batch_acc = self.compute_intervention_loss(**batch, top_k=top_k)

            global_loss += loss.item()
            total_batches += 1

            if intervention_vars_key not in global_acc:
                global_acc[intervention_vars_key] = []
                
            global_acc[intervention_vars_key].append(batch_acc)
            pbar.update()

        pbar.close()

        # Average loss
        avg_loss = global_loss / total_batches
        # Average accuracy per intervention variable
        for var in global_acc:
            global_acc[var] = sum(global_acc[var]) / len(global_acc[var])
        
        self.alignment.train()
        
        return {"avg_loss": avg_loss, "accuracy": global_acc}