import argparse
import logging
from transformers import (
    Trainer,
    TrainingArguments,
    Qwen2ForCausalLM,
    Qwen2Config
)
from src.api.tokenizer import BNCTokenizer

from datasets import Dataset
import json
from typing import List
import os


# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def prepare_dataset(texts: List[str], tokenizer: BNCTokenizer, max_length: int) -> Dataset:
    """Prepare dataset using datasets.Dataset"""
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize_function(examples):
        """Tokenize texts and create input/target pairs"""
        # Tokenize with left padding
        tokenized = tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Extract input_ids and attention_mask
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing texts",
        load_from_cache_file=True
    )
    
    return tokenized_dataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Qwen2 model for causal language modeling")
    
    # Data arguments
    parser.add_argument(
        "--train_file",
        type=str,
        default="train.txt",
        help="Path to training text file (default: train.txt)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of training samples to use (default: -1)"
    )
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=20,
        help="Minimum text length to include in training (default: 20)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization (default: 128)"
    )
    
    # Tokenizer arguments
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=20000,
        help="Tokenizer vocabulary size (default: 20000)"
    )
    
    # Model arguments
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="Model hidden size (default: 512)"
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=6,
        help="Number of hidden layers (default: 6)"
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--num_key_value_heads",
        type=int,
        default=8,
        help="Number of key-value heads (default: 8)"
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=2048,
        help="Intermediate size in feed-forward layer (default: 2048)"
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=512,
        help="Maximum position embeddings (default: 512)"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for model checkpoints (default: ./results)"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device (default: 4)"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Number of steps between logging (default: 1)"
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help="Directory for storing logs (default: ./logs)"
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    logger.info("Training configuration:")
    logger.info(f"  Train file: {args.train_file}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Epochs: {args.num_train_epochs}")
    logger.info(f"  Batch size: {args.per_device_train_batch_size}")
    logger.info(f"  Max samples: {args.max_samples}")
    logger.info(f"  Max sequence length: {args.max_length}")
    logger.info(f"  Vocab size: {args.vocab_size}")
    logger.info(f"  Hidden size: {args.hidden_size}")
    logger.info(f"  Number of layers: {args.num_hidden_layers}")
    
    # Check if training file exists
    if not os.path.exists(args.train_file):
        logger.error(f"Training file '{args.train_file}' not found!")
        return
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = BNCTokenizer(
        top_k=args.vocab_size
    )
    tokenizer.padding_side = "left"

    # Load model configuration
    logger.info("Creating model configuration...")
    config = Qwen2Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
    )

    # Initialize model
    logger.info("Initializing model...")
    model = Qwen2ForCausalLM(config)
    
    # Log trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Define training arguments
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_only_model=True,
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        report_to="tensorboard",
        bf16=True
    )

    # Load and prepare training data
    logger.info(f"Loading training data from {args.train_file}...")
    with open(args.train_file, "r") as f:
        texts = f.readlines()
    
    texts = [text.strip() for text in texts if len(text.strip()) > args.min_text_length]
    
    if len(texts) > args.max_samples:
        texts = texts[:args.max_samples]
        logger.info(f"Using {args.max_samples} samples out of {len(texts)} available")
    else:
        logger.info(f"Using all {len(texts)} samples")
    
    logger.info("Preparing dataset...")
    train_dataset = prepare_dataset(
        texts,
        tokenizer,
        max_length=args.max_length
    )

    # Initialize Trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset
    )

    logger.info("Starting training...")
    # Start training
    trainer.train()
    
    logger.info(f"Training completed! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()