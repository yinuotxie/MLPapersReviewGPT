import argparse
import json
import logging
import os
from datetime import datetime

import torch
import wandb
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)

from peft import LoraConfig
from trl import SFTTrainer, setup_chat_format
from typing import Tuple

def setup_logging() -> Tuple[logging.Logger, logging.Logger]:
    """
    Set up logging configuration
    """
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    log_filename = f"train.log"
    error_filename = f"error.log"

    info_logger = logging.getLogger('info')
    error_logger = logging.getLogger('error')

    general_handler = logging.FileHandler(os.path.join(log_directory, log_filename))
    error_handler = logging.FileHandler(os.path.join(log_directory, error_filename))

    general_handler.setLevel(logging.INFO)
    error_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    general_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    info_logger.addHandler(general_handler)
    error_logger.addHandler(error_handler)

    info_logger.setLevel(logging.INFO)
    error_logger.setLevel(logging.ERROR)

    return info_logger, error_logger


info_logger, error_logger = setup_logging()


def load_configuration() -> None:
    """
    Load configuration from the config.json file
    """
    with open('config.json') as config_file:
        config = json.load(config_file)
    os.environ["TRANSFORMERS_CACHE"] = config["TRANSFORMERS_CACHE"]
    # set up wandb environment variables
    os.environ["WANDB_PROJECT"] = config["WANDB_PROJECT"]
    os.environ["WANDB_LOG_MODEL"] = config["WANDB_LOG_MODEL"]
    os.environ["HUGGINGFACE_ACCESS_TOKEN"] = config["HUGGINGFACE_ACCESS_TOKEN"]

    
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Train a model with specified parameters")
    parser.add_argument("--train_file", type=str, default="data/train_dataset.json", help="Path to the training dataset")
    parser.add_argument("--test_file", type=str, default="data/test_dataset.json", help="Path to the testing dataset")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Model name to use for training")
    parser.add_argument("--output_dir", type=str, default="model/mistral_7b_academic", help="Output directory to save the model")
    parser.add_argument("--num_of_epochs", type=int, default=1, help="Number of epochs to train the model for")
    parser.add_argument("--max_seq_length", type=int, default=8000, help="Maximum sequence length for the model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device during evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of steps before performing a backward/update pass")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory")
    parser.add_argument("--optim", type=str, default="adamw_torch_fused", help="Optimizer to use")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of steps to train the model for")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy to use")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy to use")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Max gradient norm")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="Type of learning rate scheduler")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load the best model at the end of training")
    parser.add_argument("--run_name ", type=str, default="train", help="Name of the run")
    
    return parser.parse_args()


def load_datasets(train_file: str, test_file: str) -> Tuple[Dataset, Dataset]:
    """
    Load training and testing datasets
    
    Args:
        train_file: str: Path to the training dataset
        test_file: str: Path to the testing dataset
    
    Returns:
        Tuple[Dataset, Dataset]: Tuple containing the training and testing datasets
    """
    try:
        info_logger.info("Loading datasets...")
        train_dataset = load_dataset("json", data_files=train_file, split="train")
        test_dataset = load_dataset("json", data_files=test_file, split="train")
        info_logger.info(f"Train dataset size: {len(train_dataset)}")
        info_logger.info(f"Test dataset size: {len(test_dataset)}")
        return train_dataset, test_dataset
    except Exception as e:
        error_logger.error(f"An error occurred while loading the datasets. Error: {e}", exc_info=True)
        raise e


def initialize_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Initialize the model and tokenizer
    
    Args:
        model_name: str: Name of the model to initialize
        
    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Tuple containing the model and tokenizer
    """
    try:
        info_logger.info(f"Loading model and tokenizer from {model_name}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            cache_dir=os.environ["TRANSFORMERS_CACHE"]
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=os.environ["TRANSFORMERS_CACHE"])
        model, tokenizer = setup_chat_format(model, tokenizer)
        return model, tokenizer
    except Exception as e:
        error_logger.error(f"An error occurred while setting up the model. Error: {e}", exc_info=True)
        raise e 


def prepare_training(args: argparse, 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    train_dataset: Dataset, 
    test_dataset: Dataset) -> SFTTrainer:
    """
    Prepare the training process
    
    Args:
        args: argparse: Command line arguments
        model: AutoModelForCausalLM: Model to train
        tokenizer: AutoTokenizer: Tokenizer to use
        train_dataset: Dataset: Training dataset
        test_dataset: Dataset: Testing dataset
        
    Returns:
        SFTTrainer: Trainer object for training the model
    """
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_of_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        push_to_hub=args.push_to_hub,
        load_best_model_at_end=args.load_best_model_at_end,
        report_to="wandb",
        run_name=args.run_name
    )

    # Set up Lora configuration
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": False}
    )

    return trainer

def main():
    info_logger, error_logger = setup_logging()
    load_configuration()
    args = parse_arguments()
    
    wandb.login()

    api = HfApi()
    api.get_token_permission(token=os.environ["HUGGINGFACE_ACCESS_TOKEN"])

    train_dataset, test_dataset = load_datasets(args.train_file, args.test_file)

    model, tokenizer = initialize_model(args.model_name)

    trainer = prepare_training(args, model, tokenizer, train_dataset, test_dataset)

    try:
        info_logger.info("Starting training process...")
        trainer.train()
        info_logger.info("Training completed.")
        trainer.save_model()
        info_logger.info("Model saved successfully.")
    except Exception as e:
        error_logger.error(f"An error occurred during training. {e}", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
