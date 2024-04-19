import argparse
import os
import json
from datetime import datetime

import torch
import wandb
from accelerate import PartialState
from datasets import load_dataset, Dataset
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, setup_chat_format
from typing import Tuple
from utils import log_training_args, setup_logger

info_logger = setup_logger("info_logger", "logs/train_info.log")
error_logger = setup_logger("error_logger", "logs/train_error.log")


def load_configuration() -> None:
    """
    Load configuration from the 'config.json' file and set environment variables.

    The 'config.json' file should contain the following keys:
    - WANDB_PROJECT: The project name for Weights & Biases logging.
    - WANDB_LOG_MODEL: A flag to log the model to Weights & Biases.
    - WANDB_KEY: The API key for Weights & Biases.
    - HUGGINGFACE_ACCESS_TOKEN: The access token for Hugging Face.
    """
    try:
        with open("config.json", "r") as config_file:
            config = json.load(config_file)

        # Set up environment variables for Weights & Biases and Hugging Face
        os.environ["WANDB_PROJECT"] = config.get("WANDB_PROJECT", "")
        os.environ["WANDB_LOG_MODEL"] = config.get("WANDB_LOG_MODEL", "")
        os.environ["WANDB_KEY"] = config.get("WANDB_KEY", "")
        os.environ["HUGGINGFACE_ACCESS_TOKEN"] = config.get(
            "HUGGINGFACE_ACCESS_TOKEN", ""
        )
    except FileNotFoundError:
        print("Error: 'config.json' file not found.")
        raise
    except json.JSONDecodeError:
        print("Error: 'config.json' is not a valid JSON file.")
        raise


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for training configuration.

    Returns:
        argparse.Namespace: The namespace containing all argument values.
    """
    parser = argparse.ArgumentParser(
        description="Train a model with specified parameters"
    )

    # Dataset arguments
    parser.add_argument(
        "--train_file",
        type=str,
        default="demo/data/abstract_dataset/train.json",
        help="Path to the training dataset",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="demo/data/abstract_dataset/test.json",
        help="Path to the testing dataset",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Model name to use for training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model/mistral_7b_lora_paper_review",
        help="Output directory to save the model",
    )
    parser.add_argument(
        "--flash_attention",
        action="store_true",
        help="Use Flash Attention instead of regular attention",
    )

    # Training arguments
    parser.add_argument(
        "--num_of_epochs",
        type=int,
        default=3,
        help="Number of epochs to train the model for",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1280,
        help="Maximum sequence length for the model",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size per device during training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size per device during evaluation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps before performing a backward/update pass",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_false",
        help="Use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--auto_find_batch_size",
        action="store_true",
        help="Automatically find the optimal batch size",
    )

    # Optimization arguments
    parser.add_argument(
        "--optim", type=str, default="adamw_torch", help="Optimizer to use"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.3, help="Max gradient norm"
    )

    # Logging and saving arguments
    parser.add_argument(
        "--logging_steps", type=int, default=45, help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=450, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--save_strategy", type=str, default="steps", help="Save strategy to use"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=450, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        help="Evaluation strategy to use",
    )

    # Scheduler and fine-tuning arguments
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of steps to train the model for",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Type of learning rate scheduler",
    )

    # Mixed precision training
    parser.add_argument(
        "--fp16", action="store_false", help="Enable mixed precision training"
    )

    # Miscellaneous arguments
    parser.add_argument(
        "--push_to_hub",
        action="store_false",
        help="Whether to push the model to the hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="travis0103/mistral_7b_paper_review_lora",
        help="Model ID on the huggingface hub",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_false",
        help="Load the best model at the end of training",
    )
    parser.add_argument("--run_name", type=str, default="train", help="Name of the run")

    return parser.parse_args()


def print_num_trainable_parameters(
    model: AutoModelForCausalLM, perf_config: LoraConfig
) -> None:
    """
    Print the number of trainable parameters of the given model.

    Args:
        model (AutoModelForCausalLM): The model for which the trainable parameters are counted.
        perf_config (LoraConfig): The peft configuration for the model.
    """
    model = get_peft_model(model, perf_config)
    info_logger.info(model.print_trainable_parameters())


def initialize_model(
    model_name: str, use_flash_attn: bool
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Initialize the model and tokenizer with specified configurations.

    Args:
        model_name (str): The name or path of the model to initialize.
        use_flash_attn (bool): Whether to use Flash Attention or not.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The initialized model and tokenizer.
    """
    try:
        info_logger.info(f"Loading model and tokenizer from {model_name}...")

        # Configure model for 4-bit quantization with BitsAndBytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        device_string = PartialState().process_index

        if use_flash_attn:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                use_cache=False,
                device_map={"": device_string},
                attn_implementation="flash_attention_2",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                use_cache=False,
                device_map={"": device_string},
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if use_flash_attn:
            tokenizer.padding_side = "left"
        else:
            tokenizer.padding_side = "right"
            
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

        model, tokenizer = setup_chat_format(model, tokenizer)

        return model, tokenizer
    except Exception as e:
        error_logger.info(
            f"An error occurred while setting up the model and tokenizer. {e}",
            exc_info=True,
        )


def prepare_training(
    args: argparse.Namespace,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> SFTTrainer:
    """
    Prepare the training process by setting up the training arguments and trainer.

    Args:
        args (argparse.Namespace): Command line arguments.
        model (AutoModelForCausalLM): Model to train.
        tokenizer (AutoTokenizer): Tokenizer to use.
        train_dataset (Dataset): Training dataset.
        test_dataset (Dataset): Testing dataset.

    Returns:
        SFTTrainer: Trainer object for training the model.
    """
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # training arguments
        num_train_epochs=args.num_of_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        auto_find_batch_size=args.auto_find_batch_size,
        # optimization arguments
        optim=args.optim,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        # Scheduler and fine-tuning arguments
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        # mixed precision training
        fp16=args.fp16,
        fp16_full_eval=True,
        # Logging and saving arguments
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        # Miscellaneous arguments
        push_to_hub=args.push_to_hub,
        load_best_model_at_end=args.load_best_model_at_end,
        report_to="wandb",
        run_name=args.run_name,
        hub_model_id=args.hub_model_id,
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        max_seq_length=args.max_seq_length,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
    )

    info_logger.info(print_num_trainable_parameters(model, peft_config))

    return trainer


def main():
    args = parse_arguments()

    # check if the gpu supports flash attention v2
    if args.flash_attention:
        if not torch.cuda.is_available():
            error_logger.info("Flash Attention v2 requires a GPU to run.")
            return
        if not torch.cuda.get_device_properties(0).major >= 8:
            error_logger.info(
                "Flash Attention v2 requires a GPU with compute capability >= 8.0."
            )
            return

    # check if the gpu supports fp16 mixed precision training
    if args.fp16:
        if not torch.cuda.is_available():
            error_logger.info("Mixed precision training requires a GPU to run.")
            return
        if not torch.cuda.get_device_properties(0).major >= 7:
            error_logger.info(
                "Mixed precision training requires a GPU with compute capability >= 7.0."
            )
            return

    load_configuration()

    log_training_args(args, info_logger)

    info_logger.info("Loading datasets...")
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    test_dataset = load_dataset("json", data_files=args.test_file, split="train")
    info_logger.info("Datasets loaded successfully.")
    info_logger.info("Training dataset size: %d", len(train_dataset))
    info_logger.info("Testing dataset size: %d", len(test_dataset))
    info_logger.info("=" * 30)

    info_logger.info("Logging to Weights & Biases...")
    wandb.init(
        project="cis6200_academic_gpt",
        config=vars(args),
        name=args.run_name,
        group="DDP",
    )
    info_logger.info("Logging to Hugging Face...")
    login(token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"))
    info_logger.info("Logging setup completed.")
    info_logger.info("=" * 30)

    info_logger.info("Initializing model and tokenizer...")
    model, tokenizer = initialize_model(args.model_name, args.flash_attention)
    info_logger.info("Model and tokenizer initialized successfully.")
    info_logger.info("=" * 30)

    info_logger.info("Preparing training...")
    trainer = prepare_training(args, model, tokenizer, train_dataset, test_dataset)
    info_logger.info("Training prepared successfully.")
    info_logger.info("=" * 30)

    start = datetime.now()
    info_logger.info(f"Starting training at {start:%Y-%m-%d %H:%M:%S}")
    trainer.train()
    end = datetime.now()
    info_logger.info(f"Training completed at {end:%Y-%m-%d %H:%M:%S}")
    training_duration = end - start
    info_logger.info(f"Training duration: {training_duration}")
    info_logger.info("=" * 30)
    
    # Save the model
    trainer.save_model(args.output_dir)    
    info_logger.info("Model saved successfully.")
    info_logger.info("=" * 30)
    
if __name__ == "__main__":
    main()
