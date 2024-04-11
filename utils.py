import argparse
import json
import logging
import os

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
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)

        # Set up environment variables for Weights & Biases and Hugging Face
        os.environ["WANDB_PROJECT"] = config.get("WANDB_PROJECT", "")
        os.environ["WANDB_LOG_MODEL"] = config.get("WANDB_LOG_MODEL", "")
        os.environ["WANDB_KEY"] = config.get("WANDB_KEY", "")
        os.environ["HUGGINGFACE_ACCESS_TOKEN"] = config.get("HUGGINGFACE_ACCESS_TOKEN", "")
    except FileNotFoundError:
        print("Error: 'config.json' file not found.")
        raise
    except json.JSONDecodeError:
        print("Error: 'config.json' is not a valid JSON file.")
        raise
    

def print_training_args(args: argparse.Namespace) -> None:
    """
    Print the training arguments to the log.

    Args:
        args (argparse.Namespace): The parsed arguments containing training configurations.
    """
    logging.info("Training arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}\n")
    logging.info("=" * 30)