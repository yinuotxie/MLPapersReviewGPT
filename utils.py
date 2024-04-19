import argparse
import logging


def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    Function to setup loggers.

    Args:
        name (str): The name of the logger.
        log_file (str): The file path to the log file.
        level (int): The logging level.

    Returns:
        logging.Logger: The logger object.
    """
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def log_training_args(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Print the training arguments to the log.

    Args:
        args (argparse.Namespace): The parsed arguments containing training configurations.
        logger (logging.Logger): The logger object to log the arguments.
    """
    logger.info("Training arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}\n")
    logger.info("=" * 30)


def clean_json_output(output: str) -> str:
    """
    Cleans the JSON output from the OpenAI GPT-4 model by removing unnecessary characters.

    This function aims to strip extraneous formatting or characters from the JSON output,
    such as backticks or leading 'json' strings that might be present in the formatted output.

    Args:
        output: A string containing the JSON-formatted output from the GPT-4 model.

    Returns:
        A string with the JSON output cleaned up.
    """
    # Remove backticks, 'json' literals, and any leading/trailing whitespace
    cleaned_output = output.strip("`").replace("json\n", "").strip()
    return cleaned_output
