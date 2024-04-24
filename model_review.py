"""
This script provides functionality to generate reviews for academic papers formatted as PDFs using advanced language models. 
It allows for conditional quantization of the model for performance optimization on supported devices.
"""

import re
import argparse
import time
import torch
import scipdf
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from prompts import SYSTEM_PROMPT
from pdf_parser import parse_pdf_abstract, generate_input
from utils import setup_logger


def load_model(model_id: str, quantize: bool, device: str) -> tuple:
    """
    Loads a specified model from Hugging Face with optional quantization.

    Args:
        model_id (str): Identifier for the model on Hugging Face.
        quantize (bool): Flag to determine if quantization should be applied.
        device (str): Device to perform the computation on.

    Returns:
        tuple: Loaded model and tokenizer from Hugging Face.
    """
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id, 
            device_map=device
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer


def extract_output(output: str) -> str:
    """
    Extracts categorized sections from a given text based on specific headings.
    
    Args:
    output (str): The text from which to extract information, where each section is expected
                to begin with a heading in brackets.
    
    Returns:
    str: A formatted string with each heading and its associated content.
    """
    # Define the pattern to capture each section with lookahead for the next section or end of string
    pattern = r"\[(Significance and novelty|Potential reasons for acceptance|Potential reasons for rejection|Suggestions for improvement)\]" \
            r"(.*?)(?=\[(Significance and novelty|Potential reasons for acceptance|Potential reasons for rejection|Suggestions for improvement)\]|\Z)"
    
    # Focus on the relevant part of the output after "[/INST]"
    relevant_output = output.split("[/INST]")[-1]
    
    # Extract sections using regex, ensuring DOTALL to match across lines
    sections = re.findall(pattern, relevant_output, flags=re.DOTALL)
    
    # Convert list of tuples into a dictionary for easier access and avoid duplication
    section_dict = {}
    for section in sections:
        header, content = section[0], section[1].strip()
        if header not in section_dict:
            section_dict[header] = content
    
    # Build the final formatted output
    final_output = ""
    for key, value in section_dict.items():
        final_output += f"[{key}]\n{value}\n\n"
    
    return final_output


def inference(
    user_input: str,
    model: AutoPeftModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
) -> str:
    """
    Generates a review based on user input using a pre-loaded model and tokenizer.

    Args:
        user_input (str): Formatted user input for the model.
        model (AutoPeftModelForCausalLM): Loaded language model.
        tokenizer (AutoTokenizer): Tokenizer for the model.
        device (str): Device to perform the computation on.

    Returns:
        str: Generated and extracted section text as output.
    """
    try:
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_input},
        ]

        encoded_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
            device
        )
        generated_ids = model.generate(
            encoded_input,
            max_new_tokens=1024,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return extract_output(decoded_output)
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate structured academic paper reviews using a trained language model."
    )
    parser.add_argument(
        "--pdf_file", type=str, required=True, help="Path to the PDF file."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="travis0103/mistral_7b_paper_review_lora",
        help="Hugging Face model identifier.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply quantization to the model for performance optimization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference.",
    )

    args = parser.parse_args()
    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please switch to CPU or install a GPU."
        )
    if args.device == "cpu" and args.quantize:
        raise RuntimeError(
            "Quantization is not supported on CPU. Please switch to a CUDA device."
        )

    output_logger = setup_logger("output_logger", "logs/output.log")
    args = parser.parse_args()

    # Parse the PDF content
    output_logger.info("Parsing PDF file...")
    pdf = scipdf.parse_pdf_to_dict(args.pdf_file)
    content = parse_pdf_abstract(pdf)
    user_input = generate_input(content)

    # Generate user input
    user_input = generate_input(content)
    output_logger.info(user_input)
    output_logger.info("=" * 50)

    output_logger.info("Loading model...")
    model, tokenizer = load_model(args.model_id, args.quantize, args.device)
    output_logger.info("=" * 50)

    # Generate the review
    output_logger.info("Generating review...")
    start_time = time.time()
    reviews = inference(user_input, model, tokenizer, args.device)
    end_time = time.time()
    output_logger.info(f"Review generated in {end_time - start_time:.2f} seconds.")
    output_logger.info("=" * 50)

    output_logger.info("Review:")
    output_logger.info(reviews)
