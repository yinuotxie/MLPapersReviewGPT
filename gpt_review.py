"""
This script provides functionality to generate reviews for academic papers formatted as PDFs.
It utilizes an OpenAI GPT model to generate these reviews based on either the full content or
just the abstract of the papers, with support for both standard and one-shot review generation modes.
"""

import argparse
import os
import openai
import scipdf
import time
from prompts import SYSTEM_PROMPT
from pdf_parser import parse_pdf_content, parse_pdf_abstract, generate_user_input
from utils import setup_logger

ONE_SHOT_USER = """
[TITLE]\nOn the optimal precision of GANs\n\n[ABSTRACT]\nGenerative adversarial networks (GANs) are known to face model misspecification when learning disconnected distributions. Indeed, continuous mapping from a unimodal latent distribution to a disconnected one is impossible, so GANs necessarily generate samples outside of the support of the target distribution. In this paper, we make the connection between the performance of GANs and their latent space configuration. In particular, we raise the following question: what is the latent space partition that minimizes the measure of out-of-manifold samples? Building on a recent result of geometric measure theory, we prove there exist optimal GANs when the dimension of the latent space is larger than the number of modes. In particular, we show that these generators structure their latent space as a `simplicial cluster' - a Voronoi partition where centers are equally distant.  We derive both an upper and a lower bound on the optimal precision of GANs learning disconnected manifolds. Interestingly, these two bounds have the same order of decrease: $\\sqrt{\\log m}$, $m$ being the number of modes. Finally, we perform several experiments to exhibit the geometry of the latent space and experimentally show that GANs have a geometry with similar properties to the theoretical one."
"""

ONE_SHOT_ASSISTANT = """
[Significance and novelty]\n<Insights on GANs using Gaussian isoperimetry> The paper brings in a novel perspective by using Gaussian isoperimetry to explore GAN representations and sheds light onto the key aspects of GANs, from mode collapse to their precision/recall performance, especially in the case of disconnected distributions.\n<Exploration of latent space organization for GANs> The exploration of latent space organization for GANs and how it affects precision is a significant and somewhat new contribution to the field of generative models and machine learning.\n\n[Potential reasons for acceptance]\n<Novel theoretical perspective> The paper offers a novel theoretical perspective on GANs, using results from Gaussian isoperimetry to analyze the precision of GANs under certain circumstances.\n<Insights for future research> The paper provides theoretical and empirical insights that can guide future research in improving GAN precision and exploring latent space organization.\n\n[Potential reasons for rejection]\n<Unclear relevance to real data setting> The paper lacks clarity on how the theoretical findings relate to real-world data, which impedes the practical impact of the research.\n<Limitations in practical application> The paper's limitations in connecting theoretical insights to practical applications may warrant further revisions before acceptance.\n\n[Suggestions for improvement]\n<Clarify relevance to real-world data> Further clarity is needed on how the theoretical results can be practically applied to real-world datasets and how the assumptions align with the complexities of real data.\n<Enhance empirical validation> The paper might benefit from strengthening the connection between the theoretical insights and empirical validation, especially in exploring the relevance of assumptions to practical scenarios.\n\n"
"""


def inference(user_input: str, model: str, one_shot: bool, client: openai.Client) -> str:
    """
    Generate reviews for a given PDF paper using the specified GPT model.
    This function handles PDF parsing, content extraction, and interfacing with the OpenAI API to generate the review.

    Args:
        user_input (str): Formatted user input for the model.
        model (str): The OpenAI GPT model identifier to use for generating the review.
        one_shot (bool): Determines if one-shot mode is used, requiring special handling.
        client (openai.Client): OpenAI client instance for sending requests.

    Returns:
        str: The generated review as a string.
    """
    try:
        if one_shot:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ONE_SHOT_USER},
                {"role": "assistant", "content": ONE_SHOT_ASSISTANT},
                {"role": "user", "content": user_input},
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        return response.choices[0].message.content
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf_file", type=str, required=True, help="The path to the PDF file."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt-3.5-turbo", "gpt-4-turbo"],
        default="gpt-4-turbo",
        help="The GPT model to use for inference.",
    )
    parser.add_argument(
        "--openai_api_key", type=str, required=True, help="The OpenAI API key."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["full", "abstract"],
        default="abstract",
        help="The method to use for review generation.",
    )
    parser.add_argument(
        "--one_shot", action="store_true", help="Use one-shot mode if set."
    )

    output_logger = setup_logger("output_logger", "output.log")
    args = parser.parse_args()

    if args.one_shot and args.method == "full":
        print("One-shot mode only supports the abstract method.")
        args.method = "abstract"

    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    
    # Parse the PDF content
    output_logger.info("Parsing PDF file...")
    pdf_dict = scipdf.parse_pdf_to_dict(args.pdf_file)
    content = parse_pdf_abstract(pdf_dict) if args.method == "abstract" else parse_pdf_content(pdf_dict)
        
    # Generate user input
    user_input = generate_user_input(content)
    output_logger.info(user_input)
    output_logger.info("=" * 50)
    client = openai.Client()
    
    # Generate the review
    output_logger.info("Generating review...")
    start_time = time.time()
    reviews = inference(user_input, args.model, args.one_shot, client)
    end_time = time.time()
    output_logger.info(f"Review generated in {end_time - start_time:.2f} seconds.")
    output_logger.info("=" * 50)
    
    output_logger.info(reviews)
