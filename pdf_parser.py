"""
This module provides functions to parse PDF content and extract structured information including
title, abstract, captions, and sections.
"""

import re


def compile_regex_patterns() -> dict:
    """
    Compile and return a dictionary of regular expressions for text normalization.

    Returns:
        dict: Dictionary mapping pattern names to compiled regular expression objects.
    """
    patterns = {
        "triple_newline": re.compile(r"\n{3}", flags=re.S),
        "newline_space_newline": re.compile(r"\n \n", flags=re.S),
        "escaped_newline": re.compile(r"\\n", flags=re.S),
        "double_newline": re.compile(r"\n{2}", flags=re.S),
    }
    return patterns


regex_patterns = compile_regex_patterns()


def normalize_text(content: str, patterns=regex_patterns) -> str:
    """
    Normalize the input text by replacing multiple newline patterns with a single newline.

    Args:
        content (str): The input text to normalize.
        patterns (dict): Dictionary of compiled regular expressions for text cleaning.

    Returns:
        str: The normalized text.
    """
    for pattern in patterns.values():
        content = re.sub(pattern, "\n", content)
    return content


def captions_to_string(captions: dict) -> str:
    """
    Convert a dictionary of captions into a formatted string.

    Args:
        captions (dict): Dictionary of captions where keys are captions identifiers and values are the captions themselves.

    Returns:
        str: Formatted string representation of captions.
    """
    return "\n".join(f"{key}: {value}" for key, value in captions.items())


def parse_pdf_content(pdf_dict: dict) -> dict:
    """
    Extract structured information from a PDF content dictionary.

    Args:
        pdf_dict (dict): Dictionary containing PDF metadata and content.

    Returns:
        dict: Dictionary of structured information including title, abstract, captions, and sections.
    """
    result = {}
    title = pdf_dict.get("title")
    abstract = pdf_dict.get("abstract")
    captions = {}
    num_figures = num_tables = 0

    for figure in pdf_dict.get("figures", []):
        figure_type = figure["figure_type"]
        caption = figure["figure_caption"]
        if figure_type == "figure":
            captions[f"Figure {num_figures + 1}"] = caption
            num_figures += 1
        elif figure_type == "table":
            captions[f"Table {num_tables + 1}"] = caption
            num_tables += 1

    result.update(
        {
            "[TITLE]": normalize_text(title),
            "[ABSTRACT]": normalize_text(abstract),
            "[CAPTIONS]": normalize_text(captions_to_string(captions)),
        }
    )

    for section in pdf_dict.get("sections", []):
        heading = section["heading"].upper()
        text = normalize_text(section["text"])
        if text.strip():
            result[f"[{heading}]"] = text

    return result


def parse_pdf_abstract(pdf_dict: dict) -> dict:
    """
    Extract the abstract from a PDF content dictionary.

    Args:
        pdf_dict (dict): Dictionary containing PDF metadata and content.

    Returns:
        str: Normalized abstract text.
    """
    title = pdf_dict.get("title")
    abstract = pdf_dict.get("abstract")

    result = {
        "[TITLE]": normalize_text(title),
        "[ABSTRACT]": normalize_text(abstract),
    }
    
    return result


def generate_input(article_dict: dict) -> str:
    """
    Generate a structured user input string from an article dictionary.

    Args:
        article_dict (dict): Dictionary containing structured article information.

    Returns:
        str: User-friendly structured string for display or further processing.
    """
    user_input = "\n".join(
        f"{key}\n{value}\n" for key, value in article_dict.items() if value
    )
    return user_input.strip()


# Example usage:
# import scipdf
# article_dict = scipdf.parse_pdf_to_dict('demo/Transformers.pdf')
# parsed_article = parse_pdf_content(article_dict)
# print(generate_user_input(parsed_article))
