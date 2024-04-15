import re


def compile_regex_patterns() -> dict:
    """
    Compile and return a dictionary of regular expressions used for cleaning text.
    
    Returns:
        dict: A dictionary of compiled regular expressions.
    """
    patterns = {
        'triple_newline': re.compile(r"\n{3}", flags=re.S),
        'newline_space_newline': re.compile(r"\n \n", flags=re.S),
        'escaped_newline': re.compile(r"\\n", flags=re.S),
        'double_newline': re.compile(r"\n{2}", flags=re.S)
    }
    return patterns


regex_patterns = compile_regex_patterns()


def normalize_text(content: str, patterns=regex_patterns) -> str:
    """
    Normalize the input text by replacing multiple newlines and other specified patterns with a single newline.
    
    Args:
        content (str): The input text to normalize.
        patterns (dict): A dictionary of compiled regular expressions used for cleaning text.

    Returns:
        str: The normalized text.
    """
    for pattern in patterns.values():
        content = re.sub(pattern, "\n", content)
    return content


def captions_to_string(captions: dict) -> str:
    """Convert a dictionary of captions into a formatted string."""
    return '\n'.join(f"{key}: {value}" for key, value in captions.items())


def clean_caption(caption: str) -> str:
    """Remove prefix from the caption and clean it."""
    return " ".join(caption.split(":")[2:]).strip()


def parse_pdf_content(pdf_dict: dict) -> dict:
    """Parse the PDF content from a dictionary and extract structured information including title, abstract, and captions."""
    result = {}
    title = pdf_dict.get('title')
    abstract = pdf_dict.get('abstract')
    captions = {}

    num_figures = num_tables = 0

    for figure in pdf_dict.get('figures', []):
        figure_type = figure['figure_type']
        if figure_type == 'figure':
            captions[f'Figure {num_figures + 1}'] = clean_caption(figure['figure_caption'])
            num_figures += 1
        else:
            captions[f'Table {num_tables + 1}'] = figure['figure_caption']
            num_tables += 1

    result.update({
        '[TITLE]': normalize_text(title),
        '[ABSTRACT]': normalize_text(abstract),
        '[CAPTIONS]': normalize_text(captions_to_string(captions)),
    })

    for section in pdf_dict.get('sections', []):
        heading = section['heading'].upper()
        text = normalize_text(section['text'])
        if text.strip():
            result[f'[{heading}]'] = text

    return result


def generate_user_input(article_dict: dict) -> str:
    """Generate a structured string from article dictionary for user input."""
    user_input = '\n'.join(f"{key}\n{value}\n" for key, value in article_dict.items() if value)
    return user_input.strip()


# import scipdf
# # Example usage:
# article_dict = scipdf.parse_pdf_to_dict('demo/Transformers.pdf')
# parsed_article = parse_pdf_content(article_dict)
# print(generate_user_input(parsed_article))