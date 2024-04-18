"""
This script provides functions to calculate various similarity metrics between sets of reviews.
These metrics include the hit rate, Jaccard index, Sørensen-Dice coefficient, and the Szymkiewicz-Simpson coefficient.
"""


def calculate_hit_rate(hit_count: int, total_human_reviews: int) -> float:
    """
    Calculate the hit rate, which is the ratio of hits to the total number of human reviews.

    Args:
        hit_count (int): The number of hits.
        total_human_reviews (int): The total number of human reviews considered.

    Returns:
        float: The hit rate as a float.
    """
    return hit_count / total_human_reviews


def calculate_jaccard_index(
    hit_count: int, total_human_reviews: int, total_gpt_reviews: int
) -> float:
    """
    Calculate the Jaccard index, a statistic used for gauging the similarity and diversity of sample sets.
    Jaccard index = (Intersection of human and GPT reviews) / (Union of human and GPT reviews).

    Args:
        hit_count (int): The number of intersecting reviews.
        total_human_reviews (int): The total number of human reviews.
        total_gpt_reviews (int): The total number of GPT-generated reviews.

    Returns:
        float: The Jaccard index as a float.
    """
    return hit_count / (total_human_reviews + total_gpt_reviews - hit_count)


def calculate_sorensen_dice_coefficient(
    hit_count: int, total_human_reviews: int, total_gpt_reviews: int
) -> float:
    """
    Calculate the Sørensen-Dice coefficient, which is a measure of the similarity between two samples.
    Sørensen-Dice coefficient = (2 * Intersection of human and GPT reviews) / (Total human reviews + Total GPT reviews).

    Args:
        hit_count (int): The number of intersecting reviews.
        total_human_reviews (int): The total number of human reviews.
        total_gpt_reviews (int): The total number of GPT-generated reviews.

    Returns:
        float: The Sørensen-Dice coefficient as a float.
    """
    return 2 * hit_count / (total_human_reviews + total_gpt_reviews)


def calculate_szymkiewicz_simpson_coefficient(
    hit_count: int, total_human_reviews: int, total_gpt_reviews: int
) -> float:
    """
    Calculate the Szymkiewicz-Simpson coefficient, also known as the Simpson's coefficient, which measures the degree of overlap between two sets.
    Simpson's coefficient = Intersection of human and GPT reviews / Minimum of (Total human reviews, Total GPT reviews).

    Args:
        hit_count (int): The number of intersecting reviews.
        total_human_reviews (int): The total number of human reviews.
        total_gpt_reviews (int): The total number of GPT-generated reviews.

    Returns:
        float: The Szymkiewicz-Simpson coefficient as a float.
    """
    min_total = min(total_human_reviews, total_gpt_reviews)
    return hit_count / min_total


# Example usage:
# hit_count = 10
# total_human_reviews = 50
# total_gpt_reviews = 30
# print("Hit Rate:", calculate_hit_rate(hit_count, total_human_reviews))
# print("Jaccard Index:", calculate_jaccard_index(hit_count, total_human_reviews, total_gpt_reviews))
# print("Sørensen-Dice Coefficient:", calculate_sorensen_dice_coefficient(hit_count, total_human_reviews, total_gpt_reviews))
# print("Szymkiewicz-Simpson Coefficient:", calculate_szymkiewicz_simpson_coefficient(hit_count, total_human_reviews, total_gpt_reviews))
