"""
This file contains the prompts for instructions for the tasks.
"""

## ============================ SYSTEM PROMPT ============================ ##

# system prompt for generating reviews
SYSTEM_PROMPT = """
You are a professional machine learning conference reviewer who reviews a given paper and considers 4 criteria: [Significance and novelty], [Potential reasons for acceptance], [Potential reasons for rejection], and [Suggestions for improvement]. Please ensure that for each criterion, you summarize and provide random number of detailed supporting points from the content of the paper. And for each supporting point within each of criteria, use the format: '<title of supporting point>' followed by a detailed explanation. The criteria you need to focus on are:

1. [Significance and novelty]: Assess the importance of the paper in its research field and the innovation of its methods or findings。
2. [Potential reasons for acceptance]: Summarize reasons that may support the acceptance of the paper, based on its quality, research results, experimental design, etc.
3. [Potential reasons for rejection]: Identify and explain flaws or shortcomings that could lead to the paper's rejection.
4. [Suggestions for improvement]: Provide specific suggestions to help the authors improve the paper and increase its chances of acceptance.

After reading the content of the paper provided below, your response should only include your reviews only, which means always start with [Significance and novelty], dont' repeat the given paper and output things other than your reviews in required format, just extract and summarize information related to these criteria from the provided paper. The paper is given as follows:
"""

## ============================ SUMMARY PROMPT ============================ ##

# summary prompt for making a summary of the review
SUMMARY_PROMPT = """
Your goal is to identify the key concerns raised in the list of reviews, focusing only on potential
reasons for rejection.

Please provide your analysis in JSON format, including a concise summary, and the exact
wording from the review. 
    
Submission Title: {Title}

=====Review:
```
{Review_Text}
```
=====

Example JSON format:
{{
    "1": {{"summary": "<your concise summary>", "verbatim": "<concise, copy the exact
    wording in the review>"}},
    "2": ... 
}}

Analyze the review and provide the key concerns in the format specified above. Ignore minor
    issues like typos and clarifications. Output only json.
"""


## ============================ REVIEW_COMPARISON_RPOMPT ============================ ##

REVIEW_COMPARISON_RPOMPT = """
Your task is to carefully analyze and accurately match the key concerns raised in two reviews, 
ensuring a strong correspondence between the matched points. Examine the verbatim closely.

=====Review A: 
{Review_A}

===== 

=====Review B: 
{Review_B}

===== 

Please follow the example JSON format below for matching points. For instance, if point from review A is nearly identical to point from review B, it should look like this:
{{ 
    "A3-B2": {{"rationale": "<explain why A3 and B2 are nearly identical>","similarity": "<5-10, only an integer>"}},
    ...
}}

**Note that you should only match points with a significant degree of similarity in their concerns. Refrain from matching points with only superficial similarities or weak connections.** For each matched pair, rate the similarity on a scale of 5-10:
- 5 Somewhat Related: Points address similar themes but from different angles.
- 6 Moderately Related: Points share a common theme but with different perspectives or suggestions.
- 7 Strongly Related: Points are largely aligned but differ in some details or nuances.
- 8 Very Strongly Related: Points offer similar suggestions or concerns, with slight differences.
- 9 Almost Identical: Points are nearly the same, with minor differences in wording or presentation.
- 10 Identical: Points are exactly the same in terms of concerns, suggestions, or praises.

If no match is found, output an empty JSON object. Provide your output as JSON only.
"""
