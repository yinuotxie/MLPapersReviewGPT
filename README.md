# Machine Learning Paper Reviews GPT
![pipeline](assets/pipeline.png)

## Description
This project, developed as a final project for the [UPenn CIS6200 Advanced Topics in Deep Learning](https://docs.google.com/document/d/1dkQ4XRhaiZFjGu5i_8Qcoi6MkHwOfivmFFWhBrBF30I/edit), aims to investigate the use of large language models (LLMs) for generating machine learning paper reviews. It is inspired by the study ["Can large language models provide useful feedback on research papers? A large-scale empirical analysis"](https://arxiv.org/pdf/2310.01783.pdf) and utilizes similar techniques. For more details, please see our [project report](report.pdf).

## Methods
We used models such as GPT-3.5-turbo and GPT-4-turbo, and fine-tuned [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) on our custom dataset.

### Dataset
Our dataset, sourced from [OpenReview](https://openreview.net/), includes papers and their reviews. We also used GPT-4 to generate summary reviews, which aided in the fine-tuning process. The dataset is published on the Huggingface dataset hub and can be accessed [here](travis0103/abstract_paper_review).

### Fine-tuning
The models were fine-tuned using the `train.py` script. Our fine-tuned model is available on the Huggingface model hub, accessible [here](travis0103/mistral_7b_paper_review_lora).

## Usage 
The project provides two pipelines for generating reviews:

### Installation
Clone the repository and set up the environment:
```bash
git clone git@github.com:yinuotxie/MLPapersReviewGPT.git
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Note: The scipdf_parser package, required for PDF text extraction, must run within a Docker container. Instructions are available in the [scipdf_parser repository](https://github.com/titipata/scipdf_parser).

### Model Pipeline
Generate reviews using the fine-tuned model. Currently, only the abstracts of papers are supported:
```bash
python model_review.py 
    --pdf_file <path_to_pdf_file> 
    --device <device> 
    --model_id <model_id> 
    --quantize
```

### GPT Pipeline
Alternatively, use the GPT pipeline to generate reviews:
```bash
python gpt_review.py 
    --pdf_file <path_to_pdf_file> 
    --openai_api_key <your_openai_api_key> 
    --model <gpt-3.5-turbo or gpt-4-turbo> 
    --method <full or abstract> 
    --one_shot
```

## Acknowledgements
We extend our deepest gratitude to our professor, [Prof. Lyle Ungar](https://www.cis.upenn.edu/~ungar/), for his invaluable guidance and support throughout the project. We also thank the teaching assistants, Visweswaran Baskaran, Haotong (Victor) Tian, and Royina Karegoudra Jayanth, for their helpful feedback and assistance. 

## References
* [Can large language models provide useful feedback on research papers? A large-scale empirical analysis](https://arxiv.org/pdf/2310.01783.pdf)
* [scipdf_parser](https://github.com/titipata/scipdf_parser)
* [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)