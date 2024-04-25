'''
    Generate UI for uploading pdf and feeding into gpt model and review model to generate reviews for given paper.
    Commands of setting up environment after pip install -r requirements:

    pip install dash dash-bootstrap-components
    pip install git+https://github.com/titipata/scipdf_parser
    python -m spacy download en_core_web_sm

    docker pull grobid/grobid:0.8.0
    docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.0

    To run the app:
        python app.py
'''
from dotenv import load_dotenv
import os
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from pdf_parser import *
import scipdf
import spacy
import time
from utils import setup_logger
from prompts import SYSTEM_PROMPT
import model_review, gpt_review
import openai
import torch

spacy.load('en_core_web_sm')
load_dotenv()

# uploaded file directory
uploaded_directory = "C:/Users/cresc/Downloads"

# set up logger
output_logger = setup_logger("output_logger", "logs/output.log")

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model_id = "travis0103/mistral_7b_paper_review_lora"
quantize = False

# load model
openai.api_key = os.getenv('OPENAI_API_KEY')
client = openai.Client()
gpt_model = "gpt-4-turbo"
one_shot = False

# review-model
model, tokenizer = model_review.load_model(model_id, quantize, device)
output_logger.info("=" * 50)

# app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("CIS6200 Project Group 1: Academic GPT Demo"), width=12)
        ]),
        dbc.Row([
            dbc.Col(dcc.Upload(
                id='upload-pdf',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a PDF File')
                ]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                },
                multiple=False, # multiple file allowance
            ), width=12)
        ]),
        dbc.Row([
            dbc.Col(dcc.Textarea(
                id='output-text',
                style={'width': '100%', 'height': 50},
                placeholder="Extracted abstract will be shown here.",
            ), width=12)
        ])
        dbc.Row([
            dbc.Col(dcc.Textarea(
                id='output-gpt-abstract',
                style={'width': '25%', 'height': 200},
                placeholder="GPT-abstract reviews.",
            ), width=12)
        ])
        dbc.Row([
            dbc.Col(dcc.Textarea(
                id='output-gpt-full',
                style={'width': '25%', 'height': 200},
                placeholder="GPT-full reviews",
            ), width=12)
        ])
        dbc.Row([
            dbc.Col(dcc.Textarea(
                id='output-model',
                style={'width': '50%', 'height': 200},
                placeholder="Model output.",
            ), width=12)
        ])
    ])
])

@app.callback(
    Output('output-text', 'value'),
    Input('upload-pdf', 'contents'),
    State('upload-pdf', 'filename'),
    State('upload-pdf', 'last_modified')
)
def update_output(contents, filename, date):
    print("filename",  filename)
    if contents is None:
        return 'No PDF file uploaded.'
    content_type, content_string = contents.split(',')
    if 'application/pdf' not in content_type:
        return 'File is not a PDF. Please upload a PDF file.'
    
    # extract input from pdf
    output_logger.info("Parsing PDF file...")
    article_dict = scipdf.parse_pdf_to_dict(uploaded_directory + "/" + filename)
    content = parse_pdf_abstract(article_dict)
    user_input = generate_user_input(content)
    output_logger.info(content["[TITLE]"])
    output_logger.info(content["[ABSTRACT]"])
    return user_input

    # from base64 import b64decode
    # import io
    # try:
    #     with pdfplumber.open(io.BytesIO(b64decode(content_string))) as pdf:
    #         all_text = ''
    #         for page in pdf.pages:
    #             all_text += page.extract_text() + '\n'
    #         return all_text
    # except Exception as e:
    #     return f'An error occurred: {e}'

@app.callback(
    Output('output-gpt-abstract', 'value'),
    Input('output-text', 'value')
)
def update_gpt_abstract_output(user_input):
    # send to backend model
    output_logger.info("=" * 50)
    output_logger.info("Generating review...")

    start_time = time.time()
    gpt_reviews = gpt_review.inference(user_input, gpt_model, one_shot, client)
    end_time = time.time()
    output_logger.info(f"GPT Review generated in {end_time - start_time:.2f} seconds.")
    output_logger.info("GPT Review:")
    output_logger.info(gpt_reviews)
    return gpt_reviews

@app.callback(
    Output('output-model', 'value'),
    Input('output-text', 'value')
)
def update_gpt_abstract_output(user_input):
    # send to backend model
    start_time = time.time()
    model_reviews = model_review.inference(user_input, model, tokenizer, device)
    end_time = time.time()
    output_logger.info(f"Model Review generated in {end_time - start_time:.2f} seconds.")
    output_logger.info("=" * 50)

    # get response from backend
    output_logger.info("Model Review:")
    output_logger.info(model_reviews)
    return model_reviews

# Run the app
if __name__ == '__main__':

    app.run_server(debug=True)

    # filename = "1611.03530.pdf"
    # article_dict = scipdf.parse_pdf_to_dict(uploaded_directory + "/" + filename)
    # parsed_abstract = parse_pdf_abstract(article_dict)
    # parsed_article = parse_pdf_content(article_dict)
    # print(generate_user_input(parsed_article))