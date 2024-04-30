'''
    Generate UI for uploading pdf and feeding into gpt model and review model to generate reviews for given paper.
    Commands of setting up environment after pip install -r requirements:

    pip install dash dash-bootstrap-components
    pip install git+https://github.com/titipata/scipdf_parser
    pip install spacy
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
from pdf_parser import parse_pdf_abstract, parse_pdf_content, generate_input
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
quantize = True

# load model
openai.api_key = os.getenv('OPENAI_API_KEY')
client = openai.Client()
gpt_model = "gpt-4-turbo"
one_shot = False

# review-model
# model, tokenizer = model_review.load_model(model_id, quantize, device)
output_logger.info("=" * 50)

# app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("CIS6200 Project Group 1: Academic GPT Demo"), width=12),
            dbc.Checklist(
                options=[
                    {"label": "One-Shot", "value": 1}
                ],
                value=[],
                id="options",
                switch=True,
            ),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.RadioItems(
                    id='mode-switch',
                    className='btn-group',
                    inputClassName='btn-check',
                    labelClassName='btn btn-outline-primary',
                    labelCheckedClassName='active',
                    options=[
                        {'label': 'Local', 'value': 'local'},
                        {'label': 'Online', 'value': 'online'}
                    ],
                    value='local',
                    style={'width': '100%', 'padding': '10px'}
                )
            ], width=12)
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
            dbc.Col(dbc.Input(
            id='online-url-input',
            type='text',
            placeholder='Enter URL of PDF file',
            style={'width': '100%', 'height': '60px'}
        ), width=12)
        ]),
        dbc.Row([
            dbc.Label("Extracted Text"),
            dbc.Col(dcc.Textarea(
                id='output-text',
                style={'width': '100%', 'height': 200},
                placeholder="Extracted abstract will be shown here.",
            ), width=12)
        ]),
        dbc.Row([
            dbc.Label("GPT Reviews (Left: Abstract, Right: Full Provided)"),
            dbc.Col(dcc.Textarea(
                id='output-gpt-abstract',
                style={'width': '100%', 'height': 300},
                placeholder="GPT-abstract reviews.",
            ), width=6),
            dbc.Col(dcc.Textarea(
                id='output-gpt-full',
                style={'width': '100%', 'height': 300},
                placeholder="GPT-full reviews",
            ), width=6)
        ]),
        dbc.Row([
            dbc.Label("Model Output"),
            dbc.Col(dcc.Textarea(
                id='output-model',
                style={'width': '100%', 'height': 300},
                placeholder="Model output.",
            ), width=12)
        ]),
        dbc.Row([
            dbc.Label("Raw Model Output Before Pruning"),
            dbc.Col(dcc.Textarea(
                id='output-model-raw',
                style={'width': '100%', 'height': 300},
                placeholder="Raw model output before pruning.",
            ), width=12)
        ]),
        dcc.Store(id='enable-one-shot', data=False),
        dcc.Store(id='output-full-text')
    ])
])

@app.callback(
    Output("enable-one-shot", "data"),
    Input("options", "value")
)
def on_form_change(options):
    print("one-shot enabled?", options)
    if len(options) > 0 and options[0] == 1:
        return True
    return False

@app.callback(
    Output('upload-pdf', 'style'),
    Output('online-url-input', 'style'),
    Input('mode-switch', 'value'),
    # prevent_initial_call=True
)
def toggle_components(selected_option):
    if selected_option == 'local':
        return {
            'width': '100%', 
            'display': 'block', 
            'height': '60px', 
            'lineHeight': '60px',
            'borderWidth': '1px', 
            'borderStyle': 'dashed',
            'textAlign': 'center', 
            'margin': '10px'
            }, {'width': '100%', 'display': 'none'}
    elif selected_option == 'online':
        return {'width': '100%', 'display': 'none'}, {'width': '100%', 'height': '60px', 'display': 'block'}


@app.callback(
    Output('output-text', 'value'),
    Output('output-full-text', 'data'),
    Input('upload-pdf', 'contents'),
    Input('online-url-input', 'value'),
    State('upload-pdf', 'filename'),
    State('upload-pdf', 'last_modified')
)
def update_output(contents, url, filename, date):
    output_logger.info("Parsing PDF file...")
    print("parsing pdf...")
    if url:
        print("url", url)
        article_dict = scipdf.parse_pdf_to_dict(url)
    else:
        print("filename",  filename)
        if contents is None:
            return 'No PDF file uploaded.', ""
        content_type, content_string = contents.split(',')
        if 'application/pdf' not in content_type:
            return 'File is not a PDF. Please upload a PDF file.', ""
    
        # extract input from pdf
        article_dict = scipdf.parse_pdf_to_dict(uploaded_directory + "/" + filename)
    content = parse_pdf_abstract(article_dict)
    user_input = generate_input(content)
    output_logger.info(content["[TITLE]"])
    output_logger.info(content["[ABSTRACT]"])
    print("user_input generated")

    full_input = generate_input(parse_pdf_content(article_dict))
    return user_input, full_input

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
    Output('output-gpt-full', 'value'),
    Input('output-text', 'value'),
    Input('output-full-text', 'data'),
    Input('enable-one-shot', 'data'),
)
def update_gpt_abstract_output(user_input, full_input, one_shot_enabled):
    if user_input and user_input != "No PDF file uploaded.":
        one_shot = one_shot_enabled
        print("gpt reviews generating... one_shot=", one_shot)
        # send to backend model
        output_logger.info("=" * 50)
        output_logger.info("Generating review...")

        start_time = time.time()
        gpt_reviews = gpt_review.inference(user_input, gpt_model, one_shot, client)
        end_time = time.time()
        output_logger.info(f"GPT Review generated in {end_time - start_time:.2f} seconds.")
        output_logger.info("GPT Review:")
        output_logger.info(gpt_reviews)

        gpt_full_reviews = gpt_review.inference(full_input, gpt_model, one_shot, client)
        print("return type", type(gpt_reviews), type(gpt_full_reviews))
        return gpt_reviews, gpt_full_reviews
    return "", ""

@app.callback(
    Output('output-model', 'value'),
    Output('output-model-raw', 'value'),
    Input('output-text', 'value')
)
def update_model_output(user_input):
    # return ""
    if user_input and user_input != "No PDF file uploaded.":
        print("model reviews generating...")
        # send to backend model
        start_time = time.time()
        raw_output, model_reviews = model_review.inference(user_input, model, tokenizer, device)
        end_time = time.time()
        output_logger.info(f"Model Review generated in {end_time - start_time:.2f} seconds.")
        output_logger.info("=" * 50)

        # get response from backend
        output_logger.info("Model Review:")
        output_logger.info(model_reviews)
        return raw_output, model_reviews
    return ""

# Run the app
if __name__ == '__main__':

    app.run_server(debug=True, host='0.0.0.0', port=8080)

    # filename = "1611.03530.pdf"
    # article_dict = scipdf.parse_pdf_to_dict(uploaded_directory + "/" + filename)
    # parsed_abstract = parse_pdf_abstract(article_dict)
    # parsed_article = parse_pdf_content(article_dict)
    # print(generate_user_input(parsed_article))