import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from pdf_parser import *


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
                style={'width': '100%', 'height': 200},
                placeholder="Extracted text will be shown here...",
            ), width=12)
        ])
    ])
])

@app.callback(
    Output('output-text', 'value'),
    Input('upload-pdf', 'contents')
)
def update_output(contents):
    if contents is None:
        return 'No PDF file uploaded.'
    content_type, content_string = contents.split(',')
    if 'application/pdf' not in content_type:
        return 'File is not a PDF. Please upload a PDF file.'
    
    # extract input from pdf

    # send to backend model

    # get response from backend

    

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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)