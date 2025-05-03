# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
from dash import Dash, dcc, html, Input, Output, State
from dash import Dash, dash_table

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

col_style = {'display':'grid', 'grid-auto-flow': 'row'}
row_style = {'display':'grid', 'grid-auto-flow': 'column'}

from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import numpy as np
import plotly.express as px
import pandas as pd

import requests
import base64
import io

app = Dash(__name__)

df = pd.read_csv("iris_extended_encoded.csv",sep=',')
df_csv = df.to_csv(index=False)

app.layout = html.Div(children=[
    html.H1(children='Iris classifier'),
    dcc.Store(id='dataset-store'),
    dcc.Tabs([
    dcc.Tab(label="Explore Iris training data", style=tab_style, selected_style=tab_selected_style, children=[

    html.Div([
        html.Div([
            html.Label(['File name to Load for training or testing'], style={'font-weight': 'bold'}),
            dcc.Input(id='file-for-train', type='text', style={'width':'100px'}),
            html.Div([
                html.Button('Load', id='load-val', style={"width":"60px", "height":"30px"}),
                html.Div(id='load-response', children='Click to load')
            ], style=col_style)
        ], style=col_style),

        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select CSV File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin-top': '10px'
                },
                multiple=False
            ),
            html.Button('Upload', id='upload-val', style={"width":"60px", "height":"30px"}),
            html.Div(id='upload-response', children='Click to upload')
        ], style=col_style| {'margin-top':'20px'})

    ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),


html.Div([
    html.Div([
        html.Div([
            html.Label(['Feature'], style={'font-weight': 'bold'}),
            dcc.Dropdown(
                list(df.columns[1:]), #<dropdown values for histogram>
                df.columns[1],           #<default value for dropdown>
                id='hist-column'
            )
            ], style=col_style ),
        dcc.Graph( id='selected_hist' )
    ], style=col_style | {'height':'400px', 'width':'400px'}),

    html.Div([

    html.Div([

    html.Div([
        html.Label(['X-Axis'], style={'font-weight': 'bold'}),
        dcc.Dropdown(
            list(df.columns[1:]), #<dropdown values for scatter plot x-axis>
            df.columns[1],           #<default value for dropdown>
            id='xaxis-column'
            )
        ]),

    html.Div([
        html.Label(['Y-Axis'], style={'font-weight': 'bold'}),
        dcc.Dropdown(
               list(df.columns[1:]), #<dropdown values for scatter plot y-axis>
               df.columns[2],           #<default value for dropdown>
            id='yaxis-column'
            )
        ])
    ], style=row_style | {'margin-left':'50px', 'margin-right': '50px'}),

    dcc.Graph(id='indicator-graphic')
    ], style=col_style)
], style=row_style),

    html.Div(id='tablecontainer', children=[
        dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], page_size=15,
            id='datatable' )
        ])
    ]),
    dcc.Tab(label="Build model and perform training", id="train-tab", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),
            
            html.Div([
                html.Button('New model', id='build-val', style={'width':'90px', "height":"30px"}),
                dcc.Loading(id='loading-build', type='default', children=html.Div(id='build-response'))
            ], style=col_style | {'margin-top':'20px'}),
            
            html.Div([
                html.Label(['Enter a model ID for re-training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            html.Div([
                html.Button('Re-Train', id='train-val', style={"width":"90px", "height":"30px"})
            ], style=col_style | {'margin-top':'20px', 'width':'90px'})

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

        # html.Div(id='container-button-train', children='')
        dcc.Loading(id="loading-train", type="default", children=html.Div(id='container-button-train'), fullscreen=False)
    ]),
    dcc.Tab(label="Score model", id="score-tab", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a row text (CSV) to use in scoring'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='row-for-score', type='text', style={'width':'300px'}))
            ], style=col_style | {'margin-top':'20px'}),
            html.Div([
                html.Label(['Enter a model ID for scoring'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-score', type='text'))
            ], style=col_style | {'margin-top':'20px'}), 
            html.Div([
                html.Label(['Enter the actual class label (optional)'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='actual-for-score', type='text'))
            ], style=col_style | {'margin-top':'20px'}),           
            html.Div([
                html.Button('Score', id='score-val', style={'width':'90px', "height":"30px"}),
                html.Div(id='score-response', children='Click to score')
            ], style=col_style | {'margin-top':'20px'})
        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),
        
        html.Div(id='container-button-score', children='')
    ]),

    dcc.Tab(label="Test Iris data", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),
            html.Div([
                html.Label(['Enter a model ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            html.Div([
                html.Button('Test', id='test-val'),
            ], style=col_style | {'margin-top':'20px', 'width':'90px'})

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

        html.Div(id='container-button-test', children='')
    ])

    ])
])

# callbacks for Explore data tab

@app.callback(
    Output('load-response', 'children'),
    Input('load-val', 'n_clicks'),
    State('file-for-train', 'value'),
    prevent_initial_call = True
)
def update_output_load(nclicks, filename):
    global df, df_csv

    if nclicks != None:
        # load local data given input filename
        df = pd.read_csv(filename)
        df_csv = df.copy()
        print(f"[LOAD] Loading file: {filename}")
        return 'Load done.'
    else:
        return ''


@app.callback(
    Output('build-response', 'children'),
    Input('build-val', 'n_clicks'),
    State('dataset-for-train', 'value'),
    prevent_initial_call=True
)
def update_output_build(nclicks, dataset_id):
    print (nclicks)
    if nclicks != None:
        print(f"[BUILD] Button clicked. Dataset ID: {dataset_id}")
        # invoke new model endpoint to build and train model given data set ID
        print(f"[BUILD] Sending POST to /iris/model with dataset={dataset_id}")
        response = requests.post("http://localhost:4000/iris/model", json={"dataset": int(dataset_id)})
        print(f"[BUILD] Response: {response.json()}")
        response.raise_for_status()
        model_id = response.json().get("model_id", "Unknown")
        # return the model ID 
        return f"Model ID: {model_id}"
    else:
        return ''

@app.callback(
    Output('upload-response', 'children'),
    Input('upload-val', 'n_clicks'),
    State('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_output_upload(nclicks, contents):
    global df_csv

    if nclicks != None:
        # invoke the upload API endpoint
        print("[UPLOAD] Upload triggered.")
        content_type, content_string = contents.split(',')
        print("[UPLOAD] Decoding file and reading CSV...")
        decoded = base64.b64decode(content_string)
        df_csv = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        files = {'train': io.StringIO(decoded.decode('utf-8'))}
        print("[UPLOAD] Sending POST to /iris/datasets")
        response = requests.post("http://localhost:4000/iris/datasets", files=files)
        print(f"[UPLOAD] Response: {response.json()}")
        dataset_id = response.json()["dataset_id"]
        # return the dataset ID generated
        return f"Dataset ID: {dataset_id}"
    else:
        return ''

@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value')
)
def update_graph(xaxis_column_name, yaxis_column_name):

    fig = px.scatter(x=df.loc[:,xaxis_column_name].values,
                     y=df.loc[:,yaxis_column_name].values)

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=xaxis_column_name)
    fig.update_yaxes(title=yaxis_column_name)

    return fig


@app.callback(
    Output('selected_hist', 'figure'),
    Input('hist-column', 'value')
)
def update_hist(hist_column_name):

    fig = px.histogram(df, x=hist_column_name)

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=hist_column_name)

    return fig

@app.callback(
    Output('datatable', 'data'),
    Input('load-val', 'n_clicks'),
    Input('upload-val', 'n_clicks'),
    prevent_initial_call=True
)
def update_table(load_clicks, upload_clicks):
    return df.to_dict('records')

@app.callback(
    Output('upload-data', 'children'),
    Input('upload-data', 'filename'),
    prevent_initial_call=True
)
def show_uploaded_filename(filename):
    if filename:
        return html.Div([
            html.Span(f"{filename} uploaded. Ready to submit.")
        ])
    return html.Div([
        'Drag and Drop or ',
        html.A('Select CSV File')
    ])


# callbacks for Training tab

@app.callback(
    Output('container-button-train', 'children'),
    Input('train-val', 'n_clicks'),
    State('model-for-train', 'value'),
    State('dataset-for-train', 'value'),
    prevent_initial_call=True
)
def update_output_train(nclicks, model_id, dataset_id):
    if nclicks != None and model_id and dataset_id:
        # add API endpoint request here
        print(f"[TRAIN] Re-training model {model_id} with dataset {dataset_id}")
        print(f"[TRAIN] Sending PUT to /iris/model/{model_id}?dataset={dataset_id}")
        url = f"http://localhost:4000/iris/model/{model_id}?dataset={dataset_id}"
        response = requests.put(url)
        print(f"[TRAIN] Response: {response.json()}")
        response.raise_for_status()

        # Extract training history
        history = response.json().get("Training_history", {})  # Expected to be a dict with loss/accuracy over epochs
        train_df = pd.DataFrame( history )
        train_fig = px.line(train_df, title="Training Progress (Loss & Accuracy)")

        return dcc.Graph( figure=train_fig )
    else:
        return ""

# callbacks for Scoring tab

@app.callback(
    Output('score-response', 'children'),
    Input('score-val', 'n_clicks'),
    State('row-for-score', 'value'),
    State('model-for-score', 'value'),
    State('actual-for-score', 'value'),
    prevent_initial_call=True
)
def update_output_score(nclicks, row_text, model_id, actual_class):
    if nclicks != None:
        print(f"[SCORE] Scoring model {model_id} with input: {row_text}")
        # add API endpoint request for scoring here with constructed input row
        row_cleaned = row_text.replace(" ", "")
        print(f"[SCORE] Sending GET to /iris/model/{model_id}/score?fields={row_cleaned}")
        actual_class = actual_class if actual_class else "-1"
        url = f"http://localhost:4000/iris/model/{model_id}/score?fields={row_cleaned}&actual={actual_class}"
        response = requests.get(url)
        print(f"[SCORE] Response: {response}")
        response.raise_for_status()
        result = response.json()
        score_result = result.get("prediction", "No prediction returned.")
        
        return score_result
    else:
        return ""
    
# callbacks for Testing tab

@app.callback(
    Output('container-button-test', 'children'),
    Input('test-val', 'n_clicks'),
    State('model-for-test', 'value'),
    State('dataset-for-test', 'value'),
    prevent_initial_call=True
)
def update_output_test(nclicks, model_id, dataset_id):
    if nclicks != None:
        print(f"[TEST] Testing model {model_id} with dataset {dataset_id}")
        # add API endpoint request for testing with given dataset ID
        print(f"[TEST] Sending GET to /iris/model/{model_id}/test?dataset={dataset_id}")
        url = f"http://localhost:4000/iris/model/{model_id}/test?dataset={dataset_id}"
        response = requests.get(url)
        print(f"[TEST] Response: {response}")
        response.raise_for_status()
        result = response.json()

        actual = result.get("actual", [])
        print(f"[TEST] Actual: {actual}")
        predicted = result.get("predicted", [])
        print(f"[TEST] Predicted: {predicted}")
        accuracy = result.get("accuracy", None)
        print(f"[TEST] Accuracy: {accuracy:.2%}")

        test_df = pd.DataFrame({'Index': list(range(len(actual))), 'Actual': actual, 'Predicted': predicted})

        # Compute confusion matrix
        labels = sorted(list(set(actual + predicted)))
        cm = confusion_matrix(test_df['Actual'], test_df['Predicted'], labels=labels)

        # Create annotated heatmap
        z = cm.tolist()
        x = [str(l) for l in labels]
        y = [str(l) for l in labels]
        test_fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Viridis', showscale=True, annotation_text=[[str(cell) for cell in row] for row in z])
        test_fig.update_layout(title=f'Confusion Matrix (Accuracy: {accuracy:.2%})', xaxis_title='Predicted Label', yaxis_title='True Label')
        return dcc.Graph( figure=test_fig )
    else:
        return ""


if __name__ == '__main__':
    app.run(debug=True)
