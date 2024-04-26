import numpy as np
from dash.exceptions import PreventUpdate
from tqdm import tqdm
import joblib
import pandas as pd
from evaluation import Evaluation, measure_kendall_correlation
from shap_lime_cf import XAI
from neighborhood import Neighborhood
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
import random
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
from itertools import combinations
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import dash
from conference_code.conference import Framework
import plotly.graph_objs as go
from dash import dcc, html
import json

minmax = MinMaxScaler()
input_data = {"data": "heart_db", "classifier": "SVM", "fold": "fold1"}
e = Evaluation(input_data, "linear")

f = Framework(e)
features = e.features
# sample = e.data.df[e.features].iloc[25]
# sample["Major_Vessels"]=1
samples = e.testdf[e.features]
sample = e.testdf[e.features].iloc[0]
sample["Major_Vessels"]=1
# e.data.df[e.features].iloc[0]
# for freezing, 25 with major vessels =1
neighborhood = pd.DataFrame()
weights = []

settings = {}
size_slider = 200
distance = "Euc"
selected_features = []
heart_20 = [27, 59, 91, 118, 123, 182, 183, 188, 213, 231, 275, 295]

# opposite
settings1 = {
    "distance": True,  # Assuming switch3 relates to some 'distance' boolean
    "training": False,
    "custom": False,
    "balanced": -1,
    "restricted": 1,
    "random": True
}

# balanced
settings2 = {
    "distance": True,  # Assuming switch3 relates to some 'distance' boolean
    "training": False,
    "custom": False,
    "balanced": 2,
    "restricted": -1,
    "random": True
}
settings5 = {
    "distance": True,  # Assuming switch3 relates to some 'distance' boolean
    "training": False,
    "custom": False,
    "balanced":-1,
    "restricted": -1,
    "random": True
}

# same
settings3 = {
    "distance": True,  # Assuming switch3 relates to some 'distance' boolean
    "training": False,
    "custom": False,
    "balanced": -1,
    "restricted": 0,
    "random": True
}

neighborhood1, _ = f.get_nbr(sample, settings1,size_slider, distance, selected_features)
neighborhood2, _ = f.get_nbr(sample, settings2, size_slider, distance, selected_features)
neighborhood3, _ = f.get_nbr(sample, settings3, size_slider, distance, selected_features)
neighborhood4 = e.traindf[e.features]
neighborhood5, _ = f.get_nbr(sample, settings2, size_slider, distance,
                             ["Age","Sex","Typical_Angina","Atypical_Angina","Resting_Blood_Pressure","Fasting_Blood_Sugar","Colestrol"])


means_neighborhood1 = np.mean(neighborhood1, axis=0)
means_neighborhood2 = np.mean(neighborhood2, axis=0)
means_neighborhood3 = np.mean(neighborhood3, axis=0)
means_neighborhood4 = np.mean(neighborhood4, axis=0)
means_neighborhood5 = np.mean(neighborhood5, axis=0)

def get_mean(mean_list):
    for f in e.features:
        if f in e.data.continuous:
            mean_list[f] = mean_list[f].round(e.data.dec_precisions[f])
        if f in e.data.categorical:
            categories = np.array(e.data.feature_range[f])
            nearest_category_indices = np.argmin(np.abs(mean_list[f] - categories), axis=0)
            mean_list[f] = categories[nearest_category_indices]
    return mean_list

xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood1)
shap_vals1 = xai_obj.get_shap_vals(sample)
# shap_val = xai_obj.get_shap_beeswarm(samples)

xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood2)
shap_vals2 = xai_obj.get_shap_vals(sample)

xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood3)
shap_vals3 = xai_obj.get_shap_vals(sample)

# xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood4)
shap_vals4 = e.xai.get_shap_vals(sample)

xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood5)
shap_vals5 = xai_obj.get_shap_vals(sample)

cfs = pd.DataFrame()
protos = pd.DataFrame()

def generate_feature_combinations(features):
    all_combinations = []
    for r in range(1, 4):
        for combo in combinations(features, r):
            all_combinations.append(list(combo))
    return all_combinations


fsets = generate_feature_combinations(features)
def sufficiency_sets(sample,
                     output,
                     model,
                     feature_sets,
                     use_prev):
    seets = []

    for feats in feature_sets:
        if use_prev:
            nbrhood = neighborhood
        else:
            nbrhood = e.context.generate_neighbourhood([], sample, features,
                                                       no_of_neighbours=1000,
                                                       probability=False,
                                                       bound=True,
                                                       use_range=False,
                                                       truly_random=True)
        for feat in feats:
            neighbors = nbrhood[nbrhood[feat] != sample[feat]]
        preds = model.predict(neighbors)
        neighbors = neighbors.iloc[preds != output]
        if len(neighbors) == 0:
            continue
        dists = e.context.calculateMahalanobis(neighbors[features],
                                               np.array(sample).reshape(1, -1),
                                               np.cov(e.traindf[features].values))
        inds = np.argsort(dists[:, 0])
        neighbors = neighbors.iloc[inds]

        neighbors = neighbors.iloc[:200]

        if len(neighbors) > 0:
            for feat in feats:
                neighbors[feat] = [sample[feat]] * len(neighbors)
            preds = model.predict(neighbors)
            score = sum((preds == output).astype(int)) / len(neighbors)
            if score >= 0.8:
                seets.append(feats)
        else:
            continue
    return seets

# from collections import defaultdict
# kendal_1 = defaultdict(list)
# kendal_2 = defaultdict(list)
# suff_3 = defaultdict(list)
# itens_to_find = ["Age", "Sex", "Typical_Angina", "Atypical_Angina", "Resting_Blood_Pressure",
#                                   "Fasting_Blood_Sugar", "Colestrol"]
# indices = np.where(~np.isin(features, itens_to_find))[0]
# for i in range(30):
#     sample = e.data.df[e.features].iloc[i]
#     neighborhood1, _ = f.get_nbr(sample, settings1, size_slider, distance, selected_features)
#     neighborhood2, _ = f.get_nbr(sample, settings2, size_slider, distance, selected_features)
#     neighborhood3, _ = f.get_nbr(sample, settings3, size_slider, distance, selected_features)
#     neighborhood4 = e.traindf[e.features]
#     neighborhood5, _ = f.get_nbr(sample, settings2, size_slider, distance,
#                                  ["Age", "Sex", "Typical_Angina", "Atypical_Angina", "Resting_Blood_Pressure",
#                                   "Fasting_Blood_Sugar", "Colestrol"])
#
#     suffs = sufficiency_sets(sample,
#                              e.clf.predict([sample])[0],
#                              e.clf,
#                              fsets,
#                              False)
#
#     xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood1)
#     shap_vals1 = xai_obj.get_shap_vals(sample)
#     indices_pos = np.argsort(shap_vals1)[-3:][::-1]
#     top_3_positive_features = features[indices_pos]
#     indices_neg = np.argsort(shap_vals1)[:3]
#     top_3_negative_features = features[indices_neg]
#     indices_mag = np.argsort(np.abs(shap_vals1))[-3:][::-1]
#     top_3_magnitude_features = features[indices_mag]
#     suff_3["outside_pos"].append(1 if any(set(sublist) == set(top_3_positive_features) for sublist in suffs)  else 0)
#     suff_3["outside_neg"].append(1 if any(set(sublist) == set(top_3_negative_features) for sublist in suffs)  else 0)
#     suff_3["outside_mag"].append(1 if any(set(sublist) == set(top_3_magnitude_features) for sublist in suffs)  else 0)
#
#     xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood2)
#     shap_vals2 = xai_obj.get_shap_vals(sample)
#     indices_pos = np.argsort(shap_vals2)[-3:][::-1]
#     top_3_positive_features = features[indices_pos]
#     indices_neg = np.argsort(shap_vals2)[:3]
#     top_3_negative_features = features[indices_neg]
#     indices_mag = np.argsort(np.abs(shap_vals2))[-3:][::-1]
#     top_3_magnitude_features = features[indices_mag]
#     suff_3["bal_pos"].append(1 if any(set(sublist) == set(top_3_positive_features) for sublist in suffs)  else 0)
#     suff_3["bal_neg"].append(1 if any(set(sublist) == set(top_3_negative_features) for sublist in suffs)  else 0)
#     suff_3["bal_mag"].append(1 if any(set(sublist) == set(top_3_magnitude_features) for sublist in suffs)  else 0)
#
#     xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood3)
#     shap_vals3 = xai_obj.get_shap_vals(sample)
#     indices_pos = np.argsort(shap_vals3)[-3:][::-1]
#     top_3_positive_features = features[indices_pos]
#     indices_neg = np.argsort(shap_vals3)[:3]
#     top_3_negative_features = features[indices_neg]
#     indices_mag = np.argsort(np.abs(shap_vals3))[-3:][::-1]
#     top_3_magnitude_features = features[indices_mag]
#     suff_3["inside_pos"].append(1 if any(set(sublist) == set(top_3_positive_features) for sublist in suffs)  else 0)
#     suff_3["inside_neg"].append(1 if any(set(sublist) == set(top_3_negative_features) for sublist in suffs)  else 0)
#     suff_3["inside_mag"].append(1 if any(set(sublist) == set(top_3_magnitude_features) for sublist in suffs)  else 0)
#
#     # xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood4)
#     shap_vals4 = e.xai.get_shap_vals(sample)
#     indices_pos = np.argsort(shap_vals4)[-3:][::-1]
#     top_3_positive_features = features[indices_pos]
#     indices_neg = np.argsort(shap_vals4)[:3]
#     top_3_negative_features = features[indices_neg]
#     indices_mag = np.argsort(np.abs(shap_vals4))[-3:][::-1]
#     top_3_magnitude_features = features[indices_mag]
#     suff_3["std_pos"].append(1 if any(set(sublist) == set(top_3_positive_features) for sublist in suffs)  else 0)
#     suff_3["std_neg"].append(1 if any(set(sublist) == set(top_3_negative_features) for sublist in suffs)  else 0)
#     suff_3["std_mag"].append(1 if any(set(sublist) == set(top_3_magnitude_features) for sublist in suffs)  else 0)
#
#     xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood5)
#     shap_vals5 = xai_obj.get_shap_vals(sample)
#     # kendal_1["std_outside"].append(measure_kendall_correlation(shap_vals4,shap_vals1))
#     # kendal_1["std_balanced"].append(measure_kendall_correlation(shap_vals4, shap_vals2))
#     # kendal_1["std_inside"].append(measure_kendall_correlation(shap_vals4, shap_vals3))
#     fr1 = [shap_vals4[index] for index in indices]
#     fr2 = [shap_vals5[index] for index in indices]
#     fr3 = [shap_vals2[index] for index in indices]
#     kendal_2["std_frozen"].append(measure_kendall_correlation(fr1, fr2))
#     kendal_2["balanced_frozen"].append(measure_kendall_correlation(fr3, fr2))
#
# print()




def get_confidence():
    classs = e.clf.predict([sample])
    return f'Classification{classs} with Confidence of {round(e.clf.predict_proba([sample])[0][classs[0]],2)}'







suffs = sufficiency_sets(sample,
                         e.clf.predict([sample])[0],
                         e.clf,
                         fsets,
                         False)


def necessity_sets(sample,
                   output,
                   model,
                   feature_sets,
                   use_prev):
    seets = []
    for feats in feature_sets:
        neighbors = e.context.generate_neighbourhood(feats, sample, features,
                                                     no_of_neighbours=1000,
                                                     probability=False,
                                                     bound=True,
                                                     use_range=False,
                                                     truly_random=True)

        preds = model.predict(neighbors)
        # check that the output is same as "output"
        neighbors = neighbors.iloc[preds == output]

        if len(neighbors) > 0:
            dists = e.context.calculateMahalanobis(neighbors[features],
                                                   np.array(sample).reshape(1, -1),
                                                   np.cov(e.traindf[features].values))
            inds = np.argsort(dists[:, 0])
            neighbors = neighbors.iloc[inds]
            neighbors = neighbors.iloc[:200]

            for feat in feats:
                if feat not in e.context.continuous:
                    select = e.context.feature_range[feat]
                    if sample[feat] in select:
                        select = e.nece_suff_obj.remove_values_from_list(select, sample[feat])
                    neighbors[feat] = np.random.choice(select, len(neighbors))
                elif e.context.precisions[feat] == 0:
                    select = list(range(int(e.context.mini[feat]), int(e.context.maxi[feat])))
                    if sample[feat] in select:
                        select = e.nece_suff_obj.remove_values_from_list(select, sample[feat])
                    neighbors[feat] = np.random.choice(select, len(neighbors))
                else:
                    select = list(np.random.uniform(e.context.mini[feat], e.context.maxi[feat],
                                                    2 * len(neighbors)))
                    select = [round(r, e.context.precisions[feat]) for r in select]
                    if sample[feat] in select:
                        select = e.nece_suff_obj.remove_values_from_list(select, sample[feat])
                    select = select[:len(neighbors)]
                    neighbors[feat] = select

            preds = model.predict(neighbors)
            score = sum((preds != output).astype(int)) / len(neighbors)
            if score > 0.9:
                seets.append(feats)
    return seets


neces = necessity_sets(sample,
                       e.clf.predict([sample])[0],
                       e.clf,
                       fsets,
                       False)


def generate_graph(button_id):
    # Define different data sets for each button
    print(shap_vals1)
    hover_data_dict = {
        'button-1': get_mean(means_neighborhood1),
        'button-2': get_mean(means_neighborhood2),
        'button-3': get_mean(means_neighborhood3),
        'button-4': get_mean(means_neighborhood4),
        'button-5': get_mean(means_neighborhood5),
    }
    datasets = {
        'button-1': shap_vals1,
        'button-2': shap_vals2,
        'button-3': shap_vals3,
        'button-4': shap_vals4,
        'button-5': shap_vals5,
    }

    # Get the data set for the clicked button or default data set
    values = datasets.get(button_id, [0, 0, 0, 0, 0])
    hover_texts = hover_data_dict.get(button_id, ["No hover info available"] * 5)
    print(values)
    # Create a simple bar graph with the values
    # fig = go.Figure(data=[go.Bar(x=features, y=values, text=hover_texts,
    #     hoverinfo='text')])

    # fig = go.Figure(data=[go.Bar(x=features, y = values, text=hover_texts,
    #     hoverinfo='text')])

    fig = go.Figure(data=[go.Bar(x=features, y=values)])
    fig.update_layout(
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=14
    )
    fig.update_layout(
        yaxis=dict(
            range=[-0.4, max(values) + 0.05],  # Adjust the max value as needed
            dtick=0.05,  # Set y-axis label increment
        )
    )
    return fig


right_panel = dbc.Col([
    dcc.Graph(id='main-plot', figure={}, style={'height': '80vh'}),
    # Adjusted height for the plot to make space for buttons
    html.Div([
        dbc.Button("SHAP-outside", id="button-1", className="me-1"),
        dbc.Button("SHAP-balanced", id="button-2", className="me-1"),
        dbc.Button("SHAP-inside", id="button-3", className="me-1"),
        dbc.Button("SHAP-larger neighborhood", id="button-4", className="me-1"),
        dbc.Button("SHAP-frozen features", id="button-5", className="me-1")
    ], className='d-flex justify-content-center mt-3')  # Button container with centered flexbox layout
], width=9)

columns = sample.index
input_fields = [
    html.Div([
        html.Label(f'{column}:'),
        dcc.Input(
            id=f'input-{column}',
            type='number',
            value=sample[column],
            className='mb-2',  # Add margin for spacing
            step=1  # This allows changing the values using the up and down arrows
        )
    ], className='mb-3') for column in columns
]

# Generate dropdown options dynamically from the features list
dropdown_options = [{'label': feature, 'value': feature.replace(' ', '')} for feature in features]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    dbc.Row([
        # Full-height left panel with controls
        dbc.Col(html.Div([
            html.H4('Controls', className='text-center mt-3 mb-3'),
            html.Div(id='hidden-div', style={'display': 'none'}, children=json.dumps(sample.to_dict())),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='sample-dropdown',
                        options=[{'label': col, 'value': col} for col in columns],
                        className='mb-2',
                        placeholder='Select a column',
                    )], width=6),
                dbc.Col([
                    dcc.Input(
                        id='sample-input',
                        type='number',
                        step=1,  # Allows changing the values using the up and down arrows
                        className='mb-2',
                        placeholder='Enter value',
                    )
                ], width=6),
            ]),
            html.Button('View Input Sample Prediction', id='sample-pred-button', className='mb-3'),
            html.Div(id='prediction-output', className='fancy-text-box'),
            dbc.Switch(id='switch-1', label='Use Samples from one class', value=False, className='mb-2'),
            dbc.Switch(id='switch-2', label='Select if Outside', value=False, className='mb-2', disabled=True),
            dbc.Switch(id='switch-3', label='Use Balanced Distribution', value=False, className='mb-2'),
            dbc.Switch(id='switch-4', label='Use Skewed Distribution', value=False, className='mb-2'),
            dbc.Switch(id='switch-5', label='Select for skewing towards opposite class', value=False, className='mb-2'),
            dbc.Switch(id='switch-6', label='Use Mahalanobis Distance', value=False, className='mb-2'),
            dbc.Checkbox(id='feature-freeze-checkbox', label='Is Feature Freeze', className='mb-2'),
            dcc.Dropdown(
                id='feature-dropdown',
                options=dropdown_options,
                multi=True,
                placeholder="Select features",
                className='mb-3',
                disabled=True
            ),
            html.Div(["Size of Neighborhood",
                      dcc.Slider(
                          id='size-slider',
                          min=100,
                          max=1000,
                          step=100,
                          value=5,
                          className='mb-2',
                      )
                      ]),
            html.Button('Update Neighborhood', id='update-button', className='mb-2'),
            dcc.Store(id='settings-store'),  # To store the JSON object
            html.Div(id='settings-output'),
            html.H4('Necessary and Sufficient Sets', className='mb-1'),
            dbc.Button('Necessary Sets', id='toggle-button1', className='mb-2'),
            dcc.Dropdown(
                id='nece-feats',
                options=[

                ],
                multi=True,
                style={'display': 'none'}  # Initially hide the dropdown
            ),
            dbc.Button('Sufficient Sets', id='toggle-button2', className='mb-2'),
            dcc.Dropdown(
                id='suff-feats',
                options=[

                ],
                multi=True,
                style={'display': 'none'}  # Initially hide the dropdown
            ),
            html.Div([
                html.H6('Necessary Set', className='mb-1'),
                html.P(neces, id='nece_f', className='mb-2'),
                html.H6('Sufficient Set', className='mb-1'),
                html.P(suffs, id='suff_f', className='mb-2'),
            ]),
        ]), width=3, className='border-end py-3 h-100 d-flex flex-column'),
        # Centered right panel with plot
        right_panel
    ], className='g-0', style={'height': '100vh'}),
], fluid=True)


@app.callback(
    [Output('nece-feats', 'style'),
     Output('nece-feats', 'options')],
    [Input('toggle-button1', 'n_clicks')],
    [State('nece-feats', 'style')]
)
def toggle_features(n_clicks, current_style):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    nece_features = [",".join(nece) for nece in neces]

    if current_style['display'] == 'none':
        return {'display': 'block'}, nece_features
    else:
        return {'display': 'none'}, []


@app.callback(
    [Output('suff-feats', 'style'),
     Output('suff-feats', 'options')],
    [Input('toggle-button1', 'n_clicks')],
    [State('suff-feats', 'style')]
)
def toggle_features(n_clicks, current_style):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    suff_features = [",".join(suff) for suff in suffs]

    if current_style['display'] == 'none':
        return {'display': 'block'}, suff_features
    else:
        return {'display': 'none'}, []


@app.callback(
    Output('switch-2', 'disabled'),
    [Input('switch-1', 'value')]
)
def toggle_switch(switch1_value):
    return not switch1_value


@app.callback(
    Output('switch-5', 'disabled'),
    [Input('switch-4', 'value')]
)
def toggle_switch(switch_value):
    return not switch_value




@app.callback(
    Output('main-plot', 'figure'),
    [Input('button-1', 'n_clicks'),
     Input('button-2', 'n_clicks'),
     Input('button-3', 'n_clicks'),
     Input('button-4', 'n_clicks'),
     Input('button-5', 'n_clicks')],
    prevent_initial_call=True  # This will prevent the callback from firing when the app loads
)
def update_graph(btn1,btn2,btn3,btn4,btn5):
    print("ok", flush=True)
    ctx = dash.callback_context

    print(shap_vals1)
    # lime_vals = xai_obj.get_lime(sample, False, neighborhood, weights)
    if not ctx.triggered:
        # No button has been clicked yet, return the default graph
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        print(f'Button clicked: {button_id}')  # This print statement helps to debug
        return generate_graph(button_id)

    return generate_graph(button_id)

#
# @app.callback(
#     Output('prediction-output', 'children'),
#     Input('sample-pred-button', 'n_clicks'),
#     prevent_initial_call=True
# )
# def update_output(n_clicks):
#     if n_clicks is None:
#         return ''
#     else:
#         prediction = get_confidence()
#         return html.Div([html.P(prediction, className='fancy-text-box')])

@app.callback(
    [Output('prediction-output', 'children'),
    Output('hidden-div', 'children')],  # A hidden div that stores the series
    [Input('sample-pred-button', 'n_clicks'),
     Input('sample-input', 'value')],
    [State('sample-dropdown', 'value')],
    prevent_initial_call=True
)
def update_output(n_clicks,new_value, selected_column):
    if n_clicks is None:
        return ''
    else:
        prediction = get_confidence()
        sample[selected_column] = new_value
        print("here: ",prediction)
        neighborhood1, _ = f.get_nbr(sample, settings1, size_slider, distance, selected_features)
        neighborhood2, _ = f.get_nbr(sample, settings2, size_slider, distance, selected_features)
        neighborhood3, _ = f.get_nbr(sample, settings3, size_slider, distance, selected_features)
        neighborhood4 = e.traindf[e.features]
        neighborhood5, _ = f.get_nbr(sample, settings2, size_slider, distance,
                                     ["Age","Sex","Typical_Angina","Atypical_Angina","Resting_Blood_Pressure","Fasting_Blood_Sugar","Colestrol"])
        global means_neighborhood1,means_neighborhood2, means_neighborhood3, means_neighborhood4, means_neighborhood5
        means_neighborhood1 = np.mean(neighborhood1, axis=0)
        means_neighborhood2 = np.mean(neighborhood2, axis=0)
        means_neighborhood3 = np.mean(neighborhood3, axis=0)
        means_neighborhood4 = np.mean(neighborhood4, axis=0)
        means_neighborhood5 = np.mean(neighborhood5, axis=0)

        xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood1)
        global shap_vals1
        shap_vals1 = xai_obj.get_shap_vals(sample)

        xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood2)
        global shap_vals2
        shap_vals2 = xai_obj.get_shap_vals(sample)

        xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood3)
        global shap_vals3
        shap_vals3 = xai_obj.get_shap_vals(sample)

        # xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood4)
        global shap_vals4
        shap_vals4 = e.xai.get_shap_vals(sample)

        xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood5)
        global shap_vals5
        shap_vals5 = xai_obj.get_shap_vals(sample)
        return html.Div([html.P(prediction, className='fancy-text-box')]),json.dumps(sample.to_dict())


@app.callback(
    Output('settings-store', 'data'),
    [Input('update-button', 'n_clicks')],
    [State('switch-1', 'value'),
     State('switch-2', 'value'),
     State('switch-3', 'value'),
     State('switch-4', 'value'),
     State('switch-5', 'value'),
     State('switch-6', 'value'),
     State('feature-freeze-checkbox', 'value'),
     State('feature-dropdown', 'value'),
     State('size-slider', 'value')]
)
def update_settings(n_clicks, switch1, switch2, switch3, switch4, switch5, switch6, is_feature_freeze,
                    selected_features_, size_slider_):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    dbc.Switch(id='switch-1', label='Use Samples from one class', value=False, className='mb-2'),
    dbc.Switch(id='switch-2', label='Select if Outside', value=False, className='mb-2', disabled=True),
    dbc.Switch(id='switch-3', label='Use Balanced Distribution', value=False, className='mb-2'),
    dbc.Switch(id='switch-4', label='Use Skewed Distribution', value=False, className='mb-2'),
    dbc.Switch(id='switch-5', label='Select for skewing towards opposite class', value=False, className='mb-2'),
    dbc.Switch(id='switch-6', label='Use Mahalanobis Distance', value=False, className='mb-2'),
    dbc.Checkbox(id='feature-freeze-checkbox', label='Is Feature Freeze', className='mb-2'),
    balanced = -1
    if not switch1: restricted = -1
    if switch2:
        restricted = 1
    else:
        restricted = 0
    if switch3:
        balanced = 2
    elif switch4:
        if switch5:
            balanced = 1
        else:
            balanced = 0
    global distance
    if switch6:
        distance = "MB"
    else:
        distance = "Euc"
    global selected_features,size_slider,settings
    selected_features = selected_features_
    size_slider = size_slider_

    settings = {
        "distance": True,  # Assuming switch3 relates to some 'distance' boolean
        "training": False,
        "custom": False,
        "balanced": balanced,
        "restricted": restricted,
        "random": True
    }

    return settings


@app.callback(
    Output('feature-dropdown', 'disabled'),
    [Input('feature-freeze-checkbox', 'value')]
)
def toggle_dropdown(is_feature_freeze):
    return not is_feature_freeze


@app.callback(
    Output('sample-input', 'value'),
    [Input('sample-dropdown', 'value')]
)
def update_input(selected_column):
    return sample[selected_column]


# @app.callback(
#     Output('hidden-div', 'children'),  # A hidden div that stores the series
#     [Input('sample-input', 'value')],
#     [State('sample-dropdown', 'value')]
# )
# def update_series(new_value, selected_column):
#     sample[selected_column] = new_value
#     print(sample)
#     neighborhood1, _ = f.get_nbr(sample, settings1, size_slider, distance, selected_features)
#     neighborhood2, _ = f.get_nbr(sample, settings2, size_slider, distance, selected_features)
#     neighborhood3, _ = f.get_nbr(sample, settings3, size_slider, distance, selected_features)
#     neighborhood4, _ = f.get_nbr(sample, settings2, 500, distance, selected_features)
#     neighborhood5, _ = f.get_nbr(sample, settings2, size_slider, distance,
#                                  ["Age", "Sex", "Typical_Angina", "Atypical_Angina"])
#
#     xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood1)
#     global shap_vals1
#     shap_vals1 = xai_obj.get_shap_vals(sample)
#
#     xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood2)
#     global shap_vals2
#     shap_vals2 = xai_obj.get_shap_vals(sample)
#
#     xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood3)
#     global shap_vals3
#     shap_vals3 = xai_obj.get_shap_vals(sample)
#
#     xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood4)
#     global shap_vals4
#     shap_vals4 = xai_obj.get_shap_vals(sample)
#
#     xai_obj = XAI(e.clf, e.data, e.train_inds, "SVM", None, neighborhood5)
#     global shap_vals5
#     shap_vals5 = xai_obj.get_shap_vals(sample)
#     return json.dumps(sample.to_dict())


if __name__ == '__main__':
    app.run_server(debug=True)
