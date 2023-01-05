import base64
import io
import pathlib
import sys
# sys.path
import os

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


import numpy as np
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from PIL import Image
from io import BytesIO
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import pandas as pd
import plotly.graph_objs as go
import scipy.spatial.distance as spatial_distance


import pickle

from utils.pred_post_inf import PostInfluencePredict



# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

data_dict = {
    "af_all": pd.read_csv(DATA_PATH.joinpath("af_all.csv"), dtype={'fid': str}, encoding="ISO-8859-1"),
}

# Import datasets here for running the Local version
IMAGE_DATASETS = ("af_all")
WORD_EMBEDDINGS = ("wikipedia_3000", "twitter_3000")

# Init postinfluencer predictor
post_predictor = PostInfluencePredict()


# gloabal variables
pred_post_dict = {
    'uid':{
        'from_uid': None,
        'to_uid': None
    },
    'name':
    {
        'from_uid': None,
        'to_uid': None
    }
    
}

last_clickData = None


def numpy_to_b64(array, scalar=True):
    # Convert from 0-1 to 0-255
    if scalar:
        array = np.uint8(255 * array)

    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return im_b64


# Methods for creating components in the layout code
def Card(children, **kwargs):
    return html.Section(children, className="card-style")


def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )


def NamedInlineRadioItems(name, short, options, val, **kwargs):
    return html.Div(
        id=f"div-{short}",
        style={"display": "inline-block"},
        children=[
            f"{name}:",
            dcc.RadioItems(
                id=f"radio-{short}",
                options=options,
                value=val,
                labelStyle={"display": "inline-block", "margin-right": "7px"},
                style={"display": "inline-block", "margin-left": "7px"},
            ),
        ],
    )


def create_layout(app):
    # Actual layout of the app
    return html.Div(
        className="row",
        style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
        children=[
            # Header
            html.Div(
                className="row header",
                id="app-header",
                style={"background-color": "#f9f9f9"},
                children=[
                    html.Div(
                        [
                            html.Img(
                                src=app.get_asset_url("dash-logo.png"),
                                className="logo",
                                id="plotly-image",
                            )
                        ],
                        className="three columns header_img",
                    ),
                    html.Div(
                        [
                            html.H3(
                                "Influencer Explorer",
                                className="header_title",
                                id="app-title",
                            )
                        ],
                        className="nine columns header_title_container",
                    ),
                ],
            ),
         
            html.Div(
                className="row background",
                style={"padding": "50px"},
                children=[
                    html.H4("User level",)
                ] 
                
            ),
            # Body
            html.Div(
                className="row background",
                style={"padding": "10px"},
                children=[
                    html.Div(
                        className="three columns",
                        children=[
                            Card(
                                [
                                    dcc.Dropdown(
                                        id="dropdown-dataset",
                                        searchable=False,
                                        clearable=False,
                                        options=[
                                            # {
                                            #     "label": "Smartphone Influencers",
                                            #     "value": "af_smartphone",
                                            # },
                                            {
                                                "label": "ALL Influencers",
                                                "value": "af_all",
                                            },
                                          
                                        ],
                                        placeholder="Select a dataset",
                                        value="af_all",
                                    ),

                                    dcc.RadioItems(
                                        id='radio-choose-user',
                                        options=[
                                            {'label': 'From user', 'value': 'from_uid'},
                                            {'label': 'To user', 'value': 'to_uid'}
                                        ]
                                    ),

                                    Card(
                                    style={"padding": "5px"},
                                    children=[
                                        html.Div(id="div-show-post-pred-info"),
                                    ]),

                                    html.Div([
                                        dcc.Textarea(
                                            id='textarea-input-post',
                                            # value='Textarea content initialized\nwith multiple lines of text',
                                            placeholder='Enter content of post',
                                            style={'width': '100%', 'height': "200"},
                                        ),
                                    ]),

                                    html.Button('Predict influence', id='button-pred-post', n_clicks=0),
                                    Card(
                                        style={"padding": "5px"},
                                        children=[
                                            html.Div(id="div-show-post-pred"),
                                        ],
                                    ),

                                    html.Button('Rank influencers', id='button-pagerank', n_clicks=0),
                                    Card(
                                        style={"padding": "5px"},
                                        children=[
                                            html.Div(id="div-show-pagerank"),
                                        ],
                                    ),
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="six columns",
                        children=[
                            dcc.Graph(id="graph-3d-plot-tsne", style={"height": "98vh"})
                        ],
                    ),
                    html.Div(
                        className="three columns",
                        id="euclidean-distance",
                        children=[
                            Card(
                                style={"padding": "5px"},
                                children=[
                                    html.Div(id="div-plot-click-af"),
                                ],
                            )
                        ],
                    ),
                ],
            ),
            html.Div(
                className="row background",
                style={"padding": "50px"},
                children=[
                    html.H4("Community level",)
                ] 
                
            ),
            # Body
            html.Div(
                className="row background",
                style={"padding": "10px"},
                children=[
                    html.Div(
                        className="three columns",
                        
                    ),
                    html.Div(
                        className="six columns",
                        children=[
                            dcc.Graph(id="graph-3d-plot-community", style={"height": "98vh"})
                        ],
                    ),
                   html.Div(
                        className="five columns",
                        children=[
                            html.H4("Top 10 influencers by Amplification Factor score"),
                            
                            Card(
                                style={"padding": "5px"},
                                children=[
                                    html.Div(id="div-plot-click-community"),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )


def demo_callbacks(app):
    
    def generate_figure_image(groups, layout):
        data = []

        for idx, val in groups:
            
            size = val['af'] / 10
            size = [3 if k == 0 else k for k in size]
            scatter = go.Scatter3d(
                name=idx,
                x=val["x"],
                y=val["y"],
                z=val["z"],
                # text=[idx for _ in range(val["x"].shape[0])],
                text=[k for k in val['af']],
                textposition="top center",
                mode="markers",
                marker=dict(size=size, symbol="circle"),
            )
            data.append(scatter)

        figure = go.Figure(data=data, layout=layout)

        return figure

    def generate_figure_community(groups, layout):
        data = []

        for idx, val in groups:
            
            size = [np.mean(val['af'])]
            scatter = go.Scatter3d(
                name=idx,
                x=[np.mean(val["x"])],
                y=[np.mean(val["y"])],
                z=[np.mean(val["z"])],
                # text=[idx for _ in range(val["x"].shape[0])],
                text=[np.mean(val['af'])],
                # textposition="top center",
                mode="markers",
                marker=dict(size=size, symbol="circle"),
            )
            data.append(scatter)

        figure = go.Figure(data=data, layout=layout)

        return figure

    @app.callback(
        Output('div-show-pagerank', 'children'),
        [Input('button-pagerank', 'n_clicks'),
         Input("dropdown-dataset", "value")],
        [State('textarea-input-post', 'value')]
    )

    def cal_pagerank(n_clicks, dataset, value):
        TOP = 10
        if n_clicks > 0:
            rank_uids = post_predictor.get_top_by_post()
            # print(rank_uids)

            rs_pagerank = []
            for i in range(TOP):
                # Retrieve the image corresponding to the index
                df_row = data_dict[dataset].loc[data_dict[dataset].fid == str(rank_uids[i])].iloc[0]
                
                if str(df_row['name']) == 'nan':
                    df_row['name'] = 'anonymous'

                rs_pagerank.append(
                    html.H6("{0} | {1} : {2:.2f}".format(i+1, str(df_row['name']), df_row['af_score']))
                )
                
            return html.Div(children=rs_pagerank)

        return None


    @app.callback(
        Output('div-show-post-pred', 'children'),
        [Input('button-pred-post', 'n_clicks')],
        [State('textarea-input-post', 'value'),]
    )
    def pred_post(n_clicks, value):
        if n_clicks > 0:
            global pred_post_dict
            from_uid = pred_post_dict['uid']['from_uid']
            to_uid = pred_post_dict['uid']['to_uid']
            prob = post_predictor.predict(from_uid, to_uid, value)
            prob = prob * 100
            return html.H5("Influence probability: {0:.2f} %".format(prob))

    @app.callback(
        Output("graph-3d-plot-tsne", "figure"),
        [
            Input("dropdown-dataset", "value"),
        ],
    )
    def display_3d_scatter_plot(
        dataset,
    ):
        if dataset:
            # print('dataset: ', dataset)

            # set up needed files for post predictor
            post_predictor.load_files(dataset)

            path = f"demo_embeddings/{dataset}/"

            try:

                data_url = [
                    "demo_embeddings",
                    dataset,
                    "data.csv",
                ]

                full_path = PATH.joinpath(*data_url)
                embedding_df = pd.read_csv(
                    full_path, index_col=0, encoding="ISO-8859-1"
                )

            except FileNotFoundError as error:
                print(
                    error,
                    "\nThe dataset was not found. Please generate it using generate_demo_embeddings.py",
                )
                return go.Figure()

            # Plot layout
            axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)

            layout = go.Layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
            )

            # For Image datasets
            if dataset in IMAGE_DATASETS:
                embedding_df["label"] = embedding_df.index

                groups = embedding_df.groupby("label")
                figure = generate_figure_image(groups, layout)

            else:
                figure = go.Figure()

            return figure

    @app.callback(
        Output("graph-3d-plot-community", "figure"),
        [
            Input("dropdown-dataset", "value"),
        ],
    )
    def display_3d_scatter_plot_community(
        dataset,
    ):
        if dataset:
            path = f"demo_embeddings/{dataset}/"

            try:

                data_url = [
                    "demo_embeddings",
                    dataset,
                    "data.csv",
                ]

                full_path = PATH.joinpath(*data_url)
                # print('* full path comm. : ', full_path)
                embedding_df = pd.read_csv(
                    full_path, index_col=0, encoding="ISO-8859-1"
                )

            except FileNotFoundError as error:
                print(
                    error,
                    "\nThe dataset was not found. Please generate it using generate_demo_embeddings.py",
                )
                return go.Figure()

            # Plot layout
            axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)

            layout = go.Layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
            )

            # For Image datasets
            if dataset in IMAGE_DATASETS:
                embedding_df["label"] = embedding_df.index

                groups = embedding_df.groupby("label")
                figure = generate_figure_community(groups, layout)

            else:
                figure = go.Figure()

            return figure

    # when clicking node in user embedding
    
    @app.callback(
        [
            Output("div-plot-click-af", "children"),
            Output("div-show-post-pred-info", "children")
        ],
        [
            Input("graph-3d-plot-tsne", "clickData"),
            Input("dropdown-dataset", "value"),
            Input("radio-choose-user", "value"),
        ],
    )

    def display_click_af(
        clickData, dataset, radio_value
    ):
        # print('*Click data: ', clickData)

        if dataset in IMAGE_DATASETS and clickData:
            # Load the same dataset as the one displayed
            try:
                data_url = [
                    "demo_embeddings",
                    dataset,
                    "data.csv",
                ]         

                full_path = PATH.joinpath(*data_url)

                embedding_df = pd.read_csv(full_path, dtype={'fid': str})

                af_info_url = [
                    "af_info",
                    "af_all",
                    "dirs.hdf5"
                ]

            except FileNotFoundError as error:
                print(
                    error,
                    "\nThe dataset was not found. Please generate it using generate_demo_embeddings.py",
                )
                return

            # Convert the point clicked into float64 numpy array
            click_point_np = np.array(
                [clickData["points"][0][i] for i in ["x", "y", "z"]]
            ).astype(np.float64)
            # Create a boolean mask of the point clicked, truth value exists at only one row
            bool_mask_click = (
                embedding_df.loc[:, "x":"z"].eq(click_point_np).all(axis=1)
            )
            # Retrieve the index of the point clicked, given it is present in the set
            if bool_mask_click.any():
                clicked_idx = embedding_df[bool_mask_click].index[0]

                fid = embedding_df.iloc[clicked_idx]['fid']
                # print('fid: ', fid)

                # Retrieve the image corresponding to the index
                df_row = data_dict[dataset].loc[data_dict[dataset].fid == fid].iloc[0]
                # print('df row: ', df_row)
                
                rs_info = html.Div([
                        html.P('Name: ' + str(df_row['name'])),
                        html.P('User id: ' + str(df_row['fid'])),
                        html.P('Active index: {0:.2f}'.format(df_row['active'])),
                        html.P('Fame of posts index: {0:.2f}'.format(df_row['post_fame'])),
                        # html.P('influence counts: ' + str(df_row['influence_count'])),
                        # html.P('Industries : ' + str(df_row['industries'])),
                        # html.P('Work : ' + str(df_row['work'])),
                        html.P('Amplification factor score: {0:.2f}'.format(df_row['af_score'])),
                        # html.P('influenced fids: ' + str(inf_fids)),
                ])

                textarea_value = str(df_row['name'])


                if radio_value is None:
                    return rs_info, None

                global pred_post_dict
                global last_clickData
                if clickData != last_clickData:
                    pred_post_dict['uid'][radio_value] = str(df_row['fid'])
                    pred_post_dict['name'][radio_value] = str(df_row['name'])
                    last_clickData = clickData

                rs_pred_post_info = html.Div([
                    html.H6('From user: ' + str(pred_post_dict['name']['from_uid'])),
                    html.H6('To user: ' + str(pred_post_dict['name']['to_uid'])),
                ]) if not pred_post_dict['name']['from_uid'] is None else html.Div([])

                # print(pred_post_dict)

                return rs_info, rs_pred_post_info

        return None, None


    # when clicking in community node
    @app.callback(
        Output("div-plot-click-community", "children"),
        [
            Input("graph-3d-plot-community", "clickData"),
            Input("dropdown-dataset", "value"),
        ],
    )

    def display_click_community(
        clickData, dataset
    ):
        comm_df = pd.DataFrame()
        if dataset in IMAGE_DATASETS and clickData:
            # Load the same dataset as the one displayed

            try:
                data_url = [
                    "demo_embeddings",
                    dataset,
                    "data.csv",
                ]         

                full_path = PATH.joinpath(*data_url)
                embedding_df = pd.read_csv(full_path, dtype={'fid': str})

                af_info_url = [
                    "af_info",
                    "af_all",
                    "dirs.hdf5"
                ]
      
            except FileNotFoundError as error:
                print(
                    error,
                    "\nThe dataset was not found. Please generate it using generate_demo_embeddings.py",
                )
                return

            comm_id = clickData["points"][0]['curveNumber'] - 1

            comm_df = embedding_df[embedding_df.iloc[:,0] == comm_id]
            comm_df = comm_df.sort_values('af', ascending=False).head(10)

        if len(comm_df) == 0:
            return None

        display = []
        for i, dt in enumerate(comm_df.to_dict(orient='records')):
            display.append(
                    html.H6("{0} | {1}: {2:.2f}".format(i+1, dt['name'], dt['af']))
                )
            
        return html.Div(children = display)
        return dash_table.DataTable(
                                    columns=[{"name": i, "id": i} for i in ['name', 'af']],
                                    data=comm_df.to_dict('records'),
                                )