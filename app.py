#required imports
import dash
import requests
from pydash import *
from dash import dcc
from dash import html
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from collections import Counter
from datetime import datetime
import pandas as pd
from dash import dash_table as dt
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
from plotly_calplot import calplot
import time
from plotly_calplot.layout_formatter import apply_general_colorscaling
import plotly.express as px

#filter warnings
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

#helper function for calendar plot
today_date = datetime.today().date()
wait = True
color_init = True

def get_date(int_value):
    date = datetime.utcfromtimestamp(int_value).date()
    return date

def get_date_fig(data, year):
    dates = [
        get_date(submission['creationTimeSeconds'])
        for submission in data['result']
    ]
    dummy_start_date = min(dates)
    dummy_end_date = today_date
    dummy_df = pd.DataFrame({
        "ds": pd.date_range(dummy_start_date, dummy_end_date),
        "value": np.random.randint(low=0, high=1, size=(pd.to_datetime(dummy_end_date) - pd.to_datetime(dummy_start_date)).days + 1,)
    })
    counts = dict()
    for date in dates:
        if date <= today_date:
            counts[date] = counts.get(date, 0) + 1
    for i in range(dummy_df.shape[0]):
        dummy_df.iloc[i, 1] = counts.get(dummy_df.iloc[i, 0].date(), 0)
    dummy_df = dummy_df[dummy_df['ds'].apply(lambda x:x.year) == year]
    
    fig = calplot(
        dummy_df,
        x="ds",
        y="value",
        gap=3.5,
        years_title=True,
        month_lines_width=3, 
        month_lines_color="#999",
        total_height=250,
        showscale=True,
    )
    fig.update_layout(
        yaxis=dict(title_text=f"{year}", titlefont=dict(size=30))
    )
    fig.layout.annotations[0].update(text="Daily Solves")
    max_val = max(dummy_df.iloc[:, 1])
    return fig, max_val

def get_figures_from_user_data(data):
    total_year_options = [i for i in range(2008, today_date.year + 1)]
    user_years_figs = {}
    for y in total_year_options:
        try:
            fig, max_val = get_date_fig(data, y)
            user_years_figs[y] = fig
            user_years_figs[-y] = max_val
        except:
            pass
    return user_years_figs

#define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

card2 = dbc.Card(
    [
        dbc.CardHeader("Submission Languages"),
        dbc.CardBody(dcc.Graph(id='languages-graph')),
    ],
    style={"width": "100%","display": "inline-block", "padding": "10px"},
)

card3 = dbc.Card(
    [
        dbc.CardHeader("Problems Solved by Difficulty"),
        dbc.CardBody(dcc.Graph(id='difficulty-graph')),
    ],
    style={"width": "100%","display": "inline-block", "padding": "10px"},
)

card4 = dbc.Card(
    [
        dbc.CardHeader("Problems Solved by Tag"),
        dbc.CardBody(dcc.Graph(id='tag-graph')),
    ],
    style={"width": "100%","display": "inline-block", "padding": "10px"},
)

card5 = dbc.Card(
    [
        dbc.CardHeader("Time Taken to Solve Problems of Each Difficulty"),
        dbc.CardBody(dcc.Graph(id='time-taken-violin')),
    ],
    style={"width": "100%","display": "inline-block", "padding": "10px"},
)

card6 = dbc.Card(
    [
        dbc.CardHeader("Daily Solves"),
        dbc.CardBody(
            html.Div([
                dcc.Dropdown(
                    id='year-dropdown',
                ),
                dcc.Graph(id='calendar-display'),
                dcc.Markdown(
                    '### Color Scaling',
                    style={
                        'textAlign': 'center',
                        'fontSize': '12px',
                        'fontWeight': 'bold',
                        'marginBottom': '10px',
                        'fontFamily': 'Arial, sans-serif',
                        'color': '#BBB'
                    }
                ),
                dcc.Slider(
                    id='colorscale-slider',
                    min=0,
                    max=60,
                    step=1,
                    value=10,
                    marks={i: str(i) for i in range(0, 61, 5)},
                    tooltip={'always_visible': True, 'placement': 'bottom'}
                )
            ])
        )
    ]
)

card7 = dbc.Card(
    [
        dbc.CardHeader("Verdicts Graph"),
        dbc.CardBody(dcc.Graph(id='verdicts-graph')),
    ],
    style={"width": "100%", "display": "inline-block", "padding": "10px"},
)

card8 = dbc.Card(
    [
        dbc.CardHeader("Total number of contests"),
        dbc.CardBody(dcc.Graph(id='contests-graph')),
    ],
    style={"width": "100%","display": "inline-block", "padding": "10px"},
)

card9 = dbc.Card(
    [
        dbc.CardHeader("Total number of unsolved questions"),
        dbc.CardBody(dcc.Graph(id='unsolved-graph')),
    ],
    style={"width": "100%", "display": "inline-block", "padding": "10px"},
)

#define layout of the app
app.layout = html.Div([
    html.H2(children='Codeforces Visualizations', style={'textAlign': 'center', 'color': 'white',}),

    html.Br(),
    
    html.Div(id='radioitems-div', children=[
        dbc.RadioItems(
            id='handle-selection',
            options=[
                {'label': 'One handle', 'value': 'one'},
                {'label': 'Two handles', 'value': 'two'}
            ],
            value='one',
            inline=True,
            
            
        ),
    ], style={'display':'flex', 'align-items':'center','justify-content':'center','margin-bottom':'15px' }),
    
    html.Div(
        id='inputs',
        children=[
            html.Div(id='handle-1-container', children=[
            html.Label('Enter handle 1:'),
            dbc.Input(id='input-1', type='text', placeholder='Enter handle 1'),
            html.Br()
        ], style={'display': 'none', 'margin-right':'15px'}),

        html.Div(id='handle-2-container', children=[
            html.Label('Enter handle 2:'),
            dbc.Input(id='input-2', type='text',placeholder='Enter handle 2'),
            html.Br()
        ], style={'display': 'none', 'margin-right': '15px'}),

        html.Div(id='submit-div',children=[
            dbc.Button('Submit', id='submit-button', n_clicks=0, outline=True, color='info'),
        ], style={'width' : '100px', }),
        

        ],
        style={  'display': 'flex','justify-content':'center', 'align-items': 'center'}
    ),

    html.Div(id='button-options-div',children=[
        dbc.Button('Submission Languages',id='submission-languages-button',n_clicks=0, color='info'),
        dbc.Button('Problems solved by Difficulty',id='problemsolveddifficulty-button',n_clicks=0, color='info'),
        dbc.Button('Problems solved by Tag',id='problemsolvedtag-button',n_clicks=0, color='info'),
        dbc.Button('Verdicts Graphs',id='verdicts-graphs-button',n_clicks=0, color='info'),
        dbc.Button('Number of Contest',id='number-of-contest-button',n_clicks=0, color='info'),
        dbc.Button('Unsolved Questions',id='unsolved-questions-button',n_clicks=0, color='info'),
        dbc.Button('Time to solve',id='time-to-solve-button',n_clicks=0, color='info'),
        dbc.Button('Daily Solves',id='daily-solves-button',n_clicks=0, color='info'),
    ], style={'display':'flex', 'justify-content':'space-around', 'align-items':'center'}),
    
    # html.Div(id='display-figure')
    dbc.Row(id='display-figure', children=[]),
    html.Div([
        dbc.Row([card2]),
        dbc.Row([card3]),
        dbc.Row([card4, card7]),
        dbc.Row([card8, card9]),
        dbc.Row([card5]),
        dbc.Row([card6]),
    ], style={'display':'None'}),

], style={'background-color':'black', 'color': 'white',})





# Define a callback to show/hide the input boxes depending on the selected option
@app.callback(
    [dash.dependencies.Output('handle-1-container', 'style'),
     dash.dependencies.Output('handle-2-container', 'style'),
     dash.dependencies.Output('input-1', 'value'),
     dash.dependencies.Output('input-2', 'value')],
    [dash.dependencies.Input('handle-selection', 'value')]
)
#function to show/hide input boxes
def show_hide_handles(value):
    if value == 'one':
        return {'display': 'block', 'margin-right':'15px'}, {'display': 'none'}, '', ''
    else:
        return {'display': 'block', 'margin-right':'15px'}, {'display': 'block', 'margin-right':'15px'}, '', ''


@app.callback(
    Output('display-figure', 'children'),
    [
        Input('submission-languages-button', 'n_clicks'),
        Input('problemsolveddifficulty-button', 'n_clicks'),
        Input('problemsolvedtag-button', 'n_clicks'),
        Input('verdicts-graphs-button', 'n_clicks'),
        Input('number-of-contest-button', 'n_clicks'),
        Input('unsolved-questions-button', 'n_clicks'),
        Input('time-to-solve-button', 'n_clicks'),
        Input('daily-solves-button', 'n_clicks'),
    ]
)


def update_display_figure(
    n_clicks_languages,
    n_clicks_difficulty,
    n_clicks_tag,
    n_clicks_verdicts,
    n_clicks_contests,
    n_clicks_unsolved,
    n_clicks_time,
    n_clicks_solves,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'submission-languages-button':
        return dbc.Row([card2])
    elif button_id == 'problemsolveddifficulty-button':
        return dbc.Row([card3])
    elif button_id == 'problemsolvedtag-button':
        return dbc.Row([card4])
    elif button_id == 'verdicts-graphs-button':
        return dbc.Row([card7])
    elif button_id == 'number-of-contest-button':
        return dbc.Row([card8])
    elif button_id == 'unsolved-questions-button':
        return dbc.Row([card9])
    elif button_id == 'time-to-solve-button':
        return dbc.Row([card5])
    elif button_id == 'daily-solves-button':
        return dbc.Row([card6])
    else:
        return []



#app callback and respective functions 1-6
@app.callback(
        Output('year-dropdown', 'options'),
        Input('submit-button', 'n_clicks'),
        dash.dependencies.State('input-1', 'value')
)
def update_user_data(n_clicks, handle):
    if n_clicks > 0 and handle:
        global data
        global wait
        url = 'https://codeforces.com/api/user.status'
        params = {'handle': handle}

        response = requests.get(url, params=params)
        data = response.json()

        user_years_figs = get_figures_from_user_data(data)
        dropdown_options = list(tuple([{'label': str(year), 'value': year} for year in user_years_figs.keys() if year > 0]))
        wait = True
        return dropdown_options
    return {}
    
@app.callback(
        Output('year-dropdown', 'value'),
        Input('submit-button', 'n_clicks'),
        dash.dependencies.State('input-1', 'value'),
)
def update_calendar_scaling_init(n_clicks, handle):
    global wait
    global color_init

    if n_clicks > 0 and handle and color_init:
        time.sleep(1)
        global data
        
        user_years_figs = get_figures_from_user_data(data)
        dropdown_options = [{'label': str(year), 'value': year} for year in user_years_figs.keys() if year > 0]
        color_init = False
        # print(max(list(user_years_figs.keys())))
        return max(list(user_years_figs.keys()))
    return {}

@app.callback(Output('calendar-display', 'figure'),
            Input('year-dropdown', 'value'),
            Input('colorscale-slider', 'value'),  
            Input('submit-button', 'n_clicks'),
            dash.dependencies.State('input-1', 'value'))
def update_figure(year, slider_value, n_clicks, handle):
    if n_clicks > 0 and handle and year:
        global data

        user_years_figs = get_figures_from_user_data(data)
        fig = apply_general_colorscaling(user_years_figs[year], 0, slider_value)
        return fig
    return {}

@app.callback(
    Output('time-taken-violin', 'figure'),
    Input('submit-button', 'n_clicks'),
    [dash.dependencies.State('input-1', 'value'),
    dash.dependencies.State('input-2', 'value'),
    dash.dependencies.State('handle-selection', 'value')]
)
def update_time_taken_violin(n_clicks, handle1, handle2, handle_type):
    global wait 
    if n_clicks <=0:
        return {}
    
    if handle_type=='one':
        time.sleep(1)
        global data
        solved = filter_(data['result'], lambda x: x['verdict'] == 'OK')
        difficulties = group_by(solved, lambda x: x['problem']['index'][0])

        df = pd.DataFrame(columns=['Difficulty', 'Time Taken'])

        for d in difficulties.items():
            for s in d[1]:
                timer = s['timeConsumedMillis'] / 1000 / 60  # convert to minutes
                df_row = pd.DataFrame({'Difficulty': [d[0]], 'Time Taken': [timer]})
                df = pd.concat([df, df_row])

        figure = px.violin(df, x="Difficulty", y="Time Taken", box=True, points="all", hover_data=df.columns,
                           color_discrete_sequence=px.colors.qualitative.Pastel,
                           category_orders={"Difficulty": ["A", "B", "C", "D", "E"]})
        figure.update_layout(title="Time Taken to Solve Problems of Each Difficulty")

        return figure
    else:
        time.sleep(1)
        url = 'https://codeforces.com/api/user.status'

        params1 = {'handle': handle1}
        params2 = {'handle': handle2}
        response1 = requests.get(url, params=params1)
        response2 = requests.get(url, params=params2)
        data1 = response1.json()
        data2 = response2.json()
        
        handles=[handle1, handle2]
        solved_handles = {handle: filter_(data['result'], lambda x: x['verdict'] == 'OK') for data,handle in [(data1, handle1),(data2,handle2)]}
        difficulties_handles = {handle: group_by(solved_handles[handle], lambda x: x['problem']['index'][0]) for handle in handles}
        
        df = pd.DataFrame(columns=['Handle', 'Difficulty', 'Time Taken'])

        for handle in handles:
            for difficulty, submissions in difficulties_handles[handle].items():
                for submission in submissions:
                    timer = submission['timeConsumedMillis'] / 1000 / 60  # convert to minutes
                    df_row = pd.DataFrame({'Handle': [handle], 'Difficulty': [difficulty], 'Time Taken': [timer]})
                    df = pd.concat([df, df_row])

        figure = px.violin(df, x="Difficulty", y="Time Taken", color="Handle", box=True, points="all",
                           title="Time Taken to Solve Problems of Each Difficulty",
                           category_orders={"Difficulty": ["A", "B", "C", "D", "E"]})

        return figure

    return {}





@app.callback(
    Output('tag-graph', 'figure'),
    Input('submit-button', 'n_clicks'),
    [dash.dependencies.State('input-1', 'value'),
    dash.dependencies.State('input-2', 'value'),
    dash.dependencies.State('handle-selection', 'value')]
)
def update_tag_graph_callback(n_clicks, handle1, handle2, handle_type):
    global wait
    if n_clicks <= 0:
        return {}
    if handle_type=='one':
        time.sleep(1)
        global data
        solved = filter_(data['result'], lambda x: x['verdict'] == 'OK')
        tags = group_by(solved, lambda x: x['problem']['tags'][0] if x['problem']['tags'] else 'None')

        tags_count = {}
        for tag, problems in tags.items():
            tags_count[tag] = len(problems)

        sorted_tags_count = dict(sorted(tags_count.items(), key=lambda item: item[1], reverse=True))

        if len(sorted_tags_count) > 0:
            fig = go.Figure(data=go.Scatterpolar(
                r=list(sorted_tags_count.values()),
                theta=list(sorted_tags_count.keys()),
                fill='toself'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(sorted_tags_count.values())]
                    )),
                showlegend=False,
                title='Problems Solved by Tag'
            )

            return fig
    
    else:
        def generate_scatterpolar(tags_count, handle):
            return go.Scatterpolar(
                r=list(tags_count.values()),
                theta=list(tags_count.keys()),
                fill='toself',
                name=handle
            )
        
        time.sleep(1)
        url = 'https://codeforces.com/api/user.status'

        params1 = {'handle': handle1}
        params2 = {'handle': handle2}
        response1 = requests.get(url, params=params1)
        response2 = requests.get(url, params=params2)
        data1 = response1.json()
        data2 = response2.json()


        
        all_tags_counts = []
        max_values = []
        for data, handle in [(data1, handle1), (data2, handle2)]:
            solved = filter_(data['result'], lambda x: x['verdict'] == 'OK')
            tags = group_by(solved, lambda x: x['problem']['tags'][0] if x['problem']['tags'] else 'None')

            tags_count = {}
            for tag, problems in tags.items():
                tags_count[tag] = len(problems)

            sorted_tags_count = dict(sorted(tags_count.items(), key=lambda item: item[1], reverse=True))
            all_tags_counts.append((sorted_tags_count, handle))
            max_values.append(max(sorted_tags_count.values()))

        fig = go.Figure()
        for tags_count, handle in all_tags_counts:
            fig.add_trace(generate_scatterpolar(tags_count, handle))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max_values)]
                )
            ),
            showlegend=True,
            title='Problems Solved by Tag'
        )

        return fig


    return {}



@app.callback(
    Output('difficulty-graph', 'figure'),
    Input('submit-button', 'n_clicks'),
    [dash.dependencies.State('input-1', 'value'),
    dash.dependencies.State('input-2', 'value'),
    dash.dependencies.State('handle-selection', 'value')]
)
def update_difficulty_graph(n_clicks, handle1, handle2,handle_type):
    global wait
    if n_clicks <= 0:
        return {}
    if handle_type=='one':
        time.sleep(1)
        global data
        solved = filter_(data['result'], lambda x: x['verdict'] == 'OK')
        difficulties = group_by(solved, lambda x: x['problem']['index'][0])


        sorted_difficulties = sorted(difficulties.items())

        figure = {
            'data': [{'x': [d[0] for d in sorted_difficulties],
                      'y': [len(d[1]) for d in sorted_difficulties],
                      'type': 'bar',
                      'marker': {'color': '#FDB813'}
                      }],
            'layout': {'title': 'Problems Solved by Difficulty', 'showlegend': False}
        }

        return figure
    
    else:
        time.sleep(1)
        url = 'https://codeforces.com/api/user.status'

        params1 = {'handle': handle1}
        params2 = {'handle': handle2}
        response1 = requests.get(url, params=params1)
        response2 = requests.get(url, params=params2)
        data1 = response1.json()
        data2 = response2.json()
        # for r in data2['result']:
        #     print(r['verdict'])
        solved1 = filter_(data1['result'], lambda x: x['verdict'] == 'OK')
        solved2 = filter_(data2['result'], lambda x: x['verdict'] == 'OK')
        difficulties1 = group_by(solved1, lambda x: x['problem']['index'][0])
        sorted_difficulties1 = sorted(difficulties1.items())
        
        difficulties2 = group_by(solved2, lambda x: x['problem']['index'][0])
        sorted_difficulties2 = sorted(difficulties2.items())

        # print(sorted_difficulties1)
        trace1 = go.Bar(
            x=[d[0] for d in sorted_difficulties1],
            y=[len(d[1]) for d in sorted_difficulties1],
            name=handle1
        )
        trace2 = go.Bar(
            x=[d[0] for d in sorted_difficulties2],
            y=[len(d[1]) for d in sorted_difficulties2],
            name=handle2
            )
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=1)
        fig.update_layout(barmode='group')

        return fig

    return {}

@app.callback(
    Output('languages-graph', 'figure'),
    Input('submit-button', 'n_clicks'),
    [dash.dependencies.State('input-1', 'value'),
    dash.dependencies.State('input-2', 'value'),
    dash.dependencies.State('handle-selection', 'value')]
)
def update_languages_graph(n_clicks, handle1, handle2, handle_type):
    global wait
    if n_clicks <= 0:
        return {}
    if handle_type=='one':
        time.sleep(1)
        global data
        languages = group_by(data['result'], lambda x: x['programmingLanguage'])
        print([languages])
        figure = {
            'data': [{'labels': list(languages.keys()),
                      'values': [len(languages[l]) for l in languages],
                      'type': 'pie',
                      'hole': 0.4,
                      'marker': {'colors': ['#FDB813','#B5E61D','#5FA99E','#FDB813','#2CA02C','#FFC0CB']},
                      }],
            'layout': {'title': 'Submission Languages', 'showlegend': True, 'depth': 3}
        }

        return figure
    else:
        url = 'https://codeforces.com/api/user.status'
        params1 = {'handle': handle1}
        params2 = {'handle': handle2}
        response1 = requests.get(url, params=params1)
        response2 = requests.get(url, params=params2)
        data1 = response1.json()
        data2 = response2.json()
        
        all_languages_counts = {}
        handles=[handle1, handle2]
        for data, handle in [(data1, handle1), (data2, handle2)]:
            solved = filter_(data['result'], lambda x: x['verdict'] == 'OK')
            for submission in solved:
                language = submission['programmingLanguage']
                if language not in all_languages_counts:
                    all_languages_counts[language] = {handle: 0 for handle in handles}
                all_languages_counts[language][handle] += 1

        languages = list(all_languages_counts.keys())
        handle_counts = {handle: [] for handle in handles}
        for handle in handles:
            handle_counts[handle] = [all_languages_counts[language][handle] if handle in all_languages_counts[language] else 0 for language in languages]

        fig = go.Figure()
        for handle in handles:
            fig.add_trace(go.Bar(
                x=languages,
                y=handle_counts[handle],
                name=handle
            ))

        fig.update_layout(
            xaxis=dict(title='Languages'),
            yaxis=dict(title='Number of Submissions'),
            title='Submissions by Language',
            barmode='group'
        )

        return fig

    

    return {}

@app.callback(
    dash.dependencies.Output('verdicts-graph', 'figure'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-1', 'value'),
     dash.dependencies.State('input-2', 'value'),
     dash.dependencies.State('handle-selection', 'value')]
)
def update_rating_graph(n_clicks, handle1, handle2, handle_type):
    url = 'https://codeforces.com/api/user.status'
    if handle_type == 'one':
        params1 = {'handle': handle1}
        response1 = requests.get(url, params=params1)
        data1 = response1.json()
        # df1 = pd.DataFrame(data1['result'])
        # print(df1.info())
        verdicts1 = group_by(data1['result'], 'verdict')
        trace1 = go.Bar(
            x=list(verdicts1.keys()),
            y=[len(verdicts1[v]) for v in verdicts1],
            name=handle1
        )
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(trace1, row=1, col=1)
        fig.update_layout(barmode='group')

    else:
        params1 = {'handle': handle1}
        params2 = {'handle': handle2}
        response1 = requests.get(url, params=params1)
        response2 = requests.get(url, params=params2)
        data1 = response1.json()
        data2 = response2.json()
        
        verdicts1 = group_by(data1['result'], 'verdict')
        verdicts2 = group_by(data2['result'], 'verdict')
        trace1 = go.Bar(
            x=list(verdicts1.keys()),
            y=[len(verdicts1[v]) for v in verdicts1],
            name=handle1
        )
        trace2 = go.Bar(x=list(verdicts2.keys()),
        y=[len(verdicts2[v]) for v in verdicts2],
        name=handle2
        )
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=1)
        fig.update_layout(barmode='group')
    
    return fig

@app.callback(
    dash.dependencies.Output('contests-graph', 'figure'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-1', 'value'),
     dash.dependencies.State('input-2', 'value'),
     dash.dependencies.State('handle-selection', 'value')]
)
def update_rating_graph(n_clicks, handle1, handle2, handle_type):
    url = 'https://codeforces.com/api/user.status'
    if handle_type == 'one':
        params1 = {'handle': handle1}
        response1 = requests.get(url, params=params1)
        data1 = response1.json()        
        contests1 = group_by(data1['result'], lambda x: (x['contestId'], x.get('name', 'N/A')))
        total_contests1 = len(contests1.keys())
        trace3 = go.Bar(
            x=[handle1],
            y=[total_contests1],
            name="Total Contests"
        )
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(trace3, row=1, col=1)
        fig.update_layout(barmode='group')

    else:
        params1 = {'handle': handle1}
        params2 = {'handle': handle2}
        response1 = requests.get(url, params=params1)
        response2 = requests.get(url, params=params2)
        data1 = response1.json()
        data2 = response2.json()
        contests1 = group_by(data1['result'], lambda x: (x['contestId'], x.get('name', 'N/A')))
        total_contests1 = len(contests1.keys())
        trace3 = go.Bar(
            x=[handle1],
            y=[total_contests1],
            name="Total Contests"
        )
        contests2 = group_by(data2['result'], lambda x: (x['contestId'], x.get('name', 'N/A')))
        total_contests2 = len(contests2.keys())
        trace5 = go.Bar(
            x=[handle2],
            y=[total_contests2],
            name="Total Contests"
        )
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(trace3, row=1, col=1)
        fig.add_trace(trace5, row=1, col=1)
        fig.update_layout(barmode='group')

    return fig

@app.callback(
    dash.dependencies.Output('unsolved-graph', 'figure'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-1', 'value'),
     dash.dependencies.State('input-2', 'value'),
     dash.dependencies.State('handle-selection', 'value')]
)
def update_rating_graph(n_clicks, handle1, handle2, handle_type):
    url = 'https://codeforces.com/api/user.status'
    if handle_type == 'one':
        params1 = {'handle': handle1}
        response1 = requests.get(url, params=params1)
        data1 = response1.json()
        unsolved1 = [r['problem']['name'] for r in data1['result'] if r['verdict'] != 'OK']
        unsolved1_count = Counter(unsolved1)

        x1 = list(unsolved1_count.keys())
        y1 = [unsolved1_count[k] for k in x1]

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Bar(x=[handle1], y=[sum(y1)], name=handle1), row=1, col=1)
        fig.update_layout(title_text='Unsolved Questions', xaxis_title='Player Handle', yaxis_title='Number of Unsolved Questions')
    else:
        params1 = {'handle': handle1}
        params2 = {'handle': handle2}
        response1 = requests.get(url, params=params1)
        response2 = requests.get(url, params=params2)
        data1 = response1.json()
        data2 = response2.json()
        unsolved1 = [r['problem']['name'] for r in data1['result'] if r['verdict'] != 'OK']
        unsolved2 = [r['problem']['name'] for r in data2['result'] if r['verdict'] != 'OK']
        unsolved1_count = Counter(unsolved1)
        unsolved2_count = Counter(unsolved2)

        x1 = list(unsolved1_count.keys())
        y1 = [unsolved1_count[k] for k in x1]

        x2 = list(unsolved2_count.keys())
        y2 = [unsolved2_count[k] for k in x2]

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Bar(x=[handle1], y=[sum(y1)], name=handle1), row=1, col=1)
        fig.add_trace(go.Bar(x=[handle2], y=[sum(y2)], name=handle2), row=1, col=1)
        fig.update_layout(title_text='Unsolved Questions', xaxis_title='Player Handle', yaxis_title='Number of Unsolved Questions')

    return fig

if __name__ == '__main__':
    app.run(debug=True)