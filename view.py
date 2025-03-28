from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# Dash Bootstrap components
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO


# layouts and custom callbacks
from view.layouts import statisticsLayout, correlationLayout, correlationsenLayout, parallelismLayout
from view import statistic_analysis, correlation_analysis, correlationsen_analysis, parallelism

# Import app
from view.app import app

import pandas as pd
import os

p = os.path.dirname(os.path.realpath('__file__'))
cor=pd.read_csv(str(p)+'\\outs\\correlations.csv', low_memory=False)
ut=cor['tamanhoUT'].max()

if ut>1:
    nav=[dbc.NavLink("Paralelismos", href="/parallelism", active="exact"),
         html.Hr(),
         dbc.NavLink("Estatística Descritiva", href="/statistics", active="exact"),
         html.Hr(),
         dbc.NavLink("Análise de Correlação tamanho UT=1", href="/correlationsen", active="exact"),
         html.Hr(),
         dbc.NavLink(f"Análise de Correlação tamanho UT={ut}", href="/correlation", active="exact"),]
else:
    nav = [dbc.NavLink("Paralelismos", href="/parallelism", active="exact"),
           html.Hr(),
           dbc.NavLink("Estatística Descritiva", href="/statistics", active="exact"),
           html.Hr(),
           dbc.NavLink("Análise de Correlação UT=1", href="/correlationsen", active="exact"),]


SIDEBAR_STYLE = {
    "position": "absolute",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.A(href="/", children=[
            html.Img(src=app.get_asset_url('logo.png'), alt="Sobre", width='250'),]),
        dbc.Col(ThemeSwitchAIO(aio_id="theme", themes=[dbc.themes.DARKLY, dbc.themes.DARKLY])),
        html.Hr(),
        dbc.Nav(
            nav,
            vertical=True,
            pills=True,
        ),
        html.Hr(),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)


app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == '/':
        return html.Div([dcc.Markdown('''
            ### Sobre
            ---
            Visualização dos resultados obtidos utilizando o CORLI para a identificação de paralelismos literários.
        '''),dcc.Markdown('''
            Pesquisa de dissertação de mestrado desenvolvida por Luciano Alves Machado Júnior sob orientação do professor
            Dr. Angelo Conrado Loula.
        '''),dcc.Markdown('''
            Programa de Pós-Graduação em Ciência da Computação (PGCC) da Universidade Estadual de Feira de Santana (UEFS).
        '''),dcc.Markdown('''
            Última atualização: 09/2023
        ''')],)
    elif pathname == '/statistics':
        return statisticsLayout
    elif pathname == '/parallelism':
        return parallelismLayout
    elif pathname == '/correlationsen':
        return correlationsenLayout
    elif pathname == '/correlation':
        return correlationLayout
    else:
        # If the user tries to reach a different page, return a 404 message
        return dbc.Card(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )

# Call app server
if __name__ == '__main__':
    # set debug to false when deploying app
    server = app.server
    app.run_server(debug=False)
