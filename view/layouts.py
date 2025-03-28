# Dash components, html, and dash tables
from dash import dcc
from dash import html
from dash import dash_table
import pandas as pd
import os

# Import Bootstrap components
import dash_bootstrap_components as dbc

from view.data import features_list, correlation_list, correlation_listup, method_list, features_listup

p = os.path.dirname(os.path.realpath('__file__'))
cor=pd.read_csv(str(p)+'\\outs\\correlations.csv', low_memory=False)
ut=cor['tamanhoUT'].max()
janela=cor['janela'].max()
limiar=cor['limiar'].max()

pares=len(cor)
pontos=cor['qntPontos'].sum()

config = {
  'toImageButtonOptions': {
    'format': 'svg'
  }
}

if ut > 1:
    optionsCor = correlation_listup
    valueCorA = correlation_listup[0]['value']
    valueCorB = correlation_listup[1]['value']
    optionsFeatures = features_listup
    valueFeatA = features_listup[0]['value']
    valueFeatB = features_listup[1]['value']
    p=html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'}, children=f'Tamanho UT = {ut}')
    graph=dcc.Graph(id='features-line', config=config)
    button=[
        html.Div(dcc.Input(id='inputut-on-submit', type='text')),
        html.Button('Escolher', id='submit-valut'),
        html.Div(id='container-button-ut',
                 children='Digite o índice da UT que deseja visualizar e pressione o botão Escolher')
    ]
else:
    optionsCor = correlation_list
    valueCorA = correlation_list[0]['value']
    valueCorB = correlation_list[1]['value']
    optionsFeatures = features_list
    valueFeatA = features_list[0]['value']
    valueFeatB = features_list[1]['value']
    p = html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'}, children='')
    graph = html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'}, children='')
    button = html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'}, children='')

optionsFeaturesParal = [{'label': 'Todas','value': 'Todas'}]
optionsFeaturesParal.extend(optionsFeatures)

statisticsLayout = html.Div([
    dbc.Row(dbc.Col(html.H2(style={'text-align': 'center'}, children='Estatística Descritiva das características'))),
    html.Hr(),
    dbc.Row(dbc.Col(html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children='''Essa análise será realizada ao longo do livro'''))),
    html.Br(),
    html.Hr(),
    # Features list dropdown
    dbc.Row([
        dbc.Col(html.H5(style={'text-align': 'center', 'padding-right': '1em'}, children='Selecione a Característica:'),
                xs={'size': 5, 'offset': 0}, sm={'size': 5, 'offset': 0}, md={'size': 5, 'offset': 0},
                lg={'size': 5, 'offset': 0}, xl={'size': 5, 'offset': 0}),
        dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em','color':'#000000'},
            id='features-dropdown',
            options=features_list,
            value=features_list[0]['value'],
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0})
    ]),
    html.Hr(),

    html.Br(),
    dbc.Row(dbc.Col(html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children='''Tamanho UT=1'''))),
    html.Br(),
    dbc.Row(dbc.Col(dcc.Graph(id='features-linesent', config=config), xs={'size': 12, 'offset': 0},
                    sm={'size': 12, 'offset': 0}, md={'size': 12, 'offset': 0}, lg={'size': 12, 'offset': 0})),
    html.Br(),
    html.Div([
        html.Div(id='container-sentenca',
                 children='')
    ]),
    html.Br(),
    html.Div([
        html.Div(dcc.Input(id='input-on-submit', type='text')),
        html.Button('Escolher', id='submit-val'),
        html.Div(id='container-button-basic',
                 children='Digite o índice da sentença que deseja visualizar e pressione o botão Escolher')
    ]),
    html.Hr(),
    dbc.Row(dbc.Col(p)),
    html.Br(),
    dbc.Row(dbc.Col(graph, xs={'size': 12, 'offset': 0},
                    sm={'size': 12, 'offset': 0}, md={'size': 12, 'offset': 0}, lg={'size': 12, 'offset': 0})),
    html.Div([
        html.Div(id='container-ut',
                 children='')
    ]),
    html.Br(),
    html.Div(button),

])


parallelismLayout = html.Div([
    dbc.Row(dbc.Col(html.H2(style={'text-align': 'center'}, children='Identificação de Paralelismos'),
                    )),
    dbc.Row(dbc.Col([html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                            children='Para a identificação dos paralelismos foram utilizados os coeficientes de correlação.'),
                     html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children=f'''Detecção feita ao longo do livro com tamanho da UT = {ut}'''),
                     html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children=f'''Tamanho da janela para a análise = {janela}'''),
                     html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children=f'''Limiar absoluto de interesse (valor mínimo para a correlação ser considerada
                            um paralelismo) = {limiar}'''),
                     html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children=f'''Quantidade de pares encontrados = {pares}'''),
                     html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children=f'''Quantidade de pontos (paralelismos) encontrados = {pontos}''')
                     ])),
    html.Hr(),
    html.Br(),
    dbc.Row(dbc.Col(html.H4(style={'width': '90%','padding': '8px'},
                            children='''Detecção de Paralelismos'''))),
    html.Br(),
    html.Div([
        dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em', 'color': '#000000'},
            id='parallel-dropdown',
            options=optionsFeaturesParal,
            value='Todas',
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0}),
        html.Div(id='container-button-search',
                 children='Selecione a característica para filtrar'),
    ]),
    dbc.Row(dbc.Col(
        dash_table.DataTable(
            id='parallel_table',
            editable=False,
            fixed_rows={'headers': True},
            page_action='none',
            style_table={
                'width': '100%',
                'minWidth': '75%',
                'rows':'15',
                'overflowY': 'auto',
            },
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white',
                'fontWeight': 'bold',
                'border': '1px solid white',
            },
            style_cell={
                'textAlign': 'center',
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
                'padding': '8px',
            },
            style_data={
                'border': '1px solid white'
            },
        ), xs={'size':12, 'offset':0}, sm={'size':12, 'offset':0}, md={'size':10, 'offset':0}, lg={'size':10, 'offset':0},
                xl={'size':10, 'offset':0}),justify="center"),

    html.Br(),
    html.Hr(),
    html.Br(),
    dbc.Row(dbc.Col(html.H4(style={'margin': 'auto', 'width': '90%'},
                            children='''Visualização do Paralelismo'''))),
    html.Br(),
    dbc.Row(dbc.Col(html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children='''Escolha qualquer célula na tabela acima para visualizar o paralelismo.
                           Após isso, escolha o ponto para visualizar a dispersão da janela em questão'''))),
    html.Br(),
    dbc.Row(dbc.Col(dcc.Graph(id='rolling-line-parallel', config=config), xs={'size': 12, 'offset': 0},
                    sm={'size': 12, 'offset': 0}, md={'size': 12, 'offset': 0}, lg={'size': 12, 'offset': 0})),
    html.Hr(),
    dbc.Row(dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em','color':'#000000'},
            id='pontos-dropdown',
            options=[],
            value=None,
            placeholder="Pontos",
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0})),
    dbc.Row(dbc.Col(dcc.Graph(id='scatter-plot-parallel', config=config), xs={'size': 12, 'offset': 0},
                    sm={'size': 12, 'offset': 0}, md={'size': 12, 'offset': 0}, lg={'size': 12, 'offset': 0})),
    html.Hr(),
    dbc.Row(dbc.Col(dcc.Graph(id='lines-parallel', config=config), xs={'size': 12, 'offset': 0},
                    sm={'size': 12, 'offset': 0}, md={'size': 12, 'offset': 0}, lg={'size': 12, 'offset': 0})),
    html.Hr(),
    dbc.Row(dbc.Col(html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children='''Textos da janelas'''))),
    html.Div([html.Div(id='container-parallel', children='Selecione na lista acima')]),
])


correlationsenLayout = html.Div([
    dbc.Row(dbc.Col(html.H2(style={'text-align': 'center'}, children='Análise de Correlação com UT=1'))),
    dbc.Row(dbc.Col(html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children='''Essa análise irá ser feita ao longo do livro com tamanho da UT=1'''))),
    html.Hr(),
    html.Br(),
    dbc.Row(dbc.Col(html.H4(style={'margin': 'auto', 'width': '90%'},
                            children='''Matriz de Correlação'''))),
    html.Br(),
    dbc.Row(dbc.Col(html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children='''Escolha os níveis e o método para a matriz de correlação abaixo'''))),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H5(style={'text-align': 'right', 'padding-right': '1em'}, children='Selecione o nível 01:'),
                xs={'size': 5, 'offset': 0}, sm={'size': 5, 'offset': 0}, md={'size': 5, 'offset': 0},
                lg={'size': 5, 'offset': 0}, xl={'size': 5, 'offset': 0}),
        dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em','color':'#000000'},
            id='levels1-dropdown',
            options=correlation_list,
            value=correlation_list[0]['value'],
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0}),

        dbc.Col(html.H5(style={'text-align': 'right', 'padding-right': '1em'}, children='Selecione o nível 02:'),
                xs={'size': 5, 'offset': 0}, sm={'size': 5, 'offset': 0}, md={'size': 5, 'offset': 0},
                lg={'size': 5, 'offset': 0}, xl={'size': 5, 'offset': 0}),
        dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em','color':'#000000'},
            id='levels2-dropdown',
            options=correlation_list,
            value=correlation_list[1]['value'],
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0}),

        dbc.Col(html.H5(style={'text-align': 'right', 'padding-right': '1em'}, children='Selecione o método:'),
                xs={'size': 5, 'offset': 0}, sm={'size': 5, 'offset': 0}, md={'size': 5, 'offset': 0},
                lg={'size': 5, 'offset': 0}, xl={'size': 5, 'offset': 0}),
        dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em','color':'#000000'},
            id='method-dropdown',
            options=method_list,
            value=method_list[0]['value'],
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0})
    ]),

    dbc.Row(dbc.Col(dcc.Graph(id='correlation-heatmap', config=config), xs={'size': 12, 'offset': 0},
                    sm={'size': 12, 'offset': 0}, md={'size': 12, 'offset': 0}, lg={'size': 12, 'offset': 0})),
    html.Hr(),
    dbc.Row(dbc.Col(html.H4(style={'margin': 'auto', 'width': '90%'},
                            children='''Visualização da Correlação'''))),
    html.Br(),
    dbc.Row(dbc.Col(html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children='''Escolha as características que deseja correlacionar'''))),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H5(style={'text-align': 'right', 'padding-right': '1em'}, children='Selecione a característica 01:'),
                xs={'size': 5, 'offset': 0}, sm={'size': 5, 'offset': 0}, md={'size': 5, 'offset': 0},
                lg={'size': 5, 'offset': 0}, xl={'size': 5, 'offset': 0}),
        dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em','color':'#000000'},
            id='feature1-dropdown',
            options=features_list,
            value=features_list[0]['value'],
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0}),

        dbc.Col(html.H5(style={'text-align': 'right', 'padding-right': '1em'}, children='Selecione o nível 02:'),
                xs={'size': 5, 'offset': 0}, sm={'size': 5, 'offset': 0}, md={'size': 5, 'offset': 0},
                lg={'size': 5, 'offset': 0}, xl={'size': 5, 'offset': 0}),
        dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em','color':'#000000'},
            id='feature2-dropdown',
            options=features_list,
            value=features_list[1]['value'],
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0}),
    ]),

    dbc.Row(dbc.Col(dcc.Graph(id='scatter-plot', config=config), xs={'size': 12, 'offset': 0},
                    sm={'size': 12, 'offset': 0}, md={'size': 12, 'offset': 0}, lg={'size': 12, 'offset': 0})),
    html.Br(),
    dbc.Row(dbc.Col(dcc.Graph(id='rolling-line1', config=config), xs={'size': 12, 'offset': 0},
                    sm={'size': 12, 'offset': 0}, md={'size': 12, 'offset': 0}, lg={'size': 12, 'offset': 0})),
    html.Br(),

    dbc.Row(dbc.Col(html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children='''Escolha o método para a correlação deslizante abaixo'''))),
    html.Br(),
    dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em','color':'#000000'},
            id='methodrolling-dropdown',
            options=method_list,
            value=method_list[0]['value'],
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0}),
    dbc.Row(dbc.Col(dcc.Graph(id='rolling-line2', config=config), xs={'size': 12, 'offset': 0},
                    sm={'size': 12, 'offset': 0}, md={'size': 12, 'offset': 0}, lg={'size': 12, 'offset': 0})),
])


correlationLayout = html.Div([
    dbc.Row(dbc.Col(html.H2(style={'text-align': 'center'}, children='Análise de Correlação'))),
    dbc.Row(dbc.Col(html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children=f'''Essa análise será feita ao longo do livro com tamanho da UT={ut}'''))),
    html.Hr(),
    html.Br(),
    dbc.Row(dbc.Col(html.H4(style={'margin': 'auto', 'width': '90%'},
                            children='''Matriz de Correlação'''))),
    html.Br(),
    dbc.Row(dbc.Col(html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children='''Escolha os níveis e o método para a matriz de correlação abaixo'''))),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H5(style={'text-align': 'right', 'padding-right': '1em'}, children='Selecione o nível 01:'),
                xs={'size': 5, 'offset': 0}, sm={'size': 5, 'offset': 0}, md={'size': 5, 'offset': 0},
                lg={'size': 5, 'offset': 0}, xl={'size': 5, 'offset': 0}),
        dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em','color':'#000000'},
            id='levels1up-dropdown',
            options=optionsCor,
            value=valueCorA,
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0}),

        dbc.Col(html.H5(style={'text-align': 'right', 'padding-right': '1em'}, children='Selecione o nível 02:'),
                xs={'size': 5, 'offset': 0}, sm={'size': 5, 'offset': 0}, md={'size': 5, 'offset': 0},
                lg={'size': 5, 'offset': 0}, xl={'size': 5, 'offset': 0}),
        dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em','color':'#000000'},
            id='levels2up-dropdown',
            options=optionsCor,
            value=valueCorB,
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0}),

        dbc.Col(html.H5(style={'text-align': 'right', 'padding-right': '1em'}, children='Selecione o método:'),
                xs={'size': 5, 'offset': 0}, sm={'size': 5, 'offset': 0}, md={'size': 5, 'offset': 0},
                lg={'size': 5, 'offset': 0}, xl={'size': 5, 'offset': 0}),
        dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em','color':'#000000'},
            id='methodup-dropdown',
            options=method_list,
            value=method_list[0]['value'],
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0})
    ]),

    dbc.Row(dbc.Col(dcc.Graph(id='correlationup-heatmap', config=config), xs={'size': 12, 'offset': 0},
                    sm={'size': 12, 'offset': 0}, md={'size': 12, 'offset': 0}, lg={'size': 12, 'offset': 0})),
    html.Hr(),
    dbc.Row(dbc.Col(html.H4(style={'margin': 'auto', 'width': '90%'},
                            children='''Visualização da Correlação'''))),
    html.Br(),
    dbc.Row(dbc.Col(html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children='''Escolha as características que deseja correlacionar'''))),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H5(style={'text-align': 'right', 'padding-right': '1em'}, children='Selecione a característica 01:'),
                xs={'size': 5, 'offset': 0}, sm={'size': 5, 'offset': 0}, md={'size': 5, 'offset': 0},
                lg={'size': 5, 'offset': 0}, xl={'size': 5, 'offset': 0}),
        dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em','color':'#000000'},
            id='feature1up-dropdown',
            options=optionsFeatures,
            value=valueFeatA,
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0}),

        dbc.Col(html.H5(style={'text-align': 'right', 'padding-right': '1em'}, children='Selecione o nível 02:'),
                xs={'size': 5, 'offset': 0}, sm={'size': 5, 'offset': 0}, md={'size': 5, 'offset': 0},
                lg={'size': 5, 'offset': 0}, xl={'size': 5, 'offset': 0}),
        dbc.Col(dcc.Dropdown(
            style={'text-align': 'center', 'font-size': '1em', 'width': '25em','color':'#000000'},
            id='feature2up-dropdown',
            options=optionsFeatures,
            value=valueFeatB,
            clearable=False),
            xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
            lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0}),
    ]),

    dbc.Row(dbc.Col(dcc.Graph(id='scatterup-plot', config=config), xs={'size': 12, 'offset': 0},
                    sm={'size': 12, 'offset': 0}, md={'size': 12, 'offset': 0}, lg={'size': 12, 'offset': 0})),
    html.Br(),
    dbc.Row(dbc.Col(dcc.Graph(id='rollingup-line1', config=config), xs={'size': 12, 'offset': 0},
                    sm={'size': 12, 'offset': 0}, md={'size': 12, 'offset': 0}, lg={'size': 12, 'offset': 0})),
    html.Br(),

    dbc.Row(dbc.Col(html.P(style={'font-size': '16px', 'margin': 'auto', 'width': '90%', 'opacity': '70%'},
                           children='''Escolha o método para a correlação deslizante abaixo'''))),
    html.Br(),
    dbc.Col(dcc.Dropdown(
        style={'text-align': 'center', 'font-size': '1em', 'width': '25em', 'color': '#000000'},
        id='methodrollingup-dropdown',
        options=method_list,
        value=method_list[0]['value'],
        clearable=False),
        xs={'size': 3, 'offset': 0}, sm={'size': 3, 'offset': 0}, md={'size': 3, 'offset': 0},
        lg={'size': 3, 'offset': 0}, xl={'size': 3, 'offset': 0}),
    dbc.Row(dbc.Col(dcc.Graph(id='rollingup-line2', config=config), xs={'size': 12, 'offset': 0},
                    sm={'size': 12, 'offset': 0}, md={'size': 12, 'offset': 0}, lg={'size': 12, 'offset': 0})),
])