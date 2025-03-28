# import dash IO and graph objects
from dash.dependencies import Input, Output, State
# Plotly graph objects to render graph plots
import plotly.express as px
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dash_bootstrap_templates import ThemeSwitchAIO


from view.app import app
from view.data import features_list

template_theme1 = "zephyr"
template_theme2 = "cyborg"

p = os.path.dirname(os.path.realpath('__file__'))
df=pd.read_csv(str(p)+'\\outs\\bysentence.csv', low_memory=False)

correlation = df.infer_objects()
auxdf=pd.DataFrame.from_dict(features_list)
for c in correlation.columns:
    if c not in list(auxdf['value']):
        correlation.drop(c, axis=1, inplace=True)

cor = correlation.corr(method='spearman')
correlation = correlation.corr(method='pearson')
fselection=[]

def window(length, size=2, start=0):
    while start + size <= length:
        yield slice(start, start + size)
        start += 1

@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('levels1-dropdown', 'value'),
    Input('levels2-dropdown', 'value'),
    Input('method-dropdown', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def update_figure3(x,y,z,toggle):
    fselection.clear()
    if x == 'Sílabas fonéticas (3)' or y == 'Sílabas fonéticas (3)':
        fselection.extend(['repSil_2_3', 'repSil_3_4', 'repSil_4_4'])

    if x == 'Sent. Métricas (3)' or y == 'Sent. Métricas (3)':
        fselection.extend(['versos_sc', 'versos_is', 'versos_fs'])

    if x == 'Contagem de Palavras (1)' or y == 'Contagem de Palavras (1)':
        fselection.extend(['wordCount'])

    if x == 'Unicidade de palavras (4)' or y == 'Unicidade de palavras (4)':
        fselection.extend(['palavrasUnicas', 'lemasUnicos', 'ttrPalavras', 'ttrLemas'])

    if x == 'POS-Tag (11)' or y == 'POS-Tag (11)':
        fselection.extend(['adj', 'adv', 'art', 'conjs', 'in', 'num', 'punc', 'prons', 'prp', 'noun', 'verbs'])

    if x == 'Análise de Sentimentos (11)' or y == 'Análise de Sentimentos (11)':
        fselection.extend(
            ['positivos', 'negativos', 'neutros', 'alegria', 'confianca', 'expectativa', 'medo', 'nojo', 'raiva',
             'surpresa', 'tristeza'])

    if x == 'POS+Sentimentos (4)' or y == 'POS+Sentimentos (4)':
        fselection.extend(['adjpos', 'adjneg', 'advpos', 'advneg'])

    if x == 'Entidade Nomeada (4)' or y == 'Entidade Nomeada (4)':
        fselection.extend(['nePessoa', 'neLocal', 'nePessoaELocal', 'neGeral'])

    if x == 'Polaridades e Cargas Emocionais (6)' or y == 'Polaridades e Cargas Emocionais (6)':
        fselection.extend(['polaridade', 'adjPol', 'advPol', 'adjCharge', 'advCharge', 'emoCharge'])

    if x == 'Frequências de POS (11)' or y == 'Frequências de POS (11)':
        fselection.extend(['adjFreq', 'advFreq', 'artFreq', 'conjFreq', 'inFreq', 'numFreq',
                           'puncFreq', 'pronFreq', 'prpFreq', 'nounFreq', 'verbFreq'])

    if x == 'Frequências de Sentimentos (11)' or y == 'Frequências de Sentimentos (11)':
        fselection.extend(['posFreq', 'negFreq', 'neuFreq', 'alegriaFreq', 'confiancaFreq', 'expectativaFreq',
                           'medoFreq', 'nojoFreq', 'raivaFreq', 'surpresaFreq', 'tristezaFreq'])

    if x == 'Frequências de Entidades Nomeadas (4)' or y == 'Frequências de Entidades Nomeadas (4)':
        fselection.extend(['pessoaFreq', 'localFreq', 'pes_locFreq', 'NEFreq'])

    correlation = df[fselection].corr(method=z)

    if z=='pearson':
        zz='Pearson'
    else:
        zz='Spearman'

    fig = px.imshow(correlation, title=f'Matriz de correlações entre {x} e {y} utilizando {zz}', text_auto=True, aspect="auto",
                    template=template_theme1 if toggle else template_theme2)
    fig.update_layout(
        font=dict(
            family="Arial",
            size=12,
            color="Black") if toggle else None
    )

    return fig

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('feature1-dropdown', 'value'),
    Input('feature2-dropdown', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def update_figure4(x,y,toggle):
    fig = px.scatter(df, x=x, y=y,title=f'Gráfico de dispersão entre {x} e {y}',
                      template=template_theme1 if toggle else template_theme2)
    fig.update_layout(
        font=dict(
            family="Arial",
            size=12,
            color="Black") if toggle else None
    )
    fig.update_xaxes(gridcolor='Gainsboro' if toggle else None)
    fig.update_yaxes(gridcolor='Gainsboro' if toggle else None)
    return fig

@app.callback(
    Output('rolling-line1', 'figure'),
    Input('feature1-dropdown', 'value'),
    Input('feature2-dropdown', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def update_figure5(x,y,toggle):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(y=df[x], name=x),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=df[y], name=y),
        secondary_y=True,
    )
    fig.update_layout(
        title_text=f'Características {x} e {y} ao longo do livro',
        font=dict(
            family="Arial",
            size=12,
            color="Black") if toggle else None,
        template=template_theme1 if toggle else template_theme2
    )
    fig.update_xaxes(title_text='UT', gridcolor='Gainsboro' if toggle else None)
    fig.update_yaxes(title_text=x, secondary_y=False, gridcolor='Gainsboro' if toggle else None)
    fig.update_yaxes(title_text=y, secondary_y=True, gridcolor='Gainsboro' if toggle else None)
    return fig

@app.callback(
    Output('rolling-line2', 'figure'),
    Input('feature1-dropdown', 'value'),
    Input('feature2-dropdown', 'value'),
    Input('methodrolling-dropdown', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def update_figure6(x,y,z,toggle):
    if z=='pearson':
        rolling_r = df[x].rolling(50).corr(df[y])
        fig= px.line(rolling_r,
                       title=f'Correlação deslizante em Pearson das características {x} e {y} ao longo do livro',
                       template=template_theme1 if toggle else template_theme2)
        fig.update_layout(
            font=dict(
                family="Arial",
                size=12,
                color="Black") if toggle else None
        )
        fig.update_xaxes(title_text='Janela', gridcolor='Gainsboro' if toggle else None)
        fig.update_yaxes(title_text='Coeficiente de Correlação', gridcolor='Gainsboro' if toggle else None)
        return fig
    elif z=='spearman':
        rolling_r = []
        for w in window(len(df), size=50):
            df_win = df.iloc[w, :]
            rolling_r.append(df_win[x].rank().corr(df_win[y].rank()))
        aux = [np.nan] * (50 - 1)
        rolling_r = aux + rolling_r
        fig= px.line(rolling_r,
                       title=f'Correlação deslizante em Spearman das características {x} e {y} ao longo do livro',
                       template=template_theme1 if toggle else template_theme2)
        fig.update_layout(
            font=dict(
                family="Arial",
                size=12,
                color="Black") if toggle else None
        )
        fig.update_xaxes(title_text='Janela', gridcolor='Gainsboro' if toggle else None)
        fig.update_yaxes(title_text='Coeficiente de Correlação', gridcolor='Gainsboro' if toggle else None)
        return fig
    else:
        rolling_r = df[x].rolling(50).corr(df[y])
        fig = px.line(rolling_r,
                       title=f'Correlação deslizante em Pearson das características {x} e {y} ao longo do livro',
                       template=template_theme1 if toggle else template_theme2)
        fig.update_layout(
            font=dict(
                family="Arial",
                size=12,
                color="Black") if toggle else None
        )
        fig.update_xaxes(title_text='Janela', gridcolor='Gainsboro' if toggle else None)
        fig.update_yaxes(title_text='Coeficiente de Correlação', gridcolor='Gainsboro' if toggle else None)
        return fig