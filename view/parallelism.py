from dash.dependencies import Input, Output, State

import os
import pandas as pd
from view.app import app
import ast
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html
from dash_bootstrap_templates import ThemeSwitchAIO

template_theme1 = "zephyr"
template_theme2 = "cyborg"

p = os.path.dirname(os.path.realpath('__file__'))
df=pd.read_csv(str(p)+'\\outs\\correlations.csv', low_memory=False)
sentences=pd.read_csv(str(p)+'\\outs\\sentences.csv', low_memory=False)
cor=pd.read_csv(str(p)+'\\outs\\dfup.csv')
ut=df['tamanhoUT'].max()
janela=df['janela'].max()

df.drop(columns=['tamanhoUT','janela','limiar'], inplace=True)
dfs=df
x=''
y=''

def window(length, size=2, start=0):
    while start + size <= length:
        yield slice(start, start + size)
        start += 1

@app.callback(
    [Output('parallel_table', 'data'),Output('parallel_table','columns'),
     Output('container-button-search', 'children')],
    Input('parallel-dropdown', 'value'),
)
def update_output(value):
    msg=''
    global dfs
    if value==None or value=='Todas':
        dfs=df
        dfs=dfs.reset_index()
        msg='Selecione a característica para filtrar'
    else:
        dfs= df[(df.x == value) | (df.y == value)]
        dfs = dfs.reset_index()
        if len(dfs)==0:
            msg='Não encontrado'
            dfs = df
            dfs = dfs.reset_index()
    colunas = [{"name": i, "id": i} for i in dfs.columns]
    colunas.remove({"name": 'index', "id": 'index'})
    colunas.remove({"name": 'pontos', "id": 'pontos'})
    return dfs.to_dict('records'), colunas, msg

@app.callback(
    Output('scatter-plot-parallel','figure'),
    Output('container-parallel', 'children'),
    Output('lines-parallel','figure'),
    Input('pontos-dropdown','value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def pontos(p,toggle):
    c = []
    c.append(html.P(children=''))
    if p != None:
        scatter = go.Figure()
        scatter.add_trace(go.Scatter(x=cor[x].iloc[p - (janela-1):p + 1], y=cor[y].iloc[p - (janela-1):p + 1],
                                         mode='markers', text=f'{x},{y}'))
        scatter.update_layout(
            title_text=f'Dispersão entre {x} e {y} ao longo da janela {p}',
            font=dict(
                family="Arial",
                size=12,
                color="Black") if toggle else None,
            template=template_theme1 if toggle else template_theme2
        )
        scatter.update_xaxes(title_text=x, gridcolor='Gainsboro' if toggle else None)
        scatter.update_yaxes(title_text=y, gridcolor='Gainsboro' if toggle else None)

        if (p + 1) * ut > len(sentences['palavra']):
            li = len(sentences['palavra'])
        else:
            li = (p + 1) * ut
        c.extend([html.P(children='Sentenças da Janela nº ' + str(p) + ': ' + str((p-(janela-1)) * ut) + ' a ' + str(li-1))])


        for e in range(p-(janela-1),p+1):
            if (e + 1) * ut > len(sentences['palavra']):
                lim = len(sentences['palavra'])
            else:
                lim = (e + 1) * ut
            c.extend([html.P(children='Sentenças da UT nº ' + str(e) + ': ' + str(e*ut) + ' a ' + str(lim-1))])
            text = ''
            for i in range(e * ut, lim):
                text += str(sentences['palavra'][i])
            c.extend([html.P(children=text)])
            c.extend([html.Br(), html.Br()])

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=[i for i in range(p - (janela-1), p + 1)], y=cor[x].iloc[p - (janela-1):p + 1], name=x),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=[i for i in range(p - (janela-1), p + 1)], y=cor[y].iloc[p - (janela-1):p + 1], name=y),
            secondary_y=True,
        )
        fig.update_layout(
            title_text=f'Características {x} e {y} ao longo da janela {p}',
            font=dict(
                family="Arial",
                size=12,
                color="Black") if toggle else None,
            template=template_theme1 if toggle else template_theme2
        )
        fig.update_xaxes(title_text='UT', gridcolor='Gainsboro' if toggle else None)
        fig.update_yaxes(title_text=x, secondary_y=False, gridcolor='Gainsboro' if toggle else None)
        fig.update_yaxes(title_text=y, secondary_y=True, gridcolor='Gainsboro' if toggle else None)

        return scatter, c, fig
    else:
        return go.Figure(), c, go.Figure()


@app.callback(
    Output('pontos-dropdown','options'),
    Output('pontos-dropdown','value'),
    Output('rolling-line-parallel','figure'),
    Input('parallel_table','active_cell'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graphs(active_cell,toggle):
    if active_cell:
        global x
        global dfs
        x=dfs["x"][active_cell["row"]]
        global y
        y=dfs["y"][active_cell["row"]]
        z=dfs["metodo"][active_cell["row"]]
        p = dfs['pontos'].apply(lambda x: ast.literal_eval(x))
        p = p[active_cell["row"]]

        line = go.Figure()
        if z=='pearson':
            rolling_r = cor[x].rolling(janela).corr(cor[y])
            line.add_trace(go.Scatter(y=rolling_r, name='corr'))
            line.update_layout(
                title=f'Correlação deslizante de Pearson das características {x} e {y} ao longo do livro',
                font=dict(
                    family="Arial",
                    size=12,
                    color="Black") if toggle else None,
            template=template_theme1 if toggle else template_theme2)
        else:
            rolling_r = []
            for w in window(len(cor), size=janela):
                df_win = cor.iloc[w, :]
                rolling_r.append(df_win[x].rank().corr(df_win[y].rank()))
            aux = [np.nan] * (janela - 1)
            rolling_r = aux + rolling_r
            line.add_trace(go.Scatter(y=rolling_r, name='corr'))
            line.update_layout(
                title=f'Correlação deslizante de Spearman das características {x} e {y} ao longo do livro',
                font=dict(
                    family="Arial",
                    size=12,
                    color="Black") if toggle else None,
            template=template_theme1 if toggle else template_theme2)

        pon=[]
        for i in p:
            pon.append(rolling_r[i])

        line.add_trace(go.Scatter(x=p, y=pon, mode='markers', name='pontos'))
        line.update_xaxes(title_text='Janela', gridcolor='Gainsboro' if toggle else None)
        line.update_yaxes(title_text='Coeficiente de Correlação', gridcolor='Gainsboro' if toggle else None)
        return p, p[0], line
    else:
        return [], None, go.Figure()
