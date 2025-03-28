# import dash IO and graph objects
from dash.dependencies import Input, Output, State
# Plotly graph objects to render graph plots
import plotly.express as px
import pandas as pd
from dash import html
import os
from dash_bootstrap_templates import ThemeSwitchAIO

# Import app
from view.app import app

template_theme1 = "zephyr"
template_theme2 = "cyborg"

p = os.path.dirname(os.path.realpath('__file__'))
sentenca=pd.read_csv(str(p)+'\\outs\\sentences.csv')
dfsent=pd.read_csv(str(p)+'\\outs\\bysentence.csv',low_memory=False)
df=pd.read_csv(str(p)+'\\outs\\dfup.csv')
cor=pd.read_csv(str(p)+'\\outs\\correlations.csv', low_memory=False)
ut=cor['tamanhoUT'].max()


# Callback to features-line
@app.callback(
    Output('features-linesent', 'figure'),
    Output('container-sentenca', 'children'),
    Input('features-dropdown', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def update_figure(selected_feature,toggle):

    fig = px.line(dfsent[selected_feature],
                  title=f'Variação de {selected_feature} ao longo do livro com UT=1',
                  markers=True,template=template_theme1 if toggle else template_theme2)
    fig.update_layout(
        font=dict(
            family="Arial",
            size=12,
            color="Black") if toggle else None
    )
    fig.update_xaxes(title_text='Sentença', gridcolor='Gainsboro' if toggle else None)
    fig.update_yaxes(title_text=selected_feature, gridcolor='Gainsboro' if toggle else None)
    idx = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    text = dfsent[selected_feature].describe()
    return fig, [idx[i] + ': '+ str(text[i])+' |||| ' for i in range(8)]


@app.callback(
    Output('container-button-basic', 'children'),
    Input('submit-val','n_clicks'),
    State('input-on-submit', 'value')
)
def update_output(n_clicks,value):
    if value!=None and int(value)<len(sentenca) and int(value)>=0:
        return sentenca['palavra'][int(value)]
    else:
        return 'Digite o índice da sentença que deseja visualizar e pressione o botão Escolher'

@app.callback(
    Output('container-button-ut', 'children'),
    Input('submit-valut','n_clicks'),
    State('inputut-on-submit', 'value')
)
def update_output(n_clicks,value):
    c=[]
    if value!=None and int(value)<len(df) and int(value)>=0:
        if (int(value) + 1) * ut > len(sentenca['palavra']):
            lim = len(sentenca['palavra'])
        else:
            lim = (int(value) + 1) * ut
        c.extend([html.P(children='Sentenças da UT nº' + value + ':')])
        text = ''
        for i in range(int(value) * ut, lim):
            text += str(sentenca['palavra'][i])
        c.extend([html.P(children=text)])
        return c
    else:
        return 'Digite o índice da UT que deseja visualizar e pressione o botão Escolher'


@app.callback(
    Output('features-line', 'figure'),
    Output('container-ut', 'children'),
    Input('features-dropdown', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def update_figure2(selected_feature, toggle):
    fig = px.line(df[selected_feature],
                  title=f'Variação de {selected_feature} ao longo do livro (por UT)',
                  markers=True, template=template_theme1 if toggle else template_theme2)
    fig.update_layout(
        font=dict(
            family="Arial",
            size=12,
            color="Black") if toggle else None
    )
    fig.update_xaxes(title_text='UT', gridcolor='Gainsboro' if toggle else None)
    fig.update_yaxes(title_text=selected_feature, gridcolor='Gainsboro' if toggle else None)
    idx=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    text = df[selected_feature].describe()
    return fig, [idx[i] + ': '+ str(text[i])+' |||| ' for i in range(8)]