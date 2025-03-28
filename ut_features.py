import pandas as pd
import numpy as np
import re
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from view.data import levelscor, levels

def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=False))  # deacc=True removes punctuations


def remove_stopwords(texts, stopwords):
        return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in texts]


def format_topics_sentences(ldamodel, corpus, texts):
    # Saída inicial
    sent_topics_df = pd.DataFrame()

    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: x[1], reverse=True)
        # Obtém os tópicos, suas contribuições percentuais e palavras-chave para cada documento
        for j, (topic_num, prop_topic) in enumerate(row):
            wp = ldamodel.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in wp])
            sent_topics_df.loc[i, f"Tópico {j + 1}"] = int(topic_num)
            sent_topics_df.loc[i, f"Perc. Contribuição {j + 1}"] = round(prop_topic, 4)
            sent_topics_df.loc[i, f"Palavras Chave {j + 1}"] = topic_keywords

    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df

def window(length, size=2, start=0):
    while start + size <= length:
        yield slice(start, start + size)
        start += 1

def topic_modeling(windowsize):
    print('Aguarde!!')
    print('Aplicando modelagem de tópicos')
    from nltk.corpus import stopwords
    sentences = pd.read_csv('outs\\sentences.csv')
    sentenceswindow = sentences.groupby(np.arange(len(sentences)) // windowsize).sum()
    stopwords = stopwords.words('portuguese')
    stopwords = list(set(stopwords))
    stopwords.extend(['los', 'las', 'afinal', 'ia', 'ali', 'fizera', 'pag', 'pág',
                      'les', 'mim', 'sra', 'viu', 'tantas', 'todas', 'todos',
                      'tantos', 'outro', 'outra', 'outros', 'outras', 'sobre',
                      'nada', 'certo', 'certa', 'apenas', 'dentro', 'tal', 'iam',
                      'toda', 'disse', 'todo', "cair", 'tão', 'haviam', 'ou',
                      'ainda', 'havia', 'et', 'des', 'logo', 'vai', 'novos',
                      'dali', 'capítulo', 'lá', 'onde', 'sei', 'por', 'que',
                      'nã³s', 'porque', 'porquê', 'então', 'lo', 'sr', 'ter', 'la',
                      'pra', 'ah', 'aí', 'uns', 'pois', 'assim', 'ir', 'vinha',
                      'muita', 'tanta', 'tanto', 'ui', 'muito', 'si', 'quase'])

    stopwords.extend(['ainda', 'alguns', 'antes', 'apenas', 'aqui', 'assim', 'cada',
                      'coisa', 'dar', 'desde', 'deve', 'enquanto', 'então', 'faz',
                      'fazer', 'fez', 'ficou', 'havia', 'maior', 'melhor', 'menos',
                      'mesma', 'nada', 'neste', 'nesse', 'nisto', 'nisso', 'onde',
                      'outra', 'outros', 'outro', 'outras', 'pode', 'podem', 'porque',
                      'porquê', 'qualquer', 'quanto', 'quase', 'quer', 'sido', 'que',
                      'quê', 'tão', 'ter', 'toda', 'todo', 'todas', 'todos', 'tudo',
                      'vai', 'vão', 'vem', 'vêm'])

    data = sentenceswindow.palavra.values.tolist()
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    data_words = list(sent_to_words(data))
    data_words_nostops = remove_stopwords(data_words, stopwords)

    id2word = corpora.Dictionary(data_words_nostops)
    texts = data_words_nostops
    corpus = [id2word.doc2bow(text) for text in texts]

    # Construindo LDA Model
    num_topics = 5
    random_state = 100
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = 1  # para o log possuir info valor deve ser 1
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=random_state,
                                                chunksize=chunksize,
                                                passes=passes,
                                                iterations=iterations,
                                                eval_every=eval_every,
                                                alpha='auto',
                                                eta='auto')

    df_topic_sents_keywords = format_topics_sentences(lda_model, corpus, data)

    dftopicos = df_topic_sents_keywords.reset_index()
    dftopicos.to_csv('outs\\topicos.csv', encoding='utf-8', index=False)

    props = []
    for i in range(len(dftopicos)):
        props.append([0] * num_topics)

    for i, row in enumerate(lda_model[corpus]):
        row = sorted(row, key=lambda x: x[1], reverse=True)
        # Obtém os tópicos, suas contribuições percentuais e palavras-chave para cada documento
        for j, (topic_num, prop_topic) in enumerate(row):
            props[i][topic_num] = prop_topic

    # print(props)
    propTc = pd.DataFrame(props)
    propTc.columns = ["Tópico " + str(i + 1) for i in range(propTc.shape[1])]
    propTc.to_csv('outs\\propTc.csv', encoding='utf-8', index=False)


def ut_organize(ut):
    print('Organizando datasets')
    dfword = pd.read_csv('outs\\dfword.csv', low_memory=False)

    features = []
    for f in ['ADP', 'N', 'adj', 'adv', 'art', 'conj-c', 'conj-s', 'in', 'intj', 'n',
              'n-adj', 'num', 'pron-det', 'pron-indp', 'pron-pers', 'prop', 'prp', 'punc',
              'v-fin', 'v-ger', 'v-inf', 'v-pcp', '-1', '0', '1', 'B-ABSTRACCAO',
              'B-COISA', 'B-LOCAL', 'B-OBRA', 'B-ORGANIZACAO',
              'B-PESSOA', 'B-TEMPO', 'B-VALOR', 'I-ABSTRACCAO',
              'I-LOCAL', 'I-OBRA', 'I-ORGANIZACAO', 'I-PESSOA',
              'I-TEMPO', 'I-VALOR', 'O']:
        if f in dfword.columns:
            features.append(f)

    dfword[features] = dfword[features].notnull().astype('int')

    dfe = pd.read_csv('outs\\basesdf\\emo.csv')

    mivesis = pd.read_csv('outs\\mivesis.csv')
    mivesis.columns += '_is'

    mivesfs = pd.read_csv('outs\\mivesfs.csv')
    mivesfs.columns += '_fs'

    mivessc = pd.read_csv('outs\\mivessc.csv')
    mivessc.columns += '_sc'

    allpro2_3 = pd.read_csv('outs\\allpro2_3.csv')
    allpro2_3.columns += '_2_3'

    allpro3_4 = pd.read_csv('outs\\allpro3_4.csv')
    allpro3_4.columns += '_3_4'

    allpro4_4 = pd.read_csv('outs\\allpro4_4.csv')
    allpro4_4.columns += '_4_4'

    dfword['Polarity'] = dfe['Emotion']

    dict_map = {0: 0, 1: 0, 2: 1}
    dfword['adjpos'] = dfword['adj'] + dfword['1']
    dfword['adjpos'] = dfword['adjpos'].map(dict_map)

    dfword['adjneg'] = dfword['adj'] + dfword['-1']
    dfword['adjneg'] = dfword['adjneg'].map(dict_map)

    dfword['advpos'] = dfword['adv'] + dfword['1']
    dfword['advpos'] = dfword['advpos'].map(dict_map)

    dfword['advneg'] = dfword['adv'] + dfword['-1']
    dfword['advneg'] = dfword['advneg'].map(dict_map)

    dfbysentence = dfword.groupby(['Sentence']).sum()
    dfaux = dfword[['SentenceLength','Sentence']]
    dfaux= dfaux.groupby(['Sentence']).mean()
    dfbysentence['SentenceLength'] = dfaux['SentenceLength']
    dfbysentence = pd.concat([dfbysentence, mivesis, mivesfs, mivessc, allpro2_3, allpro3_4, allpro4_4], axis=1)

    # Pontuação
    pont = pd.read_csv('outs\\basesdf\\pos.csv')
    pont = pont.pivot(columns='POSTag')
    pont = pont['palavra']

    # Freq. Palavra e TTR
    df = pd.read_csv('outs\\basesdf\\pos.csv')
    df['Sentence'] = dfword['Sentence']
    df = df[df.POSTag != 'punc']
    uniquewords = len(df['palavra'].unique())
    df = df.drop('POSTag', axis=1)
    dfs = df.groupby('Sentence').nunique()
    dfbysentence['UniqueW'] = dfs['palavra']
    dfs['SentenceLength'] = dfbysentence['SentenceLength']
    dfs['ttr'] = dfs['palavra'] / dfs['SentenceLength']
    dfbysentence['ttrW'] = dfs['ttr']

    # Freq. Lema e TTR Lema
    dflema = pd.read_csv('outs\\basesdf\\lema.csv')
    df = pd.read_csv('outs\\basesdf\\pos.csv')
    dflema['POSTag'] = df['POSTag']
    dflema['Sentence'] = dfword['Sentence']
    dflema = dflema[dflema.POSTag != 'punc']
    uniquelemas = len(dflema['lema'].unique())
    dflema = dflema.drop(columns=['palavra', 'POSTag'], axis=1)
    dflemas = dflema.groupby('Sentence').nunique()
    dfbysentence['UniqueL'] = dflemas['lema']
    dflemas['SentenceLength'] = dfbysentence['SentenceLength']
    dflemas['ttr'] = dflemas['lema'] / dflemas['SentenceLength']
    dfbysentence['ttrL'] = dflemas['ttr']
    df = df[df.POSTag != 'punc']

    # Polaridades e Cargas Emocionais
    dfbysentence['adjPol'] = dfbysentence['adjpos'] - dfbysentence['adjneg']
    dfbysentence['advPol'] = dfbysentence['advpos'] - dfbysentence['advneg']
    dfbysentence['polDif'] = dfbysentence['1'] - dfbysentence['-1']
    dfbysentence['adjCharge'] = dfbysentence['adjpos'] + dfbysentence['adjneg']
    dfbysentence['advCharge'] = dfbysentence['advpos'] + dfbysentence['advneg']
    dfbysentence['emoCharge'] = dfbysentence['1'] + dfbysentence['-1']

    # Calculos de POS
    adv = len(dfword[(dfword['adv'] == 1)])
    adj = len(dfword[(dfword['adj'] == 1)])
    verb = len(
        dfword[(dfword['v-fin'] == 1) | (dfword['v-ger'] == 1) | (dfword['v-inf'] == 1) | (dfword['v-pcp'] == 1)])
    pron = len(dfword[(dfword['pron-det'] == 1) | (dfword['pron-indp'] == 1) | (dfword['pron-pers'] == 1)])
    conj = len(dfword[(dfword['conj-c'] == 1) | (dfword['conj-s'] == 1)])
    prp = len(dfword[(dfword['prp'] == 1)])
    prop = len(dfword[(dfword['prop'] == 1)])
    noun = len(dfword[(dfword['n'] == 1) | (dfword['N'] == 1)])
    dfbysentence['verbs'] = dfbysentence['v-fin'] + dfbysentence['v-ger'] + dfbysentence['v-inf'] + dfbysentence[
        'v-pcp']
    dfbysentence['prons'] = dfbysentence['pron-det'] + dfbysentence['pron-indp'] + dfbysentence['pron-pers']
    dfbysentence['conjs'] = dfbysentence['conj-c'] + dfbysentence['conj-s']
    dfbysentence['noun'] = dfbysentence['n'] + dfbysentence['N']

    # Cálculos de Entidades
    dfbysentence['NE'] = dfbysentence['B-PESSOA'] + dfbysentence['B-LOCAL']
    dfbysentence['TotalNE'] = dfbysentence['B-ABSTRACCAO'] + dfbysentence['B-COISA'] + dfbysentence['B-LOCAL'] + \
                              dfbysentence['B-OBRA'] + dfbysentence['B-ORGANIZACAO'] + dfbysentence['B-PESSOA'] + \
                              dfbysentence['B-TEMPO'] + dfbysentence['B-VALOR'] + dfbysentence['I-ABSTRACCAO'] + \
                              dfbysentence['I-LOCAL'] + dfbysentence['I-OBRA'] + dfbysentence['I-ORGANIZACAO'] + \
                              dfbysentence['I-PESSOA'] + dfbysentence['I-TEMPO']

    # Frequências (Normalizações)
    #PoS
    dfbysentence['adjFreq'] = dfbysentence['adj'] / dfbysentence['SentenceLength']
    dfbysentence['advFreq'] = dfbysentence['adv'] / dfbysentence['SentenceLength']
    dfbysentence['artFreq'] = dfbysentence['art'] / dfbysentence['SentenceLength']
    dfbysentence['conjFreq'] = dfbysentence['conjs'] / dfbysentence['SentenceLength']
    dfbysentence['inFreq'] = dfbysentence['in'] / dfbysentence['SentenceLength']
    dfbysentence['numFreq'] = dfbysentence['num'] / dfbysentence['SentenceLength']
    dfbysentence['puncFreq'] = dfbysentence['punc'] / dfbysentence['SentenceLength']
    dfbysentence['pronFreq'] = dfbysentence['prons'] / dfbysentence['SentenceLength']
    dfbysentence['prpFreq'] = dfbysentence['prp'] / dfbysentence['SentenceLength']
    dfbysentence['nounFreq'] = dfbysentence['noun'] / dfbysentence['SentenceLength']
    dfbysentence['verbFreq'] = dfbysentence['verbs'] / dfbysentence['SentenceLength']
    #AS
    dfbysentence['posFreq'] = dfbysentence['1'] / dfbysentence['SentenceLength']
    dfbysentence['negFreq'] = dfbysentence['-1'] / dfbysentence['SentenceLength']
    dfbysentence['neuFreq'] = dfbysentence['0'] / dfbysentence['SentenceLength']
    dfbysentence['joyFreq'] = dfbysentence['Joy'] / dfbysentence['SentenceLength']
    dfbysentence['trustFreq'] = dfbysentence['Trust'] / dfbysentence['SentenceLength']
    dfbysentence['anticipationFreq'] = dfbysentence['Anticipation'] / dfbysentence['SentenceLength']
    dfbysentence['fearFreq'] = dfbysentence['Fear'] / dfbysentence['SentenceLength']
    dfbysentence['disgustFreq'] = dfbysentence['Disgust'] / dfbysentence['SentenceLength']
    dfbysentence['angerFreq'] = dfbysentence['Anger'] / dfbysentence['SentenceLength']
    dfbysentence['surpriseFreq'] = dfbysentence['Surprise'] / dfbysentence['SentenceLength']
    dfbysentence['sadnessFreq'] = dfbysentence['Sadness'] / dfbysentence['SentenceLength']
    #NE
    dfbysentence['pessoaFreq'] = dfbysentence['B-PESSOA'] / dfbysentence['SentenceLength']
    dfbysentence['localFreq'] = dfbysentence['B-LOCAL'] / dfbysentence['SentenceLength']
    dfbysentence['pes_locFreq'] = dfbysentence['NE'] / dfbysentence['SentenceLength']
    dfbysentence['NEFreq'] = dfbysentence['TotalNE'] / dfbysentence['SentenceLength']

    dfbysentence.rename(columns={'SentenceLength': 'wordCount', 'UniqueW': 'palavrasUnicas', 'UniqueL': 'lemasUnicos',
                       'ttrW': 'ttrPalavras', 'ttrL': 'ttrLemas', '1': 'positivos', '0': 'neutros', '-1': 'negativos',
                       'Joy': 'alegria', 'Trust': 'confianca', 'Anticipation': 'expectativa', 'Fear': 'medo',
                       'Disgust': 'nojo', 'Anger': 'raiva', 'Surprise': 'surpresa', 'Sadness': 'tristeza',
                       'Polarity': 'polaridade', 'B-PESSOA': 'nePessoa', 'B-LOCAL': 'neLocal', 'NE': 'nePessoaELocal',
                       'TotalNE': 'neGeral', 'joyFreq': 'alegriaFreq', 'trustFreq': 'confiancaFreq',
                       'anticipationFreq': 'expectativaFreq', 'fearFreq': 'medoFreq', 'disgustFreq': 'nojoFreq',
                       'angerFreq': 'raivaFreq', 'surpriseFreq': 'surpresaFreq', 'sadnessFreq': 'tristezaFreq'},
              inplace=True)

    dfbysentence.to_csv('outs\\bysentence.csv', encoding='utf-8', index=False)

    dfup = pd.read_csv('outs\\bysentence.csv', low_memory=False)
    if ut>1:
        propTc = pd.read_csv('outs\\propTc.csv')

    dfup = dfup.infer_objects()
    for c in dfup.columns:
        if c not in ['ADP', 'N', 'adj', 'adv', 'art', 'conj-c', 'conj-s', 'in', 'intj', 'n',
                     'n-adj', 'num', 'pron-det', 'pron-indp', 'pron-pers', 'prop', 'prp',
                     'punc', 'v-fin', 'v-ger', 'v-inf', 'v-pcp', 'negativos', 'neutros', 'positivos',
                     'B-ABSTRACCAO', 'B-COISA', 'neLocal', 'B-OBRA',
                     'B-ORGANIZACAO', 'nePessoa', 'B-TEMPO', 'B-VALOR',
                     'I-ABSTRACCAO', 'I-LOCAL', 'I-OBRA', 'I-ORGANIZACAO',
                     'I-PESSOA', 'I-TEMPO', 'I-VALOR', 'O', 'raiva', 'expectativa',
                     'nojo', 'medo', 'alegria', 'tristeza', 'surpresa', 'confianca',
                     'wordCount', 'polaridade', 'adjpos', 'adjneg', 'advpos', 'advneg',
                     'verbs', 'prons', 'conjs', 'noun', 'nePessoaELocal', 'neGeral', 'repSil_2_3', 'repSil_3_4',
                     'repSil_4_4', 'versos_sc', 'versos_is', 'versos_fs']:
            dfup.drop(c, axis=1, inplace=True)

    dfup = dfup.groupby(dfup.index // ut).sum()
    if ut>1:
        dfup = pd.concat([dfup, propTc], axis=1)

    # MAPEAR A SENTENÇA APÓS A REORGANIZAÇÃO
    sentencemap = []
    for i in dfword['Sentence']:
        sentencemap.append(i // ut)
    dfword.insert(0, 'up', sentencemap)

    # Freq. Palavra e TTR
    dfwup = pd.read_csv('outs\\basesdf\\pos.csv')
    dfwup['up'] = dfword['up']
    dfwup = dfwup[dfwup.POSTag != 'punc']
    dfwup = dfwup.drop('POSTag', axis=1)
    dfsup = dfwup.groupby('up').nunique()
    dfup['palavrasUnicas'] = dfsup['palavra']
    dfsup['wordCount'] = dfup['wordCount']
    dfsup['ttr'] = dfsup['palavra'] / dfsup['wordCount']
    dfup['ttrPalavras'] = dfsup['ttr']

    # Freq. Lema e TTR Lema
    dflemaup = pd.read_csv('outs\\basesdf\\lema.csv')
    dflup = pd.read_csv('outs\\basesdf\\pos.csv')
    dflemaup['POSTag'] = dflup['POSTag']
    dflemaup['up'] = dfword['up']
    dflemaup = dflemaup[dflemaup.POSTag != 'punc']
    dflemaup = dflemaup.drop(columns=['palavra', 'POSTag'], axis=1)
    dflemaups = dflemaup.groupby('up').nunique()
    dfup['lemasUnicos'] = dflemaups['lema']
    dflemaups['wordCount'] = dfup['wordCount']
    dflemaups['ttr'] = dflemaups['lema'] / dflemaups['wordCount']
    dfup['ttrLemas'] = dflemaups['ttr']

    # Polaridades e Cargas Emocionais
    dfup['adjPol'] = dfup['adjpos'] - dfup['adjneg']
    dfup['advPol'] = dfup['advpos'] - dfup['advneg']
    dfup['adjCharge'] = dfup['adjpos'] + dfup['adjneg']
    dfup['advCharge'] = dfup['advpos'] + dfup['advneg']
    dfup['emoCharge'] = dfup['positivos'] + dfup['negativos']

    # Frequências (Normalizações)
    # PoS
    dfup['adjFreq'] = dfup['adj'] / dfup['wordCount']
    dfup['advFreq'] = dfup['adv'] / dfup['wordCount']
    dfup['artFreq'] = dfup['art'] / dfup['wordCount']
    dfup['conjFreq'] = dfup['conjs'] / dfup['wordCount']
    dfup['inFreq'] = dfup['in'] / dfup['wordCount']
    dfup['numFreq'] = dfup['num'] / dfup['wordCount']
    dfup['puncFreq'] = dfup['punc'] / dfup['wordCount']
    dfup['pronFreq'] = dfup['prons'] / dfup['wordCount']
    dfup['prpFreq'] = dfup['prp'] / dfup['wordCount']
    dfup['nounFreq'] = dfup['noun'] / dfup['wordCount']
    dfup['verbFreq'] = dfup['verbs'] / dfup['wordCount']
    # AS
    dfup['posFreq'] = dfup['positivos'] / dfup['wordCount']
    dfup['negFreq'] = dfup['negativos'] / dfup['wordCount']
    dfup['neuFreq'] = dfup['neutros'] / dfup['wordCount']
    dfup['alegriaFreq'] = dfup['alegria'] / dfup['wordCount']
    dfup['confiancaFreq'] = dfup['confianca'] / dfup['wordCount']
    dfup['expectativaFreq'] = dfup['expectativa'] / dfup['wordCount']
    dfup['medoFreq'] = dfup['medo'] / dfup['wordCount']
    dfup['nojoFreq'] = dfup['nojo'] / dfup['wordCount']
    dfup['raivaFreq'] = dfup['raiva'] / dfup['wordCount']
    dfup['surpresaFreq'] = dfup['surpresa'] / dfup['wordCount']
    dfup['tristezaFreq'] = dfup['tristeza'] / dfup['wordCount']
    # NE
    dfup['pessoaFreq'] = dfup['nePessoa'] / dfup['wordCount']
    dfup['localFreq'] = dfup['neLocal'] / dfup['wordCount']
    dfup['pes_locFreq'] = dfup['nePessoaELocal'] / dfup['wordCount']
    dfup['NEFreq'] = dfup['neGeral'] / dfup['wordCount']
    dfup.to_csv('outs\\dfup.csv', encoding='utf-8', index=False)

# A variável 'limiar' determina o limiar ABSOLUTO de interesse. Por padrão ela está em 0.75, significando que
# correlações acima de 0.75 ou abaixo de -0.80 serão consideradas pontos paralelos no livro.
# Correlações de mesmo nível não serão consideradas, além disso, correlações entre níveis de freq absoluto e sua
# freq relativo também não serão consideradas.
# A variável 'janela' determina o tamanho da janela considerada para o rolling, com base na quantidade de
# UTs do livro, por padrão esse valor é de 10. O que significa que a cada 10 UTs será feita uma
# correlação entre os pontos
def paralelismDetection(ut, limiar=0.75, janela=10):
    print('Buscando paralelismos')
    limiar=abs(limiar)
    df=pd.read_csv('outs\\dfup.csv', low_memory=False)
    features = []
    if janela < 3:
        print('''O mínimo permitido para a janela é de 3 já que valores abaixo disso 
        podem não demonstrar correlações coerentes''')
        janela = 3

    if janela > 100:
        print('''O máximo permitido para a janela é de 100 já que valores acima disso 
                podem não mostrar correlações coerentes''')
        janela = 100

    if ut>1:
        level=levelscor
    else:
        level=levels
    for i in level.keys():
        for j in level.keys():
            if not (set(level[i]).issubset(set(level[j])) or set(level[j]).issubset(set(level[i]))):
                if 'repSil' in level[i]:
                    features.append((i, j))
                elif 'verso' in level[i]:
                    if 'repSil' not in level[j]:
                        features.append((i, j))
                elif 'cont' in level[i]:
                    if all(x not in level[j] for x in ['repSil', 'verso']):
                        features.append((i, j))
                elif 'sent' in level[i] and 'pos' in level[i]:
                    if all(x not in level[j] for x in ['repSil', 'verso', 'cont', 'pos', 'sent']):
                        features.append((i, j))
                elif 'pos' in level[i]:
                    if all(x not in level[j] for x in ['repSil', 'verso', 'cont']):
                        features.append((i, j))
                elif 'sent' in level[i]:
                    if all(x not in level[j] for x in ['repSil', 'verso', 'cont', 'pos']):
                        features.append((i, j))
                elif 'ne' in level[i]:
                    if all(x not in level[j] for x in ['repSil', 'verso', 'cont', 'pos', 'sent']):
                        features.append((i, j))
    detected=[]
    for f in features:
        rolling_r = df[f[0]].rolling(janela).corr(df[f[1]])
        rolling_r = rolling_r.dropna()
        if len(rolling_r) > 0:
            if round(max(rolling_r), 2) >= limiar or round(min(rolling_r), 2) <= -limiar:
                detected.append(list(f))
                detected[-1].append('pearson')
                detected[-1].append(round(df[f[0]].corr(df[f[1]]), 4))
                detected[-1].append(round(max(rolling_r), 4))
                detected[-1].append(round(min(rolling_r), 4))
                k = rolling_r.index[abs(round(rolling_r, 2)) >= limiar].tolist()
                detected[-1].append(len(k))
                detected[-1].append(k)
                detected[-1].append(ut)
                detected[-1].append(janela)
                detected[-1].append(limiar)

        rolling_r = []
        for w in window(len(df), size=janela):
            df_win = df.iloc[w, :]
            rolling_r.append(df_win[f[0]].rank().corr(df_win[f[1]].rank()))
        if len(rolling_r) > 0:
            if round(max(rolling_r), 2) >= limiar or round(min(rolling_r), 2) <= -limiar:
                detected.append(list(f))
                detected[-1].append('spearman')
                detected[-1].append(round(df[f[0]].corr(df[f[1]], method='spearman'), 4))
                detected[-1].append(round(max(rolling_r), 4))
                detected[-1].append(round(min(rolling_r), 4))
                indf = pd.Series(rolling_r)
                indf.index = indf.index + (janela - 1)
                k = indf.index[abs(round(indf,2)) >= limiar].tolist()
                detected[-1].append(len(k))
                detected[-1].append(k)
                detected[-1].append(ut)
                detected[-1].append(janela)
                detected[-1].append(limiar)

    correlations=pd.DataFrame(detected)
    correlations.columns =['x', 'y', 'metodo', 'coGeral', 'coMax', 'coMin', 'qntPontos', 'pontos', 'tamanhoUT','janela','limiar']
    correlations.to_csv('outs\\correlations.csv', encoding='utf-8', index=False)

'''if __name__ == '__main__':
    ut = int(input("Insira o tamanho da unidade textual (quant. de sentenças a serem agrupadas para a análise):"))
    if ut>1:
        if ut<50:
            print("Atenção! Para a modelagem de tópicos é recomendado UT maiores ou iguais a 50")
        topic_modeling(ut)
    else:
        print('Modelagem de tópicos não aplicada pois o tamanho da UT é igual a 1')
    ut_organize(ut)
    paralelismDetection(ut)'''
