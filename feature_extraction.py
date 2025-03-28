from NLPyPort.FullPipeline import *
import os
import shutil
import json
import pandas as pd
from itertools import product
from difflib import SequenceMatcher
from rapidfuzz import process, fuzz

def change_nlpyport():
    confi = load_congif_to_list()

    shutil.copyfile(os.getcwd() + "\\changes\\clitics.xml",
                    str(confi[1]).split("NLPyPort\\")[0] + "NLPyPort\\TokPyPort\\resources\\replacements\\clitics.xml")
    shutil.copyfile(os.getcwd() + "\\changes\\contractions.xml", str(confi[1]).split("NLPyPort\\")[
        0] + "NLPyPort\\TokPyPort\\resources\\replacements\\contractions.xml")


def processTex(input_file, options):
    print("Processando Texto\nAguarde!!\n")
    text = new_full_pipe(input_file, options=options)
    if (text != 0):
        print("Texto Lido!!!")
    return text


def polarities(text):
    liwc = json.load(open('dicts\\liwc_pt.json'))
    sentilex = json.load(open('dicts\\sentilex.json'))
    lexicon = json.load(open('dicts\\lexicon.json'))

    polarity = []

    tokens = text.tokens
    count = -1
    for token in tokens:
        count += 1
        token = token.lower()
        if token in liwc:
            if "posemo" in liwc[token]:
                polarity.append(1)
            elif "negemo" in liwc[token]:
                polarity.append(-1)
            elif text.lemas[count].lower() in liwc:
                if "posemo" in liwc[text.lemas[count].lower()]:
                    polarity.append(1)
                elif "negemo" in liwc[text.lemas[count].lower()]:
                    polarity.append(-1)
                else:
                    polarity.append(0)
            else:
                polarity.append(0)
        elif text.lemas[count].lower() in liwc:
            if "posemo" in liwc[text.lemas[count].lower()]:
                polarity.append(1)
            elif "negemo" in liwc[text.lemas[count].lower()]:
                polarity.append(-1)
            else:
                polarity.append(0)
        elif token in sentilex:
            if sentilex[token] == 1:
                polarity.append(1)
            elif sentilex[token] == -1:
                polarity.append(-1)
            elif text.lemas[count].lower() in sentilex:
                if sentilex[text.lemas[count].lower()] == 1:
                    polarity.append(1)
                elif sentilex[text.lemas[count].lower()] == -1:
                    polarity.append(-1)
                else:
                    polarity.append(0)
            else:
                polarity.append(0)
        elif text.lemas[count].lower() in sentilex:
            if sentilex[text.lemas[count].lower()] == 1:
                polarity.append(1)
            elif sentilex[text.lemas[count].lower()] == -1:
                polarity.append(-1)
            else:
                polarity.append(0)
        elif token in lexicon:
            if lexicon[token] == 1:
                polarity.append(1)
            elif lexicon[token] == -1:
                polarity.append(-1)
            elif text.lemas[count].lower() in lexicon:
                if lexicon[text.lemas[count].lower()] == 1:
                    polarity.append(1)
                elif lexicon[text.lemas[count].lower()] == -1:
                    polarity.append(-1)
                else:
                    polarity.append(0)
            else:
                polarity.append(0)
        elif text.lemas[count].lower() in lexicon:
            if lexicon[text.lemas[count].lower()] == 1:
                polarity.append(1)
            elif lexicon[text.lemas[count].lower()] == -1:
                polarity.append(-1)
            else:
                polarity.append(0)
        else:
            polarity.append(0)

    print("Polaridade ok")
    return polarity


def emotions(text):
    nrcemolex = json.load(open('dicts\\nrcemolex.json', encoding='UTF-8'))

    anger = []
    anticipation = []
    disgust = []
    fear = []
    joy = []
    sadness = []
    surprise = []
    trust = []

    tokens = text.tokens
    count = -1
    for token in tokens:
        count += 1
        token = token.lower()
        if token in nrcemolex:
            if nrcemolex[token]['anger'] == 1:
                anger.append(1)
            elif text.lemas[count].lower() in nrcemolex:
                if nrcemolex[text.lemas[count].lower()]['anger'] == 1:
                    anger.append(1)
                else:
                    anger.append(0)
            else:
                anger.append(0)

            if nrcemolex[token]['anticipation'] == 1:
                anticipation.append(1)
            elif text.lemas[count].lower() in nrcemolex:
                if nrcemolex[text.lemas[count].lower()]['anticipation'] == 1:
                    anticipation.append(1)
                else:
                    anticipation.append(0)
            else:
                anticipation.append(0)

            if nrcemolex[token]['disgust'] == 1:
                disgust.append(1)
            elif text.lemas[count].lower() in nrcemolex:
                if nrcemolex[text.lemas[count].lower()]['disgust'] == 1:
                    disgust.append(1)
                else:
                    disgust.append(0)
            else:
                disgust.append(0)

            if nrcemolex[token]['fear'] == 1:
                fear.append(1)
            elif text.lemas[count].lower() in nrcemolex:
                if nrcemolex[text.lemas[count].lower()]['fear'] == 1:
                    fear.append(1)
                else:
                    fear.append(0)
            else:
                fear.append(0)

            if nrcemolex[token]['joy'] == 1:
                joy.append(1)
            elif text.lemas[count].lower() in nrcemolex:
                if nrcemolex[text.lemas[count].lower()]['joy'] == 1:
                    joy.append(1)
                else:
                    joy.append(0)
            else:
                joy.append(0)

            if nrcemolex[token]['sadness'] == 1:
                sadness.append(1)
            elif text.lemas[count].lower() in nrcemolex:
                if nrcemolex[text.lemas[count].lower()]['sadness'] == 1:
                    sadness.append(1)
                else:
                    sadness.append(0)
            else:
                sadness.append(0)

            if nrcemolex[token]['surprise'] == 1:
                surprise.append(1)
            elif text.lemas[count].lower() in nrcemolex:
                if nrcemolex[text.lemas[count].lower()]['surprise'] == 1:
                    surprise.append(1)
                else:
                    surprise.append(0)
            else:
                surprise.append(0)

            if nrcemolex[token]['trust'] == 1:
                trust.append(1)
            elif text.lemas[count].lower() in nrcemolex:
                if nrcemolex[text.lemas[count].lower()]['trust'] == 1:
                    trust.append(1)
                else:
                    trust.append(0)
            else:
                trust.append(0)

        elif text.lemas[count].lower() in nrcemolex:
            if nrcemolex[text.lemas[count].lower()]['anger'] == 1:
                anger.append(1)
            else:
                anger.append(0)

            if nrcemolex[text.lemas[count].lower()]['anticipation'] == 1:
                anticipation.append(1)
            else:
                anticipation.append(0)

            if nrcemolex[text.lemas[count].lower()]['disgust'] == 1:
                disgust.append(1)
            else:
                disgust.append(0)

            if nrcemolex[text.lemas[count].lower()]['fear'] == 1:
                fear.append(1)
            else:
                fear.append(0)

            if nrcemolex[text.lemas[count].lower()]['joy'] == 1:
                joy.append(1)
            else:
                joy.append(0)

            if nrcemolex[text.lemas[count].lower()]['sadness'] == 1:
                sadness.append(1)
            else:
                sadness.append(0)

            if nrcemolex[text.lemas[count].lower()]['surprise'] == 1:
                surprise.append(1)
            else:
                surprise.append(0)

            if nrcemolex[text.lemas[count].lower()]['trust'] == 1:
                trust.append(1)
            else:
                trust.append(0)
        else:
            anger.append(0)
            anticipation.append(0)
            disgust.append(0)
            fear.append(0)
            joy.append(0)
            sadness.append(0)
            surprise.append(0)
            trust.append(0)
    return anger, anticipation, disgust, fear, joy, sadness, surprise, trust


def stylometric(text, polarity, anger, anticipation, disgust, fear, joy, sadness, surprise, trust):
    # criando um dataframe com index
    df = pd.DataFrame(text.tokens, columns=['palavra'])
    df.insert(1, 'POSTag', text.pos_tags)
    df = df[df.palavra != 'EOS']

    dfe = pd.DataFrame(text.tokens, columns=['palavra'])
    dfe.insert(1, 'Emotion', polarity)
    dfe = dfe[dfe.palavra != 'EOS']

    dfec = pd.DataFrame(text.tokens, columns=['palavra'])
    dfec.insert(1, 'Anger', anger)
    dfec.insert(2, 'Anticipation', anticipation)
    dfec.insert(3, 'Disgust', disgust)
    dfec.insert(4, 'Fear', fear)
    dfec.insert(5, 'Joy', joy)
    dfec.insert(6, 'Sadness', sadness)
    dfec.insert(7, 'Surprise', surprise)
    dfec.insert(8, 'Trust', trust)
    dfec = dfec[dfec.palavra != 'EOS']

    dfent = pd.DataFrame(text.tokens, columns=['palavra'])
    dfent.insert(1, 'Entity', text.entities)
    dfent = dfent[dfent.palavra != 'EOS']

    dfnp = pd.DataFrame(text.tokens, columns=['palavra'])
    dfnp.insert(1, 'np', text.np_tags)
    dfnp = dfnp[dfnp.palavra != 'EOS']

    dflema = pd.DataFrame(text.tokens, columns=['palavra'])
    dflema.insert(1, 'lema', text.lemas)
    dflema = dflema[dflema.palavra != 'EOS']

    df.to_csv('outs\\basesdf\\pos.csv', encoding='utf-8', index=False)
    dfe.to_csv('outs\\basesdf\\emo.csv', encoding='utf-8', index=False)
    dfec.to_csv('outs\\basesdf\\emoc.csv', encoding='utf-8', index=False)
    dfent.to_csv('outs\\basesdf\\ent.csv', encoding='utf-8', index=False)
    dfnp.to_csv('outs\\basesdf\\np.csv', encoding='utf-8', index=False)
    dflema.to_csv('outs\\basesdf\\lema.csv', encoding='utf-8', index=False)
    print('Dataframes base gerados!')


def words_dataframes():
    df = pd.read_csv('outs\\basesdf\\pos.csv')
    df = df.pivot(columns='POSTag')

    dfe = pd.read_csv('outs\\basesdf\\emo.csv')
    dfe = dfe.pivot(columns='Emotion')

    dfec = pd.read_csv('outs\\basesdf\\emoc.csv')
    dfec = dfec.drop(columns='palavra')

    dfent = pd.read_csv('outs\\basesdf\\ent.csv')
    dfent = dfent.pivot(columns='Entity')

    dfnp = pd.read_csv('outs\\basesdf\\np.csv')
    dfnp = dfnp.pivot(columns='np')

    dflema = pd.read_csv('outs\\basesdf\\lema.csv')
    dflema = dflema.astype('str')

    df = df['palavra']
    dfe = dfe['palavra']
    dfent = dfent['palavra']
    dfnp = dfnp['palavra']
    dflema = dflema['lema']

    dfword = pd.concat([df, dfe, dfent, dfec], axis=1)
    dfword.to_csv('outs\\dfword.csv', encoding='utf-8', index=False)
    print('Dataframe de palavras gerado!')
    

def sentences_count():
    sentence = []
    count = 0
    i = 0

    df = pd.read_csv('outs\\basesdf\\pos.csv')
    df['palavra'] = df['palavra'].astype('str')

    while i < len(df['palavra']):
        if '.' in df['palavra'][i] or '!' in df['palavra'][i] or '?' in df['palavra'][i]:
            sentence.append(count)
            if i + 1 < len(df['palavra']):
                if df['palavra'][i + 1] == '.' or df['palavra'][i + 1] == '!' or df['palavra'][i + 1] == '?' or \
                        df['palavra'][i + 1] == '...':
                    i += 1
                    sentence.append(count)
                    if i + 1 < len(df['palavra']):
                        if df['palavra'][i + 1] == '.' or df['palavra'][i + 1] == '!' or df['palavra'][i + 1] == '?' or \
                                df['palavra'][i + 1] == '...':
                            i += 1
                            sentence.append(count)
                            if i + 1 < len(df['palavra']):
                                if df['palavra'][i + 1] == '.' or df['palavra'][i + 1] == '!' or df['palavra'][
                                    i + 1] == '?' or df['palavra'][i + 1] == '...':
                                    i += 1
                                    sentence.append(count)
            i += 1
            count += 1
        else:
            sentence.append(count)
            i += 1

    dfword=pd.read_csv('outs\\dfword.csv')
    dfword.insert(0, 'Sentence', sentence)

    puncremove = pd.DataFrame()
    puncremove['Sentence'] = dfword['Sentence']
    puncremove['punc'] = dfword['punc']
    puncremove['punc'] = puncremove['punc'].notnull().astype('int')
    puncremove = puncremove.groupby('Sentence').sum()
    puncremove = puncremove['punc']

    dfword['SentenceLength'] = dfword['Sentence'].map(dfword['Sentence'].value_counts() - puncremove)
    dfword.to_csv('outs\\dfword.csv', encoding='utf-8', index=False)
    print('Sentenças contadas!')


def sentences():
    dfword = pd.read_csv('outs\\dfword.csv', low_memory=False)
    sentencesdf = pd.DataFrame()
    df = pd.read_csv('outs\\basesdf\\pos.csv', dtype=str)
    sentencesdf['Sentence'] = dfword['Sentence']
    sentencesdf['palavra'] = df['palavra']
    sentencesdf['palavra'] = sentencesdf['palavra'].astype('str')
    sentencesdf = sentencesdf.groupby(['Sentence'])['palavra'].apply(' '.join).str.replace('\n', ' ').reset_index()
    sentencesdf.drop(columns='Sentence', inplace=True)
    sentencesdf.to_csv('outs\\sentences.csv', encoding='utf-8', index=False)
    print('Sentenças geradas!')


def lema_sentences():
    dfword = pd.read_csv('outs\\dfword.csv', low_memory=False)
    sentenceslemadf = pd.DataFrame()
    dfl = pd.read_csv('outs\\basesdf\\lema.csv', dtype=str)
    sentenceslemadf['Sentence'] = dfword['Sentence']
    sentenceslemadf['lema'] = dfl['lema']
    sentenceslemadf['lema'] = sentenceslemadf['lema'].astype('str')
    sentenceslemadf = sentenceslemadf.groupby(['Sentence'])['lema'].apply(' '.join).str.replace('\n', ' ').reset_index()
    sentenceslemadf.drop(columns='Sentence', inplace=True)
    sentenceslemadf.to_csv('outs\\lemasentences.csv', encoding='utf-8', index=False)
    print('Sentenças de lemas geradas!')


def matchDfs(A, columnA, B, columnB):
    tmp = {}
    for idxA, idxB in product(A.index, B.index):
        sA = A.loc[idxA, columnA]
        sB = B.loc[idxB, columnB]

        m = SequenceMatcher(None, sA, sB).find_longest_match()
        common = sA[m[0]:m[2] + m[0]]
        common_strip = common.strip()

        if len(common_strip) >= 32:
            tmp.setdefault(idxB, []).append((32, idxA))
        elif len(common_strip) >= 30:
            tmp.setdefault(idxB, []).append((30, idxA))
        elif len(common_strip) >= 28:
            tmp.setdefault(idxB, []).append((28, idxA))
        elif len(common_strip) >= 26:
            tmp.setdefault(idxB, []).append((26, idxA))
        elif len(common_strip) >= 24:
            tmp.setdefault(idxB, []).append((24, idxA))
        elif len(common_strip) >= 22:
            tmp.setdefault(idxB, []).append((22, idxA))
        elif len(common_strip) >= 20:
            tmp.setdefault(idxB, []).append((20, idxA))
        elif len(common_strip) >= 18:
            tmp.setdefault(idxB, []).append((18, idxA))
        elif len(common_strip) >= 16:
            tmp.setdefault(idxB, []).append((16, idxA))
        elif len(common_strip) >= 14:
            tmp.setdefault(idxB, []).append((14, idxA))
        elif len(common_strip) >= 12:
            tmp.setdefault(idxB, []).append((12, idxA))
        elif len(common_strip) >= 10:
            tmp.setdefault(idxB, []).append((10, idxA))
        elif len(common_strip) >= 8:
            tmp.setdefault(idxB, []).append((8, idxA))
        elif len(common_strip) >= 6:
            tmp.setdefault(idxB, []).append((6, idxA))
        elif len(common_strip) >= 4:
            tmp.setdefault(idxB, []).append((4, idxA))

    out = pd.DataFrame({'idx_B': tmp.keys(), 'idx_A': tmp.values()})
    return out


def pickvalues(lista):
    lista.sort(key=lambda x: x[0], reverse=True)
    lista = [tupla for tupla in lista if tupla[0] == lista[0][0]]
    lista = [tupla[1] for tupla in lista]
    lista.sort()

    return lista


def mives(txt,savetxt):
    print('Aguarde!')
    with open(txt, 'r', encoding="utf8") as infile, open('temp\\mives.txt', 'w', encoding="utf8") as outfile:
        data = infile.read()
        data = data.replace("\"", "")
        data = data.replace("(", "")
        data = data.replace(")", "")
        data = data.replace("[", "")
        data = data.replace("]", "")
        data = data.replace("{", "")
        data = data.replace("}", "")
        outfile.write(data)

    mives = pd.read_csv('temp\\mives.txt', sep='¢', names=['sentenca', 'verso', 'classificacao', 'classificacao2'],
                        on_bad_lines='warn')
    mives[['sentenca', 'verso', 'classificacao', 'classificacao2']] = mives[
        ['sentenca', 'verso', 'classificacao', 'classificacao2']].astype(str)
    mivesm = mives
    mivesm['sentenca'] = mivesm['sentenca'].str.lower()
    mivesm.sentenca.replace({r'[\W_]+': ''}, regex=True, inplace=True)
    sentencesm = pd.read_csv('outs\\sentences.csv')
    sentencesm['palavra'] = sentencesm['palavra'].str.lower()
    sentencesm.palavra.replace({r'[\W_]+': ''}, regex=True, inplace=True)
    mivesfind = matchDfs(sentencesm, 'palavra', mivesm, 'sentenca')
    print(txt, ': ', len(mivesfind), 'encontradas de', len(mivesm))
    mivesfind.sort_values(by=['idx_B'], inplace=True)
    mivesfind.reset_index(drop=True, inplace=True)
    amatch = []

    for a in mivesfind['idx_A']:
        amatch.append(pickvalues(list(a)))

    for i in range(len(amatch)):
        if type(amatch[i]) is int:
            j = []
            j.append(amatch[i])
            amatch[i] = j

    if len(amatch[0]) > 1:
        amatch[0] = min(amatch[0])

    if len(amatch[1]) > 1:
        amatch[1] = [item for item in amatch[1] if item >= amatch[0][0]]
        amatch[1] = [min(amatch[1])]

    for i in range(2, len(amatch)):
        if len(amatch[i]) > 1:
            amatch[i] = [item for item in amatch[i] if item >= amatch[i - 2][0] or item >= amatch[i - 1][0]]
            if len(amatch[i]) >= 1:
                amatch[i] = [min(amatch[i])]
            else:
                amatch[i] = amatch[i - 1]

    for i in range(len(amatch)):
        amatch[i] = amatch[i][0]

    mivesfind['idx_A'] = amatch
    versos = mives['classificacao'].unique()
    mivessentence = pd.read_csv('outs\\sentences.csv')
    mivessentence[versos] = 0
    mivessentence.drop(columns='palavra', inplace=True)

    for i in range(len(mivesfind)):
        mivessentence[mives['classificacao'][mivesfind['idx_B'][i]]][mivesfind['idx_A'][i]] += 1

    mivessentence['versos'] = mivessentence.sum(numeric_only=True, axis=1)
    mivessentence.to_csv('outs\\'+savetxt, encoding='utf-8', index=False)


def allpro(txt,savetxt):
    print('Aguarde!')
    allpro = pd.read_csv(txt, index_col=0)
    allpro.reset_index(drop=True, inplace=True)
    allprom = allpro
    allprom['frase'] = allprom['frase'].str.lower()
    allprom.frase.replace({r'[\W_]+': ''}, regex=True, inplace=True)

    sentencesm = pd.read_csv('outs\\sentences.csv')
    sentencesm['palavra'] = sentencesm['palavra'].str.lower()
    sentencesm.palavra.replace({r'[\W_]+': ''}, regex=True, inplace=True)

    allprofind = matchDfs(sentencesm, 'palavra', allprom, 'frase')
    print(len(allprofind), 'encontradas de', len(allpro))

    allprofind.sort_values(by=['idx_B'], inplace=True)
    allprofind.reset_index(drop=True, inplace=True)
    amatch = []

    for a in allprofind['idx_A']:
        amatch.append(pickvalues(list(a)))

    for i in range(len(amatch)):
        if type(amatch[i]) is int:
            j = []
            j.append(amatch[i])
            amatch[i] = j

    if len(amatch[0]) > 1:
        amatch[0] = min(amatch[0])

    if len(amatch[1]) > 1:
        amatch[1] = [item for item in amatch[1] if item >= amatch[0][0]]
        amatch[1] = [min(amatch[1])]

    for i in range(2, len(amatch)):
        if len(amatch[i]) > 1:
            amatch[i] = [item for item in amatch[i] if item >= amatch[i - 2][0] or item >= amatch[i - 1][0]]
            if len(amatch[i]) >= 1:
                amatch[i] = [min(amatch[i])]
            else:
                amatch[i] = amatch[i - 1]

    for i in range(len(amatch)):
        amatch[i] = amatch[i][0]

    allprofind['idx_A'] = amatch

    fonemas = allpro['fonema'].unique()
    allprosentence = pd.read_csv('outs\\sentences.csv')
    allprosentence[fonemas] = 0
    allprosentence.drop(columns='palavra', inplace=True)

    for i in range(len(allprofind)):
        allprosentence[allpro['fonema'][allprofind['idx_B'][i]]][allprofind['idx_A'][i]] += 1

    allprosentence['repSil'] = allprosentence.sum(numeric_only=True, axis=1)
    allprosentence.to_csv('outs\\' + savetxt, encoding='utf-8', index=False)

# Biblioteca otimizada para fuzzy matching

def matchDfs_fastMives(A, columnA, B, columnB, threshold=80):
    matches = []

    for idxB, sentenca in B[columnB].items():
        best_match = process.extractOne(sentenca, A[columnA], scorer=fuzz.partial_ratio)

        if best_match and best_match[1] >= threshold:  # Apenas aceita matches acima do threshold
            idxA = A[A[columnA] == best_match[0]].index[0]
            matches.append((idxB, idxA, best_match[1]))

    return pd.DataFrame(matches, columns=['idx_B', 'idx_A', 'score'])

def mivesOp(txt, savetxt):
    print("Aguarde!")

    # Leitura e limpeza do arquivo
    with open(txt, 'r', encoding="utf8") as infile:
        data = infile.read()

    for char in ['"', '(', ')', '[', ']', '{', '}']:
        data = data.replace(char, "")

    with open('temp/mives.txt', 'w', encoding="utf8") as outfile:
        outfile.write(data)

    # Leitura otimizada do CSV
    mives = pd.read_csv('temp/mives.txt', sep='¢', names=['sentenca', 'verso', 'classificacao', 'classificacao2'],
                        on_bad_lines='warn', dtype=str)

    # Padronização de texto
    mives['sentenca'] = mives['sentenca'].str.lower().str.replace(r'[\W_]+', '', regex=True)

    # Leitura do dicionário de palavras
    sentencesm = pd.read_csv('outs/sentences.csv', dtype=str)
    sentencesm['palavra'] = sentencesm['palavra'].str.lower().str.replace(r'[\W_]+', '', regex=True)

    # Busca otimizada das correspondências
    mivesfind = matchDfs_fastMives(sentencesm, 'palavra', mives, 'sentenca')

    print(f"{txt}: {len(mivesfind)} encontradas de {len(mives)}")

    # Criação do DataFrame final
    versos = mives['classificacao'].unique()
    mivessentence = pd.read_csv('outs/sentences.csv', dtype=str)
    mivessentence[versos] = 0
    mivessentence.drop(columns='palavra', inplace=True)

    for _, row in mivesfind.iterrows():
        mivessentence.loc[row['idx_A'], mives.loc[row['idx_B'], 'classificacao']] += 1

    mivessentence['versos'] = mivessentence.sum(numeric_only=True, axis=1)
    mivessentence.to_csv(f'outs/{savetxt}', encoding='utf-8', index=False)

def matchDfs_fastAll(A, columnA, B, columnB, threshold=80):
    """
    Encontra correspondências entre dois DataFrames usando RapidFuzz.
    Apenas retorna matches com score acima do threshold.
    """
    matches = []

    for idxB, frase in B[columnB].items():
        best_match = process.extractOne(frase, A[columnA], scorer=fuzz.partial_ratio)

        if best_match and best_match[1] >= threshold:  # Apenas aceita matches acima do threshold
            idxA = A[A[columnA] == best_match[0]].index[0]
            matches.append((idxB, idxA, best_match[1]))

    return pd.DataFrame(matches, columns=['idx_B', 'idx_A', 'score'])

def allproOp(txt, savetxt):
    print("Aguarde!")

    # Leitura do arquivo
    allpro = pd.read_csv(txt, index_col=0)
    allpro.reset_index(drop=True, inplace=True)

    # Normalização do texto
    allpro['frase'] = allpro['frase'].str.lower().str.replace(r'[\W_]+', '', regex=True)

    # Carregamento do dicionário de palavras
    sentencesm = pd.read_csv('outs/sentences.csv')
    sentencesm['palavra'] = sentencesm['palavra'].str.lower().str.replace(r'[\W_]+', '', regex=True)

    # Busca otimizada das correspondências
    allprofind = matchDfs_fastAll(sentencesm, 'palavra', allpro, 'frase')

    print(f"{len(allprofind)} encontradas de {len(allpro)}")

    # Organização dos índices encontrados
    allprofind['idx_A'] = allprofind.groupby('idx_B')['idx_A'].transform(lambda x: sorted(x)[0])

    # Criação do DataFrame final
    fonemas = allpro['fonema'].unique()
    allprosentence = pd.read_csv('outs/sentences.csv')
    allprosentence[fonemas] = 0
    allprosentence.drop(columns='palavra', inplace=True)

    for _, row in allprofind.iterrows():
        allprosentence.loc[row['idx_A'], allpro.loc[row['idx_B'], 'fonema']] += 1

    allprosentence['repSil'] = allprosentence.sum(numeric_only=True, axis=1)
    allprosentence.to_csv(f'outs/{savetxt}', encoding='utf-8', index=False)





if __name__ == '__main__':
    input_file = input("Insira o caminho ou nome do arquivo (ex: domcasmurro.txt): \n")
    print('Aguarde!')
    options = {
        "tokenizer": True,
        "pos_tagger": True,
        "lemmatizer": True,
        "entity_recognition": True,
        "np_chunking": True,
        "pre_load": False,
        "string_or_array": False
    }
    change_nlpyport()
    text = process(input_file, options)
    polarity = polarities(text)
    anger, anticipation, disgust, fear, joy, sadness, surprise, trust = emotions(text)
    stylometric(text, polarity, anger, anticipation, disgust, fear, joy, sadness, surprise, trust)
    words_dataframes()
    sentences_count()
    sentences()
    #lema_sentences() #Não está sendo utilizado mas pode ser utilizado
    if 'books/' in input_file:
        input_file = input_file.replace('books/', '')
    elif 'books\\' in input_file:
        input_file = input_file.replace('books\\', '')
    input_file = input_file.replace('.txt','')

    input('ATENÇÃO, COLOQUE NA PASTA MIVES OS ARQUIVOS EXTRAÍDOS DO MIVES DA SEGUINTE FORMA:\n' + input_file + '_sc'
            '.txt para extração de sentenças completas\n'+ input_file + '_is.txt para extração no início de sentença\n'+
          input_file + '_fs.txt para extração no final de sentença\nApós isso, pressione ENTER\n')
    mives('mives\\' + input_file + '_sc.txt', 'mivessc.csv')
    mives('mives\\' + input_file + '_is.txt', 'mivesis.csv')
    mives('mives\\' + input_file + '_fs.txt', 'mivesfs.csv')

    input('ATENÇÃO, COLOQUE NA PASTA ALLPRO OS ARQUIVOS EXTRAÍDOS DO ALLPRO DA SEGUINTE FORMA:\n' + input_file + '_2_3'
            '.txt para extração no padrão 2S3L\n'+ input_file + '_3_4.txt para extração no padrão 3S4L\n'+
          input_file + '_4_4.txt para extração no padrão 4S4L\nApós isso, pressione ENTER\n')
    allpro('allpro\\' + input_file + '_2_3.txt', 'allpro2_3.csv')
    allpro('allpro\\' + input_file + '_3_4.txt', 'allpro3_4.csv')
    allpro('allpro\\' + input_file + '_4_4.txt', 'allpro4_4.csv')