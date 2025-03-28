from feature_extraction import *
from ut_features import *

#feature_extraction
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
text = processTex(input_file, options)
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
'''
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

'''

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

#ut_features
ut = int(input("Insira o tamanho da unidade textual (quant. de sentenças a serem agrupadas para a análise):"))
if ut>1:
    if ut<50:
        print("Atenção! Para a modelagem de tópicos é recomendado UT maiores ou iguais a 50")
    topic_modeling(ut)
else:
    print('Modelagem de tópicos não aplicada pois o tamanho da UT é igual a 1')
ut_organize(ut)
paralelismDetection(ut)

print("Processo completo! Rode o view.py para visualizar os resultados!")