features_list = [{'label': 'Rep. Sílabas Fonéticas 2S3L','value': 'repSil_2_3'},
            {'label': 'Rep. Sílabas Fonéticas 3S4L','value': 'repSil_3_4'},
            {'label': 'Rep. Sílabas Fonéticas 4S4L','value': 'repSil_4_4'},
            {'label': 'Versificação - Sentenças Completas','value': 'versos_sc'},
            {'label': 'Versificação - Início de Sentença','value': 'versos_is'},
            {'label': 'Versificação - Final de Sentença','value': 'versos_fs'},
            {'label': 'Contagem de Palavras','value': 'wordCount'},
            {'label': 'Palavras únicas','value': 'palavrasUnicas'},
            {'label': 'Lemas únicos','value': 'lemasUnicos'},
            {'label': 'TTR de palavras','value': 'ttrPalavras'},
            {'label': 'TTR de lemas','value': 'ttrLemas'},
            {'label': 'Adjetivo','value': 'adj'},
            {'label': 'Advérbios','value': 'adv'},
            {'label': 'Artigos','value': 'art'},
            {'label': 'Conjunções','value': 'conjs'},
            {'label': 'Interjeições','value': 'in'},
            {'label': 'Numerais','value': 'num'},
            {'label': 'Pontuações','value': 'punc'},
            {'label': 'Pronomes','value': 'prons'},
            {'label': 'Preposições','value': 'prp'},
            {'label': 'Substantivo','value': 'noun'},
            {'label': 'Verbos','value': 'verbs'},
            {'label': 'Positivos','value': 'positivos'},
            {'label': 'Negativos','value': 'negativos'},
            {'label': 'Neutros','value': 'neutros'},
            {'label': 'Alegria','value': 'alegria'},
            {'label': 'Confiança','value': 'confianca'},
            {'label': 'Expectativa','value': 'expectativa'},
            {'label': 'Medo','value': 'medo'},
            {'label': 'Nojo','value': 'nojo'},
            {'label': 'Raiva','value': 'raiva'},
            {'label': 'Surpresa','value': 'surpresa'},
            {'label': 'Tristeza','value': 'tristeza'},
            {'label': 'Adjetivos Positivos','value': 'adjpos'},
            {'label': 'Adjetivos Negativos','value': 'adjneg'},
            {'label': 'Advérbios Positivos','value': 'advpos'},
            {'label': 'Advérbios Negativos','value': 'advneg'},
            {'label': 'Polaridade Geral','value': 'polaridade'},
            {'label': 'Polaridade de Adjetivos','value': 'adjPol'},
            {'label': 'Polaridade de Advérbios','value': 'advPol'},
            {'label': 'Carga Emocional de adjetivos','value': 'adjCharge'},
            {'label': 'Carga Emocional de advérbios','value': 'advCharge'},
            {'label': 'Carga Emocional Geral','value': 'emoCharge'},
            {'label': 'REN: Pessoa','value': 'nePessoa'},
            {'label': 'REN: Local','value': 'neLocal'},
            {'label': 'REN: Pessoa e Local','value': 'nePessoaELocal'},
            {'label': 'REN: Geral','value': 'neGeral'},
            {'label': 'Frequência de Adjetivos','value': 'adjFreq'},
            {'label': 'Frequência de Advérbios','value': 'advFreq'},
            {'label': 'Frequência de Artigos','value': 'artFreq'},
            {'label': 'Frequência de Conjunções','value': 'conjFreq'},
            {'label': 'Frequência de Interjeições','value': 'inFreq'},
            {'label': 'Frequência de Numerais','value': 'numFreq'},
            {'label': 'Frequência de Pontuações','value': 'puncFreq'},
            {'label': 'Frequência de Pronomes','value': 'pronFreq'},
            {'label': 'Frequência de Preposições','value': 'prpFreq'},
            {'label': 'Frequência de Substantivos','value': 'nounFreq'},
            {'label': 'Frequência de Verbos','value': 'verbFreq'},
            {'label': 'Frequência de Positivos','value': 'posFreq'},
            {'label': 'Frequência de Negativos','value': 'negFreq'},
            {'label': 'Frequência de Neutros','value': 'neuFreq'},
            {'label': 'Frequência de Alegria','value': 'alegriaFreq'},
            {'label': 'Frequência de Confiança','value': 'confiancaFreq'},
            {'label': 'Frequência de Expectativa','value': 'expectativaFreq'},
            {'label': 'Frequência de Medo','value': 'medoFreq'},
            {'label': 'Frequência de Nojo','value': 'nojoFreq'},
            {'label': 'Frequência de Raiva','value': 'raivaFreq'},
            {'label': 'Frequência de Surpresa','value': 'surpresaFreq'},
            {'label': 'Frequência de Tristeza','value': 'tristezaFreq'},
            {'label': 'Frequência de Pessoa','value': 'pessoaFreq'},
            {'label': 'Frequência de Local','value': 'localFreq'},
            {'label': 'Frequência de Pessoa/Local','value': 'pes_locFreq'},
            {'label': 'Frequência de Entidades Nomeadas','value': 'NEFreq'}]

levels={'repSil_2_3':['repSil'], 'repSil_3_4':['repSil'], 'repSil_4_4':['repSil'], 'versos_sc':['verso'], 'versos_is':['verso'],
        'versos_fs': ['verso'], 'wordCount':['cont'], 'palavrasUnicas':['cont'], 'lemasUnicos':['cont'],
        'ttrPalavras':['cont'], 'ttrLemas':['cont'], 'adj':['pos'], 'adv':['pos'], 'art':['pos'], 'conjs':['pos'], 'in':['pos'],
        'num':['pos'], 'punc':['pos'], 'prons':['pos'], 'prp':['pos'], 'noun':['pos'], 'verbs':['pos'], 'positivos':['sent'],
        'negativos':['sent'], 'neutros':['sent'], 'alegria':['sent'], 'confianca':['sent'], 'expectativa':['sent'], 'medo':['sent'],
        'nojo':['sent'], 'raiva':['sent'], 'surpresa':['sent'], 'tristeza':['sent'], 'adjpos':['sent','pos'], 'adjneg':['sent','pos'],
        'advpos':['sent','pos'], 'advneg':['sent','pos'], 'nePessoa':['ne'], 'neLocal':['ne'], 'nePessoaELocal':['ne'], 'neGeral':['ne'],
        'polaridade':['sent'], 'adjPol':['sent'], 'advPol':['sent'], 'adjCharge':['sent'], 'advCharge':['sent'],
        'emoCharge':['sent'], 'adjFreq':['pos'], 'advFreq':['pos'], 'artFreq':['pos'], 'conjFreq':['pos'],
        'inFreq':['pos'], 'numFreq':['pos'], 'puncFreq':['pos'], 'pronFreq':['pos'], 'prpFreq':['pos'],
        'nounFreq':['pos'], 'verbFreq':['pos'], 'posFreq':['sent'], 'negFreq':['sent'], 'neuFreq':['sent'],
        'alegriaFreq':['sent'], 'confiancaFreq':['sent'], 'expectativaFreq':['sent'], 'medoFreq':['sent'],
        'nojoFreq':['sent'], 'raivaFreq':['sent'], 'surpresaFreq':['sent'], 'tristezaFreq':['sent'],
        'pessoaFreq':['ne'], 'localFreq':['ne'], 'pes_locFreq':['ne'], 'NEFreq':['ne']}

levelscor={'repSil_2_3':['repSil'], 'repSil_3_4':['repSil'], 'repSil_4_4':['repSil'], 'versos_sc':['verso'], 'versos_is':['verso'],
        'versos_fs': ['verso'], 'wordCount':['cont'], 'palavrasUnicas':['cont'], 'lemasUnicos':['cont'],
        'ttrPalavras':['cont'], 'ttrLemas':['cont'], 'adj':['pos'], 'adv':['pos'], 'art':['pos'], 'conjs':['pos'], 'in':['pos'],
        'num':['pos'], 'punc':['pos'], 'prons':['pos'], 'prp':['pos'], 'noun':['pos'], 'verbs':['pos'], 'positivos':['sent'],
        'negativos':['sent'], 'neutros':['sent'], 'alegria':['sent'], 'confianca':['sent'], 'expectativa':['sent'], 'medo':['sent'],
        'nojo':['sent'], 'raiva':['sent'], 'surpresa':['sent'], 'tristeza':['sent'], 'adjpos':['sent','pos'], 'adjneg':['sent','pos'],
        'advpos':['sent','pos'], 'advneg':['sent','pos'], 'nePessoa':['ne'], 'neLocal':['ne'], 'nePessoaELocal':['ne'], 'neGeral':['ne'],
        'polaridade':['sent'], 'adjPol':['sent'], 'advPol':['sent'], 'adjCharge':['sent'], 'advCharge':['sent'],
        'emoCharge':['sent'], 'adjFreq':['pos'], 'advFreq':['pos'], 'artFreq':['pos'], 'conjFreq':['pos'],
        'inFreq':['pos'], 'numFreq':['pos'], 'puncFreq':['pos'], 'pronFreq':['pos'], 'prpFreq':['pos'],
        'nounFreq':['pos'], 'verbFreq':['pos'], 'posFreq':['sent'], 'negFreq':['sent'], 'neuFreq':['sent'],
        'alegriaFreq':['sent'], 'confiancaFreq':['sent'], 'expectativaFreq':['sent'], 'medoFreq':['sent'],
        'nojoFreq':['sent'], 'raivaFreq':['sent'], 'surpresaFreq':['sent'], 'tristezaFreq':['sent'],
        'pessoaFreq':['ne'], 'localFreq':['ne'], 'pes_locFreq':['ne'], 'NEFreq':['ne'], 'Tópico 1':['topic'],
        'Tópico 2':['topic'], 'Tópico 3':['topic'], 'Tópico 4':['topic'], 'Tópico 5':['topic']}

features_listup = [{'label': 'Rep. Sílabas Fonéticas 2S3L','value': 'repSil_2_3'},
                    {'label': 'Rep. Sílabas Fonéticas 3S4L','value': 'repSil_3_4'},
                    {'label': 'Rep. Sílabas Fonéticas 4S4L','value': 'repSil_4_4'},
                    {'label': 'Versificação - Sentenças Completas','value': 'versos_sc'},
                    {'label': 'Versificação - Início de Sentença','value': 'versos_is'},
                    {'label': 'Versificação - Final de Sentença','value': 'versos_fs'},
                    {'label': 'Contagem de Palavras', 'value': 'wordCount'},
                    {'label': 'Palavras únicas', 'value': 'palavrasUnicas'},
                    {'label': 'Lemas únicos', 'value': 'lemasUnicos'},
                    {'label': 'TTR de palavras', 'value': 'ttrPalavras'},
                    {'label': 'TTR de lemas', 'value': 'ttrLemas'},
                    {'label': 'Adjetivo', 'value': 'adj'},
                    {'label': 'Advérbios', 'value': 'adv'},
                    {'label': 'Artigos', 'value': 'art'},
                    {'label': 'Conjunções', 'value': 'conjs'},
                    {'label': 'Interjeições', 'value': 'in'},
                    {'label': 'Numerais', 'value': 'num'},
                    {'label': 'Pontuações', 'value': 'punc'},
                    {'label': 'Pronomes', 'value': 'prons'},
                    {'label': 'Preposições', 'value': 'prp'},
                    {'label': 'Substantivo', 'value': 'noun'},
                    {'label': 'Verbos', 'value': 'verbs'},
                    {'label': 'Positivos', 'value': 'positivos'},
                    {'label': 'Negativos', 'value': 'negativos'},
                    {'label': 'Neutros', 'value': 'neutros'},
                    {'label': 'Alegria', 'value': 'alegria'},
                    {'label': 'Confiança', 'value': 'confianca'},
                    {'label': 'Expectativa', 'value': 'expectativa'},
                    {'label': 'Medo', 'value': 'medo'},
                    {'label': 'Nojo', 'value': 'nojo'},
                    {'label': 'Raiva', 'value': 'raiva'},
                    {'label': 'Surpresa', 'value': 'surpresa'},
                    {'label': 'Tristeza', 'value': 'tristeza'},
                    {'label': 'Adjetivos Positivos', 'value': 'adjpos'},
                    {'label': 'Adjetivos Negativos', 'value': 'adjneg'},
                    {'label': 'Advérbios Positivos', 'value': 'advpos'},
                    {'label': 'Advérbios Negativos', 'value': 'advneg'},
                    {'label': 'Polaridade Geral', 'value': 'polaridade'},
                    {'label': 'Polaridade de Adjetivos', 'value': 'adjPol'},
                   {'label': 'Polaridade de Advérbios', 'value': 'advPol'},
                   {'label': 'Carga Emocional de adjetivos', 'value': 'adjCharge'},
                   {'label': 'Carga Emocional de advérbios', 'value': 'advCharge'},
                   {'label': 'Carga Emocional Geral', 'value': 'emoCharge'},
                   {'label': 'REN: Pessoa', 'value': 'nePessoa'},
                   {'label': 'REN: Local', 'value': 'neLocal'},
                   {'label': 'REN: Pessoa e Local', 'value': 'nePessoaELocal'},
                   {'label': 'REN: Geral', 'value': 'neGeral'},
                   {'label': 'Frequência de Adjetivos', 'value': 'adjFreq'},
                   {'label': 'Frequência de Advérbios', 'value': 'advFreq'},
                   {'label': 'Frequência de Artigos', 'value': 'artFreq'},
                   {'label': 'Frequência de Conjunções', 'value': 'conjFreq'},
                   {'label': 'Frequência de Interjeições', 'value': 'inFreq'},
                   {'label': 'Frequência de Numerais', 'value': 'numFreq'},
                   {'label': 'Frequência de Pontuações', 'value': 'puncFreq'},
                   {'label': 'Frequência de Pronomes', 'value': 'pronFreq'},
                   {'label': 'Frequência de Preposições', 'value': 'prpFreq'},
                   {'label': 'Frequência de Substantivos', 'value': 'nounFreq'},
                   {'label': 'Frequência de Verbos', 'value': 'verbFreq'},
                   {'label': 'Frequência de Positivos', 'value': 'posFreq'},
                   {'label': 'Frequência de Negativos', 'value': 'negFreq'},
                   {'label': 'Frequência de Neutros', 'value': 'neuFreq'},
                   {'label': 'Frequência de Alegria', 'value': 'alegriaFreq'},
                   {'label': 'Frequência de Confiança', 'value': 'confiancaFreq'},
                   {'label': 'Frequência de Expectativa', 'value': 'expectativaFreq'},
                   {'label': 'Frequência de Medo', 'value': 'medoFreq'},
                   {'label': 'Frequência de Nojo', 'value': 'nojoFreq'},
                   {'label': 'Frequência de Raiva', 'value': 'raivaFreq'},
                   {'label': 'Frequência de Surpresa', 'value': 'surpresaFreq'},
                   {'label': 'Frequência de Tristeza', 'value': 'tristezaFreq'},
                    {'label': 'Frequência de Pessoa','value': 'pessoaFreq'},
                    {'label': 'Frequência de Local','value': 'localFreq'},
                    {'label': 'Frequência de Pessoa/Local','value': 'pes_locFreq'},
                    {'label': 'Frequência de Entidades Nomeadas','value': 'NEFreq'},
                    {'label': 'Tópico 1','value': 'Tópico 1'},
                    {'label': 'Tópico 2','value': 'Tópico 2'},
                    {'label': 'Tópico 3','value': 'Tópico 3'},
                    {'label': 'Tópico 4','value': 'Tópico 4'},
                    {'label': 'Tópico 5','value': 'Tópico 5'}]


correlation_list = [{'label': 'Sílabas fonéticas (3)','value': 'Sílabas fonéticas (3)'},
            {'label': 'Sent. Métricas (3)','value': 'Sent. Métricas (3)'},
            {'label': 'Contagem de Palavras (1)','value': 'Contagem de Palavras (1)'},
            {'label': 'Unicidade de palavras (4)','value': 'Unicidade de palavras (4)'},
            {'label': 'POS-Tag (11)','value': 'POS-Tag (11)'},
            {'label': 'Análise de Sentimentos (11)','value': 'Análise de Sentimentos (11)'},
            {'label': 'POS+Sentimentos (4)','value': 'POS+Sentimentos (4)'},
            {'label': 'Entidade Nomeada (4)','value': 'Entidade Nomeada (4)'},
            {'label': 'Polaridades e Cargas Emocionais (6)','value': 'Polaridades e Cargas Emocionais (6)'},
            {'label': 'Frequências de POS (11)','value': 'Frequências de POS (11)'},
            {'label': 'Frequências de Sentimentos (11)', 'value': 'Frequências de Sentimentos (11)'},
            {'label': 'Frequências de Entidades Nomeadas (4)','value': 'Frequências de Entidades Nomeadas (4)'}]


correlation_listup = [{'label': 'Sílabas fonéticas (3)','value': 'Sílabas fonéticas (3)'},
            {'label': 'Sent. Métricas (3)','value': 'Sent. Métricas (3)'},
            {'label': 'Contagem de Palavras (1)','value': 'Contagem de Palavras (1)'},
            {'label': 'Unicidade de palavras (4)','value': 'Unicidade de palavras (4)'},
            {'label': 'POS-Tag (11)','value': 'POS-Tag (11)'},
            {'label': 'Análise de Sentimentos (11)','value': 'Análise de Sentimentos (11)'},
            {'label': 'POS+Sentimentos (4)','value': 'POS+Sentimentos (4)'},
            {'label': 'Entidade Nomeada (4)','value': 'Entidade Nomeada (4)'},
            {'label': 'Polaridades e Cargas Emocionais (6)','value': 'Polaridades e Cargas Emocionais (6)'},
            {'label': 'Frequências de POS (11)', 'value': 'Frequências de POS (11)'},
            {'label': 'Frequências de Sentimentos (11)', 'value': 'Frequências de Sentimentos (11)'},
            {'label': 'Frequências de Entidades Nomeadas (4)', 'value': 'Frequências de Entidades Nomeadas (4)'},
            {'label': 'Modelagem de Tópicos (5)', 'value':'Modelagem de Tópicos (5)'}]


method_list = [{'label': 'Pearson','value': 'pearson'},
            {'label': 'Spearman','value': 'spearman'}]