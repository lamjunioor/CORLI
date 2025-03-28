import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == '__main__':
    install('spacy')
    install('NLPyPort')
    install('ipywidgets')
    install('gensim')
    install('nltk==3.4.5')
    install('plotly')
    install('dash')
    install('dash_bootstrap_components')
    install('dash_bootstrap_templates')
    import nltk
    nltk.download('floresta')
    nltk.download('punkt')
    nltk.download('stopwords')
    #nltk.download('punkt_tab')