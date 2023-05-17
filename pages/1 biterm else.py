#import module
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from pprint import pprint
import spacy
import pickle
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from io import StringIO
from ipywidgets.embed import embed_minimal_html
from nltk.stem.snowball import SnowballStemmer
import bitermplus as btm
import tmplot as tmp


#===config===
st.set_page_config(
     page_title="Coconut",
     page_icon="🥥",
     layout="wide"
)
st.header("Topic Modeling")
st.subheader('Put your CSV file here ...')

#===upload file===
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    papers = pd.read_csv(uploaded_file)
    paper = papers.dropna(subset=['Abstract'])
    paper = paper[~paper.Abstract.str.contains("No abstract available")]
    paper = paper[~paper.Abstract.str.contains("STRAIT")]
        
    #===mapping===
    paper['Abstract_pre'] = paper['Abstract'].map(lambda x: re.sub('[,:;\.!?•-]', '', x))
    paper['Abstract_pre'] = paper['Abstract_pre'].map(lambda x: x.lower())
    paper['Abstract_pre'] = paper['Abstract_pre'].map(lambda x: re.sub('©.*', '', x))
        
    #===lemmatize===
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(text):
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    paper['Abstract_lem'] = paper['Abstract_pre'].apply(lemmatize_words)
       
    #===stopword removal===
    stop = stopwords.words('english')
    paper['Abstract_stop'] = paper['Abstract_lem'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    method = st.selectbox(
            'Choose method',
            ('Biterm', 'Empty'))
        
    #===Biterm===
    if method == 'Biterm':
        num_bitopic = st.slider('Choose number of topics', min_value=8, max_value=20, step=1)
        topic_abs = paper.Abstract_stop.values.tolist()       
        X, vocabulary, vocab_dict = btm.get_words_freqs(topic_abs)
        tf = np.array(X.sum(axis=0)).ravel()
        docs_vec = btm.get_vectorized_docs(topic_abs, vocabulary)
        docs_lens = list(map(len, docs_vec))
        biterms = btm.get_biterms(docs_vec)
        model = btm.BTM(
          X, vocabulary, seed=12321, T=num_bitopic, M=20, alpha=50/8, beta=0.01)
        model.fit(biterms, iterations=20)
        p_zd = model.transform(docs_vec)
        coherence = model.coherence_
        with st.spinner('Visualizing, please wait ....'):
         
             topics_coords = tmp.prepare_coords(model)
             phi = tmp.get_phi(model)  

             btmvis = tmp.report(model=model, docs=texts)
             with StringIO() as f:
                embed_minimal_html(f, [btmvis], title="BTM")
                fig_html = f.getvalue()
             st.components.v1.html(fig_html, use_container_width=True, height=800, scrolling=True)
            
