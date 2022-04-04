import streamlit as st
import pandas as pd
import random
import re
import pickle

import nltk
nltk.download('averaged_perceptron_tagger')

import plotly.express as px
import time
import numpy as np

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

import nltk
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
lancaster = LancasterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
def clean(text):
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text

pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

def sentiwordnetanalysis(pos_data):
    pos_score,neg_score=0,0
    total_tokens = 0
    for word, pos in pos_data:
        if not pos:
            continue
        lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
        if not lemma:
            continue
        synsets = wordnet.synsets(lemma, pos=pos)
        if not synsets:
            continue
        total_tokens+=1
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        pos_score+= swn_synset.pos_score()
        neg_score+= swn_synset.neg_score()
    if total_tokens==0:
        return 0,0
    return {'pos':pos_score/total_tokens, 'neg':neg_score/total_tokens}

def stemmize(tagged):
    stemmed = ''
    for word, _ in tagged:
        stemmed = stemmed +' '+lancaster.stem(word)
    return stemmed.strip()

st.sidebar.title('CZ4034 Project: War Twitter Analysis')
option = st.sidebar.radio(
     "",
     ('War Tweets Search', 'Sentiment Analysis'))


if "page" not in st.session_state:
    st.session_state.page = 1


data = pd.read_csv('new_df.csv')
data= data.dropna().iloc[1:,1:]

rd = 0

search_result = data.sample(1000)
st.session_state.search_result_p = []




if option=='War Tweets Search':   
    st.title('War Tweets Search')     
    term = st.text_input('Search Term')
    start = time.time()
    rd = random.randint(100,1000)
    search_result = data.sample(rd).dropna()
    st.session_state.time_spent = time.time()-start



    if term:
        st.write('Term: '+str(term))


        st.write(f'got {len(search_result)} results in {np.round(10000*st.session_state.time_spent,2)} ms')
        st.write('-------')

        fig = px.histogram(search_result['polarity'], x="polarity")
        with st.expander('analysis'):
            st.plotly_chart(fig,use_container_width=True)
        with st.expander('map'):
            st.map()
        with st.expander('word cloud'):
            st.write('word_cloud')


        c1, _ = st.columns([1,3])
        with c1:
            st.session_state.page = st.number_input('Page',1,100)
            # st.write(st.session_state.page)
            PAGE_SIZE = 5
            st.session_state.search_result_p = search_result['text'][PAGE_SIZE*int(st.session_state.page-1):PAGE_SIZE*int(st.session_state.page)]

        with st.expander("See Results"):
            for s in st.session_state.search_result_p:
                st.write(s)
                # st.write(str(analyzer.polarity_scores(s)))
                st.write('-------')




if option=='Sentiment Analysis':
    st.title('Sentiment Analysis')
    test_sentiment = st.text_input('Input Text')

    
    if test_sentiment:
        st.header('1. Rule Based Analysis')
        st.subheader('1.1 Textblob')
        st.write(TextBlob(str(test_sentiment)).sentiment)

        st.subheader('1.2 VADER')
        st.write(analyzer.polarity_scores(test_sentiment))

        st.subheader('1.3 SentiWordNet')
        cleaned = clean(str(test_sentiment))
        tagged = token_stop_pos(cleaned)
        st.write(sentiwordnetanalysis(tagged))

        st.header('2. ML Based Analysis')

        stemmed = stemmize(tagged)

        st.subheader('2.1 Subjectivity Detection')
        with open('./tf-pkl/subject_text.pkl','rb') as f:
            tf_subject = pickle.load(f)
        text_tf= tf_subject.transform([stemmed])
        with open('./sklearn-pkl/subjectivity/logistic_regression.pkl','rb') as f:
            clf_subject_logistic = pickle.load(f)
        with open('./sklearn-pkl/subjectivity/neural_network.pkl','rb') as f:
            clf_subject_nn = pickle.load(f)
        y_sub_log = clf_subject_logistic.predict(text_tf)
        y_sub_nn = clf_subject_nn.predict(text_tf)
        subject_dict = {0:'non-opinioned',1:'opinioned'}
        st.markdown(f'logistic regression: *{str(subject_dict[y_sub_log[0]])}*')
        st.write(f'neural network: *{str(subject_dict[y_sub_nn[0]])}*')

        st.subheader('2.2 Polarity Detection')
        if y_sub_log[0]==1 and y_sub_nn[0]==1:
            with open('./tf-pkl/polarity_text.pkl','rb') as f:
                tf_po = pickle.load(f)
            text_tf= tf_po.transform([stemmed])
            with open('./sklearn-pkl/polarity/logistic_regression.pkl','rb') as f:
                clf_po_logistic = pickle.load(f)
            with open('./sklearn-pkl/polarity/neural_network.pkl','rb') as f:
                clf_po_nn = pickle.load(f)

            y_po_log = clf_po_logistic.predict(text_tf)
            y_po_nn = clf_po_nn.predict(text_tf)
            sentiment_dict = {
                0:'negative',
                1:'positive'
            }
            st.write(f'logistic regression: *{str(sentiment_dict[y_po_log[0]])}*')
            st.write(f'neural network: *{str(sentiment_dict[y_po_nn[0]])}*')
        else:
            st.write(f'*neutral!!!*')

        st.subheader('2.3 Side Detection')
        with open('./tf-pkl/attitude.pkl','rb') as f:
            tf_side = pickle.load(f)
        text_tf= tf_side.transform([stemmed])
        with open('./sklearn-pkl/attitude/decision_tree.pkl','rb') as f:
            clf_side = pickle.load(f)
        y_side = clf_side.predict(text_tf)
        side_dict = {0:'Russia Will Lose',1:'Russia Will Win'}
        st.write(f'Decision Tree: *{str(side_dict[y_side[0]])}*')

        st.subheader('2.4 Spatial Detection')
        with open('./tf-pkl/geo.pkl','rb') as f:
            tf_spatial = pickle.load(f)
        text_tf= tf_spatial.transform([stemmed])[:,:5000]
        with open('./sklearn-pkl/geo/adaboost.pkl','rb') as f:
            clf_spatial = pickle.load(f)
        y_side = clf_spatial.predict(text_tf)
        st.write(f'Decision Tree: *{y_side[0]}*')






    






