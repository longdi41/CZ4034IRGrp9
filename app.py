import streamlit as st
import pandas as pd
import random
import re
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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


import pysolr
from nltk.tokenize import word_tokenize
import re
import pandas
import requests

try:
    solr = pysolr.Solr('http://localhost:8983/solr/CZ4034')
    solr.ping() #test connection 
except Exception as e:
    if "Failed to connect to server" in str(e):
        print("No solr installed/running")
    elif "HTTP 404" in str(e):
        print("Core CZ4034 doesnt exist")
    else:
        print(e)

def search(search_str, page_no = 1, result_per_page = 5):
    query_string = search_str
    page_no = page_no - 1 #default start is 0
    
    params = {
        'df': 'content',
        'rows': result_per_page,
        'start': page_no,
        'spellcheck': 'true'
    }

    search_results = solr.search(query_string, search_handler='select',**params)
    collations = []
    if search_results.spellcheck.keys():
        collations = list(filter(lambda a: a != 'collation', search_results.spellcheck['collations']))
    
    return search_results.docs, search_results.hits, collations

def autocomplete_phase(phase):
    #return a list of tuples, tuple: [0] = term, [1] term freq
    #autocomplete = solr.suggest_terms("content", term)
    query_string = phase.rstrip()
    search_results = solr.search(query_string, search_handler='suggest')
    return search_results.raw_response['suggest']['default'][query_string]['suggestions']


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

def similar_search(text):
    similar = solr.more_like_this('content:'+text, mltfl='content')
    return similar.docs

st.sidebar.title('CZ4034 Project: War Twitter Analysis')
option = st.sidebar.radio(
     "",
     ('War Tweets Search','More Like This', 'Sentiment Analysis'))
if "page" not in st.session_state:
    st.session_state.page = 1


data = pd.read_csv('new_df.csv')
data= data.dropna().iloc[1:,1:]

rd = 0

search_result = data.sample(1000)
st.session_state.search_result_p = []

st.sidebar.subheader('2022 Spring, Group 9')
st.sidebar.write('LEE MING DA')
st.sidebar.write('OO JUN RUI')
st.sidebar.write('TAN JUN LONG')
st.sidebar.write('ZHANG JUNXIANG')




if option=='War Tweets Search':   
    st.title('War Tweets Search')     
    term = st.text_input('Search Term')
    
    rd = random.randint(100,1000)
    # search_result = data.sample(rd).dropna()
    
    if term:
        start = time.time()
        search_response = search(term)
        st.session_state.time_spent = time.time()-start

        result = search_response[0]
        total_number = search_response[1]
        spell_check = search_response[2]
        if spell_check:
            st.write('#### Spell Check')
            st.write(spell_check)

        complete = autocomplete_phase(term)
        if complete:
            st.write('#### Auto Complete')
            st.write([k['term'].replace('<b>','').replace('</b>','') for k in complete])

        # st.write(result)
        # search_result = [r['content'] for r in result]

        st.write(f'got {total_number} results in {np.round(10000*st.session_state.time_spent,2)} ms')
        st.write('-------')

        if not result:
            st.write('No Results!')
        else:
            c1, _ = st.columns([1,3])
            with c1:
                PAGE_SIZE = 5
                total_page = int(total_number/PAGE_SIZE)
                page = st.number_input('Page',1,total_page)
                # st.write(st.session_state.page)
                result = search(term,int(page))[0]
                search_result = [r['content'] for r in result]
                st.session_state.search_result_p = search_result

            with st.expander("See Results"):
                for s in st.session_state.search_result_p:
                    st.write(s)
                    # st.write(str(analyzer.polarity_scores(s)))
                    st.write('-------')
            
            result_all = search(term,1,total_number)
            search_result_all = [r['content'] for r in result_all[0]]
            latlon_all = [[float(x) for x in z['latlon'].split(',')] for z in result_all[0]]
            latlon_np = np.array(latlon_all).T
            latlon_df = pd.DataFrame({
                'lat':latlon_np[0],
                'lon':latlon_np[1]
            })
            # st.write(latlon_df)

            polarity = []
            for s in search_result_all:
                polarity.append(analyzer.polarity_scores(s)['compound'])


            fig = px.histogram(polarity)
            with st.expander('Sentiment Polarity Distribution'):
                st.plotly_chart(fig,use_container_width=True)
            with st.expander('map'):
                st.map(latlon_df)
            with st.expander('word cloud'):
                # Create some sample text
                # st.write(search_result_all)
                
                text = ''
                for rs in search_result_all:
                    text += ', '
                    text += rs
                # Create and generate a word cloud image:
                text = text.replace('https','')
                text = text.replace('t.co','')
                wordcloud = WordCloud().generate(text)

                pltfig = plt.figure(figsize=(20,10))
                # Display the generated image:
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.show()
                st.pyplot(pltfig)

else:
    if option=='More Like This':
        st.title('More Like This')     
        st.session_state.more_like_this = st.text_input('Term')
        if st.session_state.more_like_this:
            start = time.time()
            search_response = similar_search(st.session_state.more_like_this)
            st.session_state.time_spent = time.time()-start
            total_number = len(search_response)
            st.write(f'got {len(search_response)} results in {np.round(10000*st.session_state.time_spent,2)} ms')
            st.write('-------')

            # st.write(search_response)
            if not search_response:
                st.write('No Results!')
            else:
                c1, _ = st.columns([1,3])
                with c1:
                    PAGE_SIZE = 5
                    total_page = int(total_number/PAGE_SIZE)
                    page = int(st.number_input('Page',1,total_page))
                    # st.write(st.session_state.page)
                    result = search_response[(page-1)*PAGE_SIZE:page*PAGE_SIZE]
                    search_result = [r['content'] for r in result]
                    st.session_state.search_result_p = search_result

                with st.expander("See Results"):
                    for s in st.session_state.search_result_p:
                        st.write(s)
                        # st.write(str(analyzer.polarity_scores(s)))
                        st.write('-------')


                result_all = search_response
                search_result_all = [r['content'] for r in result_all]
                latlon_all = [[float(x) for x in z['latlon'].split(',')] for z in result_all]
                latlon_np = np.array(latlon_all).T
                latlon_df = pd.DataFrame({
                    'lat':latlon_np[0],
                    'lon':latlon_np[1]
                })
                # st.write(latlon_df)

                polarity = []
                for s in search_result_all:
                    polarity.append(analyzer.polarity_scores(s)['compound'])


                fig = px.histogram(polarity)
                with st.expander('Sentiment Polarity Distribution'):
                    st.plotly_chart(fig,use_container_width=True)
                with st.expander('map'):
                    st.map(latlon_df)
                with st.expander('word cloud'):
                    # Create some sample text
                    # st.write(search_result_all)
                    
                    text = ''
                    for rs in search_result_all:
                        text += ', '
                        text += rs
                    # Create and generate a word cloud image:
                    text = text.replace('https','')
                    text = text.replace('t.co','')
                    wordcloud = WordCloud().generate(text)

                    pltfig = plt.figure(figsize=(20,10))
                    # Display the generated image:
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    plt.show()
                    st.pyplot(pltfig)

    else:
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

                st.subheader('2.3 War-Prediction Detection')
                with open('./tf-pkl/attitude.pkl','rb') as f:
                    tf_side = pickle.load(f)
                text_tf= tf_side.transform([stemmed])
                with open('./sklearn-pkl/attitude/logistic_regression.pkl','rb') as f:
                    clf_side = pickle.load(f)
                y_side = clf_side.predict(text_tf)
                side_dict = {0:'Russia Will Lose',1:'Russia Will Win'}
                st.write(f'Logistic Regression: *{str(side_dict[y_side[0]])}*')

                st.subheader('2.4 Spatial Detection')
                with open('./tf-pkl/geo.pkl','rb') as f:
                    tf_spatial = pickle.load(f)
                text_tf= tf_spatial.transform([stemmed])[:,:5000]
                with open('./sklearn-pkl/geo/adaboost.pkl','rb') as f:
                    clf_spatial = pickle.load(f)
                y_side = clf_spatial.predict(text_tf)
                st.write(f'AdaBoost: *{y_side[0]}*')
        else:
            pass













    






