## thanks to marijavlajic@github, KahEm Chu@tds, Selva Prabhakaran@ml+ and JairParra@github for code ideas and guidelines

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
import os
import re

## Streamlit
import streamlit as st
from streamlit import components
import altair as alt

# Text Preprocessing NLTK
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.data.path.append('NLTK/stopwords')
nltk.data.path.append('NLTK/punkt')
stopwords = stopwords.words('spanish')

## Text Preprocessing spaCy
import spacy
nlp = spacy.load('SPACY/es_core_news_md')

## LDA-Model
import gensim
from gensim import models, corpora
from gensim.models import Phrases
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from pprint import pprint


## Visualization
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models

## Set global variables
data_path = os.path.join(os.getcwd(),'data')
doc_path = os.path.join(data_path,'interviews_per_topic')

# Write a title
st.title('**Topic Modeling of semi-structured interviews**')
st.write('LDA – Latent Dirichlet Allocation')

def clean_tokens(list_tokens):
    """Function for preprocessing text containing following steps:
    - convert all tokens to lower case
    - remove stopwords
    - remove any value that are not alphabetical
    - remove words with less than 3 characters
    - remove words that only contains consonants
    ===
    list_tokens - list of tokens
    """
    preprocessed_tokens=[]
    for token in list_tokens:
        token = token.lower()
        if not token in stopwords: 
            token = re.sub(r'[¿\?\:!\.,;\(\)\'\"…]+', '', token) 
            if len(token) > 2: 
                vowels=len([v for v in token if v in "áaéeíióoúu"])
                if vowels != 0: 
                    preprocessed_tokens.append(token)
    return(preprocessed_tokens)


def load_interview_files(doc_path, interview_topic, mode='no stemming'):
    """Function to read in and interviews files and output as list of lists.
    Takes the path to the directory, where interviews are stored, the name of interview topic and one of two modes.
    Mode 'no stemming' calls function 'clean_tokens' to preprocess text (Default).
    Mode 'lemmization' uses lemmization-package of spaCy and removes stopwords to preprocess text. 
    ===
    doc_path - path
    interview_topic - str
    mode - str
    """
    f'Using mode: {mode}.'
    list_interview_per_topic = []
    list_interview_per_id = []
    for filename in sorted(os.listdir(doc_path)):
        if filename.find(interview_topic) != -1:
            file_id = int(filename[0:2])
            list_interview_per_id.append(file_id)
            with open(os.path.join(doc_path,filename), 'r') as my_file:
                #print(f'reading {filename}, {file_id}')
                if mode == 'lemmization':
                    interview_text = nlp(my_file.read())
                    interview_tokens = [token.lemma_ for token in interview_text if token.text.isalpha()]
                    interview_tokens = [token.lower() for token in interview_tokens]
                    preprocessed_tokens = [token for token in interview_tokens if token not in stopwords and len(token) > 2]
                    list_interview_per_topic.append(preprocessed_tokens)
                elif mode == 'no stemming':
                    interview_tokens = word_tokenize(my_file.read(), language='spanish')
                    preprocessed_tokens = clean_tokens(interview_tokens)
                    list_interview_per_topic.append(preprocessed_tokens)
                else:
                    print(f"Error: Mode '{mode}' not found.")
                    break
    f'Found {len(list_interview_per_topic)} texts for topic {interview_topic}.'
    return(list_interview_per_topic, list_interview_per_id)

def add_ngrams(data, min_count=5):
    """Function to add bigrams and trigrams to tokenlists for phrases that appear n times or more.
        Takes tokenized text and times a phrase has to appear in the text to be transformed in n-gram.
        ===
        data = lists of tokens
        min_count = int, default = 5
    """
    bigram = Phrases(data, min_count) 
    trigram = Phrases(bigram[data], min_count)
    return [trigram[doc] for doc in data]


def remove_cus_stopwords(data, filename):
    """Function to add customized list of words and phrases to remove from tokenized text.
    Takes tokenized text and a text file of stopwords.
    ===
    data = lists of tokens
    filename = new line separated word list
    """
    with open(os.path.join(data_path,filename), 'r') as my_file:
        list_cus_stopwords = my_file.readlines()
        list_cus_stopwords = [word.strip() for word in list_cus_stopwords]
        st.text(', '.join(list_cus_stopwords))
        data_no_stopwords = []
        for list_token in data:
            data_no_stopwords.append([token for token in list_token if token not in list_cus_stopwords])
        return data_no_stopwords


def create_dict_corpus(data, no_below=0, no_above=0):
    """Function to create dictionary and corpus for LdaModel. 
    Can filter out tokens in the dictionary by their frequency. Default is no filtering.
    Takes tokenized text,
    no_below: Keep tokens which are contained in at least no_below documents.
    no_above: Keep tokens which are contained in no more than no_above documents (in %, 0 < no_above < 1)
    ==
    data = lists of token
    no_below = int, optional
    no_above = float, optional
    """
    dictionary = corpora.Dictionary(data)
    if no_below != 0 or no_above != 0:
        dictionary.filter_extremes(no_below, no_above)
    corpus = [dictionary.doc2bow(text) for text in data]
    print(len(dictionary), len(corpus))
    return dictionary, corpus


def format_topics_sentences(ldamodel, corpus, list_of_ids):
    """Function that creates dataframe with follwoing columns:
    section of interview, number of interview, number of topic, contributon of topic to specific interview section in %, keywords of topic
    Takes LDA model and corpus model was trained on.
    ===
    ldamodel = pretrained LdaModel
    corpus = LdaModel.corpus
    list_of_ids = list of topics per interview section and corresponding interview IDs
    """
    i = 0
    df_topics_per_interview = pd.DataFrame(columns=['Interview_Section' ,'Interview_Id', 'Topic_Number', 'Contrib_Perc', 'Keywords'])

    for idx, topic in enumerate(ldamodel.get_document_topics(corpus)):
        for contrib in topic:  
            #print(idx)   
            topic_num = contrib[0]
            prop_topic = contrib[1]
            word_prop = ldamodel.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in word_prop])
            df_topics_per_interview.loc[i] = [list_of_ids[idx][0], list_of_ids[idx][1], int(topic_num), round(prop_topic,4), topic_keywords]
            i += 1
    return(df_topics_per_interview)


## thanks to Khuyen Tran for this function: 
## https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know
def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        f'Computing LDA with {num_topics} topics…'
        model = LdaModel(corpus=corpus, id2word=dictionary, chunksize=500, \
         alpha='auto', eta='auto', random_state=42, \
         iterations=400, num_topics=num_topics, \
         passes=20, eval_every=1)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    number_topics = [*range(start,limit,step)]
    chart_data = pd.DataFrame({'coherence values': coherence_values}, index=number_topics)
    #plt.xlabel("Number of Topics")
    #plt.ylabel("Coherence score")
    #st.line_chart(chart_data)
    

    #chart = (
    #    alt.Chart(
    #        data=chart_data,
    #        title="Coherence Score for n number of topics"
    #    )
    #    .mark_line()
    #    .encode(
    #        x=alt.X('number_topics', axis=alt.Axis(title="Number of Topics")),
    #        y=alt.X('coherence values', axis=alt.Axis(title="Coherence score"))
    #    )
    #)
    chart= (
        alt.Chart(
            data= chart_data.reset_index(),
            title="Coherence Score for n number of topics"
            )
        .mark_line()
        .encode(
            x=alt.X('index', axis=alt.Axis(title="Number of Topics")),
            y=alt.X('coherence values', axis=alt.Axis(title="Coherence score"), scale=alt.Scale(zero=False))
        )
    )
    st.altair_chart(chart)
    return model_list, coherence_values

def main():
    ## Create list of topics per interview sections
    list_of_topics = ['ProblemasDemocraciaActual','AlternativasSociedad','AlternativasMovimiento','FormasDeOrganizacion','TeoriasPoliticas','UtopiaDistopia']

    ## Call function 'load_interview_files' to create dictionary of topics per interview section topics and tokens per interview section.
    ## Create list of topics per interview section and corresponding interview IDs.
    dict_topics = {}
    list_ids = []
    count = 0
    for topic in list_of_topics:
        dict_topics[topic], topic_ids = load_interview_files(doc_path, topic)
        for id in topic_ids:
            list_ids.append([topic,id])
        count += len(dict_topics[topic])
    ## Verbose mode
    f'Dictionary contains {count} interview parts related to {len(dict_topics)} topics.'

    ## Create list of token
    preprocessed_data = []
    for topic in list_of_topics:
        preprocessed_data  += dict_topics[topic]

    ## Call function 'add_ngrams'
    preprocessed_data = add_ngrams(preprocessed_data, 5)

    ## Call function'remove_cus_stopwords'
    preprocessed_data = remove_cus_stopwords(preprocessed_data, 'stopwords_nostemming.txt')

    ## Create dictionary and corpus as input to LdaModel
    dictionary, corpus = create_dict_corpus(preprocessed_data)

    ## Train LDA-Model
    num_topics = 92
    chunksize = 500  # size of the doc looked at every pass
    passes = 20 # number of passes through documents
    iterations = 400
    eval_every = 1  # Don't evaluate model perplexity, takes too much time

    lda_model = LdaModel(corpus=corpus, id2word=dictionary, chunksize=chunksize, \
                       alpha='auto', eta='auto', random_state=42, \
                       iterations=iterations, num_topics=num_topics, \
                       passes=passes, eval_every=eval_every)
    pprint(lda_model.print_topics(num_words=12))

    ## Create table to get keywords per topic related to section per interview.
    df_topics_per_interview = format_topics_sentences(ldamodel=lda_model, corpus=corpus, list_of_ids=list_ids)
    st.dataframe(df_topics_per_interview)
    
    ## Get word-probability pair for a selected topic. Takes topic_id (int) and number of the most significant words that are associated with the topic (int).
    df_words_per_topic = pd.DataFrame(lda_model.show_topic(21,15)) 
    st.table(df_words_per_topic)

    ## Display a Intertopic Distance Map of topics using pyLDAvis
    prepared_model_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, mds='mmds')
    pyLDAvis.save_html(prepared_model_data, 'pyLDAvis.html')

    with open('./pyLDAvis.html', 'r') as f:
        html_string = f.read()
        components.v1.html(html_string, width=1300, height=800, scrolling=False)

    ## Calculate and display coherence score of LdaModel
    coherence_model_lda = CoherenceModel(model=lda_model, texts=preprocessed_data, corpus=corpus)
    coherence_lda = coherence_model_lda.get_coherence()
    f'Coherence Score: {coherence_lda}'   

    ## Compute and display coherence scores for given number of topics
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=preprocessed_data, start=20, limit=61, step=8)

main()
