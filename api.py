import pandas as pd
import numpy as np
import requests
import re
import nltk
from nltk.corpus import stopwords
from flask import Flask,jsonify,request

import string

import summa
from summa import summarizer

from nltk import tokenize


stopword=nltk.corpus.stopwords.words('english')

def analyser(r,d,p):

    s=tokenize.sent_tokenize(r)
    p['Sentences']=s


    summary=(summarizer.summarize(r))
    p['Summary']=summary #1

    noun=[]

    for sentence in s:
        for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                noun.append(word)
    p['Nouns']=noun[0:10] #2
    verb=[]
    for sentence in s:
        for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
            if (pos == 'VB' or pos == 'VDB' or pos == 'VBG' or pos == 'VBN' or pos =='VBP' or pos =='VBZ'):
                verb.append(word)
    p['Verbs']=verb[0:10]  #3

    def rem_p(text):
        text_nop="".join([char for char in text if char not in string.punctuation])
        return text_nop

    d['body_text_clean']=d['body_text'].apply(lambda x: rem_p(x))


    def tokeniz(text):
        tokens=re.split('\W+',text)
        return tokens
    d['body_text_tokenized']=d['body_text_clean'].apply(lambda x:tokeniz(x.lower()))

    def remove_stopwords(tokenized_list):
        text=[word for word in tokenized_list if word not in stopword]
        return text
    d['body_text_nostop']=d['body_text_tokenized'].apply(lambda x:remove_stopwords(x))

    ps=nltk.PorterStemmer()

    def stemming(tokenized_text):
        text=[ps.stem(word)for word in tokenized_text]
        return text
    d['body_text_stemmed']=d['body_text_nostop'].apply(lambda x:stemming(x))
    st=d['body_text_stemmed'].tolist()
    p['Stems']=st[0:1]

    wn=nltk.WordNetLemmatizer()
    def lemmatizing(tokenized_text):
        text=[wn.lemmatize(word) for word in tokenized_text]
        return text
    d['body_text_lemmatized']=d['body_text_nostop'].apply(lambda x:lemmatizing(x))
    l=d['body_text_lemmatized'].tolist()
    p['Lemma']=l[0:1]

    return p
#send raw data from postman in json format with text data in key "text"
app= Flask(__name__)
@app.route('/api',methods=['POST'])
def ana():
    data=request.get_json()
    p={}
    r=data['text']
    d=pd.DataFrame(r.split('\n'), columns=['body_text'])
    output=analyser(r,d,p)

    return jsonify(results=output)    

if __name__=='__main__':
    app.run(port=9000,debug=True)

