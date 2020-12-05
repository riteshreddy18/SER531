# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import os

        
import spacy
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus  import stopwords
import re
from gensim.utils import lemmatize
import matplotlib.pyplot as plt

import keras
from keras.models import load_model
from keras.layers import Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense,LSTM,Dropout,Conv1D,MaxPooling1D

# %% [code]
glove_dir="./glove6b100dtxt"

embedding_index={}
f=open(os.path.join(glove_dir,'glove.6B.100d.txt'),encoding='utf8')
for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    embedding_index[word]=coefs
f.close()
print('Found %s word vectors ' % len(embedding_index))

stop = stopwords.words('english')
def cleaning(df):
    df.loc[:,'text']=pd.DataFrame(df.loc[:,'text'].str.lower())
    df.loc[:,'text'] = [re.sub(r'[^a-zA-Z]',' ', i) for i in df.loc[:,'text']]
    df.loc[:,'text'] = [re.sub(r"\b[a-zA-Z]\b", ' ', i) for i in df.loc[:,'text']]
    df.loc[:,'text'] = [re.sub(' +',' ', i) for i in df.loc[:,'text']]
    return(df)
    
def lemmatization(df, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in df.loc[:,'text']:
        doc = nlp(sent)
        texts_out.append([token.lemma_ for token in doc])
    return(texts_out)

def textprocessing(df,Is_Test=1):
    df=cleaning(df)
    df['lemmatized_text_token']=lemmatization(df)
    df['lemmatized_text_token']=df['lemmatized_text_token'].apply(lambda x:[i for i in x if i not in (stop) ])
    if Is_Test:
        with open('./tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        tokenizer=Tokenizer(oov_token='<unknown>')
        tokenizer.fit_on_texts(df['lemmatized_text_token'])
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    word2id = tokenizer.word_index
    id2word = {v:k for k, v in word2id.items()}
    vocab_size = len(word2id) + 1 
    sequence=tokenizer.texts_to_sequences(df['lemmatized_text_token'])
    sequence=pad_sequences(sequence,maxlen=200)
    if Is_Test:
        return(sequence,word2id)
    else:
        return(sequence,vocab_size,word2id)

# %% [code]
def mainfunc(lists):    
    model = load_model('./model_BiLSTM (2).h5')
    hellodf = pd.DataFrame(lists)
    # for i in 
    test_1, word1id = textprocessing(hellodf,1)
    y_pred1 = model.predict_classes([test_1[:1]],verbose=0)
    print(lists)
    print(df_test['l2'][y_pred1])

# %% [code]
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser",'ner'])
# df_test=pd.read_csv('/kaggle/input/dbpedia-classes/DBPEDIA_test.csv',encoding='utf-8-sig')

# %% [code]
# hello = df_test['text']
# mainfunc(hello[i:i+1])

# %% [code]
i = "Myrmarachne paludosa, is a species of spider of the genus Myrmarachne. It is endemic to Sri Lanka."
def run(i):
    data = {'text':  [i]}
    hel = pd.DataFrame (data, columns = ['text'])
    print(mainfunc(hel))
run(i)