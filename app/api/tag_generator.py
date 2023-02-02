import numpy as np
import pandas as pd
import torch
# import scipy
import pickle
# from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# import tensorflow_hub as hub

class TagGenerator:
    def __init__(self):
        pass

    def preprocess_fct(self, title,text):
        title = BeautifulSoup(title, features='html.parser').get_text().lower()
        text = BeautifulSoup(text, features='html.parser').get_text().lower()
        tokenizer = nltk.RegexpTokenizer(r'[a-zA_\-+#]*\.?[a-zA_\+#]+')
        tokens_list_title = tokenizer.tokenize(title)
        tokens_list_text = tokenizer.tokenize(text)
        english_stop_words= stopwords.words('english')
        clean_tokens_list_title = [word for word in tokens_list_title if word not in english_stop_words]
        clean_tokens_list_text = [word for word in tokens_list_text if word not in english_stop_words]
        trans = WordNetLemmatizer()
        trans_title = [trans.lemmatize(word) for word in clean_tokens_list_title]
        trans_text = [trans.lemmatize(word) for word in clean_tokens_list_text]
        final_text = trans_title + trans_text

        return " ".join(final_text)

    def tfidf_embedding(self, text):
        loaded_vec = TfidfVectorizer(decode_error='replace', vocabulary=pickle.load(open("feature.pkl","rb")))
        embedding = loaded_vec.fit_transform([text])
        dense_array = embedding.toarray()
        features = pickle.load(open("feature.pkl","rb"))
        doc_df = pd.DataFrame(dense_array)
        return doc_df

    def predict_emdedded_matrix(self, doc_df):
        # print('loading the MultiOutputClassifier model with torch ... ')
        loaded_model = torch.load('tfidf_model_2.pt')
        print('Predicting the tags for your title and text  ...  ')
        pred = loaded_model.predict(doc_df)
        doc_pred = pd.DataFrame(pred)
        target_labels = pd.read_csv('tfidf_labels_tags.csv')
        array_tags = []
        for row in target_labels.values:
            array_tags.append(row[0])
        doc_pred.columns = array_tags
        doc_pred = doc_pred.T[0]
        doc_tags = doc_pred[doc_pred == 1]
        print('doc_tags : ',doc_tags)
        print('type of doc_tags : ', type(doc_tags))
        print('\n Done')
        print('array of tags : ',np.array(doc_tags.index))
        array = np.array(doc_tags.index)
        print(" ".join(array))
        return " ".join(array)

    def generate_tag(self, title, text):
        print('1/ Preprocessing the title & Text ...')
        #call the preprocessing fonction
        text = self.preprocess_fct(title, text)
        print(' ')
        print('\n Done')
        print('2/ Transforming the Text into an embedding matrix ... ')
        doc_df = self.tfidf_embedding(text)
        print('\n Done')
        print('3/ Load the MultiOutputClassifier and 126 target_labels to predict the tags of the input text ...')
        return self.predict_emdedded_matrix(doc_df)