import numpy as np 
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import datasets
import transformers


from pypdf import PdfReader
def reader (path):
    reader = PdfReader(path)
    text=""
    for page in reader.pages:
        text+=page.extract_text()
    return text



from nltk import pos_tag
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import string
import re

puncuation=set(string.punctuation)
stop_words_english=set(stopwords.words("english"))
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    sentences = sent_tokenize(text)
    features = {'feature': ""}

    for sent in sentences:
        for criteria in ['skills', 'education','job','experience']:
            if criteria in sent:
                words = word_tokenize(sent)
                words = [word for word in words if word not in stop_words_english]
                # POS tagger to identify and remove stop words and other irrelevant words
                tagged_words = pos_tag(words)
                filtered_words = [word for word, tag in tagged_words if tag not in ['DT', 'IN', 'TO', 'PRP', 'WP']]
                features['feature'] += " ".join(filtered_words)

    return features


print (preprocess_text(reader(r"data\ACCOUNTANT\10554236.pdf")))