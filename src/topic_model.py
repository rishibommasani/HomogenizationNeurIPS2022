from data_nlp import load_datasets
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def learn_topic_model(datasets, num_topics):
	data = []
	max_docs = 10000
	max_len = 200
	for dataset in tqdm(datasets):
		for i, row in tqdm(enumerate(dataset)):
			if i < max_docs:
				doc = row['text']
				sentences = sent_tokenize(doc)
				preprocessed_sentences = [gensim.utils.simple_preprocess(sentence, deacc=True) for sentence in sentences]
				no_stop_sentences = [[word for word in sentences if word not in stop_words] for sentence in preprocessed_sentences]
				flat_doc = [item for sublist in no_stop_sentences for item in sublist][:max_len]
				print(type(flat_doc), type(flat_doc[0]))
				data.append(flat_doc)
			else:
				break


	print('Number of documents: {}'.format(len(data)))
	for i in range(0, len(data), max_docs // 2):
		print(data[i])

	id2word = corpora.Dictionary(data)
	corpus = [id2word.doc2bow(text) for text in data]

	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
	print(lda_model.print_topics())


if __name__ == '__main__':
	dataset_names = ['agnews', 'imdb', 'hatespeech', 'yahoo']
	seed = 1
	cache_dir = 'src/topic_models'
	datasets = [load_datasets(dataset_name, seed, cache_dir = cache_dir)[0] for dataset_name in dataset_names]
	num_topics = 10
	topic_model = learn_topic_model(datasets, num_topics)