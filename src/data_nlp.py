from datasets import load_dataset
from transformers import RobertaTokenizer
from pathlib import Path

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def load_agnews(cache_dir = None):
	print(cache_dir)
	dataset = load_dataset('ag_news', cache_dir = cache_dir, download_mode = 'force_redownload')
	train_dataset, test_dataset = dataset['train'], dataset['test']
	train_cache_file_name = cache_dir + '/agnews_train'
	test_cache_file_name = cache_dir + '/agnews_test'
	train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], padding=True, truncation=True), batched=True, cache_file_name = train_cache_file_name)
	test_dataset = test_dataset.map(lambda examples: tokenizer(examples['text'], padding=True, truncation=True), batched=True, cache_file_name = test_cache_file_name) 
	return train_dataset, test_dataset


def load_imdb(cache_dir = None):
	dataset = load_dataset('imdb', cache_dir = cache_dir, download_mode = 'force_redownload')
	train_dataset, test_dataset = dataset['train'], dataset['test']
	train_cache_file_name = cache_dir + '/imdb_train'
	test_cache_file_name = cache_dir + '/imdb_test'
	train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], padding=True, truncation=True), batched=True, cache_file_name = train_cache_file_name)
	test_dataset = test_dataset.map(lambda examples: tokenizer(examples['text'], padding=True, truncation=True), batched=True, cache_file_name = test_cache_file_name) 
	return train_dataset, test_dataset


def load_hatespeech(cache_dir = None):
	dataset = load_dataset('hate_speech18', cache_dir = cache_dir, download_mode = 'force_redownload')
	dataset = dataset['train'].train_test_split(test_size=0.2)
	train_dataset, test_dataset = dataset['train'], dataset['test'] 
	train_cache_file_name = cache_dir + '/hatespeech_train'
	test_cache_file_name = cache_dir + '/hatespeech_test'
	train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], padding=True, truncation=True), batched=True, cache_file_name = train_cache_file_name)
	test_dataset = test_dataset.map(lambda examples: tokenizer(examples['text'], padding=True, truncation=True), batched=True, cache_file_name = test_cache_file_name) 
	return train_dataset, test_dataset


def load_yahoo(cache_dir = None):
	dataset = load_dataset('yahoo_answers_topics', cache_dir = cache_dir, download_mode = 'force_redownload')
	train_dataset, test_dataset = dataset['train'], dataset['test']

	train_dataset = train_dataset.rename_column('question_title', 'text')
	train_dataset = train_dataset.rename_column('topic', 'label')
	test_dataset = test_dataset.rename_column('question_title', 'text')
	test_dataset = test_dataset.rename_column('topic', 'label')

	train_cache_file_name = cache_dir + '/yahoo_train'
	test_cache_file_name = cache_dir + '/yahoo_test'
	train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], padding=True, truncation=True), batched=True, cache_file_name = train_cache_file_name)
	test_dataset = test_dataset.map(lambda examples: tokenizer(examples['text'], padding=True, truncation=True), batched=True, cache_file_name = test_cache_file_name) 
	return train_dataset, test_dataset


def count_labels(dataset):
	label_set = {entry['label'] for entry in dataset}
	num_labels = len(label_set)
	return num_labels


def load_datasets(dataset_name, seed, cache_dir = None):
	if dataset_name == 'agnews':
		train_dataset, test_dataset = load_agnews(cache_dir = cache_dir)
	elif dataset_name == 'imdb':
		train_dataset, test_dataset = load_imdb(cache_dir = cache_dir)
	elif dataset_name == 'hatespeech':
		train_dataset, test_dataset = load_hatespeech(cache_dir = cache_dir)
	elif dataset_name == 'yahoo':
		train_dataset, test_dataset = load_yahoo(cache_dir = cache_dir)
	else:
		raise NotImplementedError

	num_labels = count_labels(test_dataset)
	
	train_dataset = train_dataset.shuffle(seed = seed)
	
	return train_dataset, test_dataset, num_labels 
