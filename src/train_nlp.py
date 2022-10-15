from quinine import Quinfig, QuinineArgumentParser
from schema import  get_train_schema
from adaptation_nlp import train
from data_nlp import load_datasets
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
	quinfig = QuinineArgumentParser(schema=get_train_schema()).parse_quinfig()
	seed = quinfig.seed
	output_dir = quinfig.output_dir
	model_name = quinfig.model_name
	cache_dir = None

	print(seed, output_dir, cache_dir, model_name)

	for dataset_name in tqdm(['agnews', 'imdb', 'hatespeech', 'yahoo']): 
		train_dataset, test_dataset, num_labels = load_datasets(dataset_name, seed, cache_dir = cache_dir)
		for adaptation_method in tqdm(['probing', 'bitfit', 'finetuning']):
			train(dataset_name, train_dataset, test_dataset, num_labels, adaptation_method, seed, output_dir, cache_dir = cache_dir, model_name = model_name)
