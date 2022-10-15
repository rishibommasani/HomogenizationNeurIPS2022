import numpy as np
import torch
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from transformers import RobertaTokenizer
from datasets import load_metric
device = "cuda" if torch.cuda.is_available() else "cpu"

name2param_name = {'roberta-base' : 'roberta', 'roberta-large' : 'roberta', 'bert-base-uncased' : 'bert'}


def adaptation(model, adaptation_method, model_name):
	model_parameter_name = name2param_name[model_name]
	if adaptation_method == 'finetuning':
		pass
	elif adaptation_method == 'probing':
		for n, p in model.named_parameters():
			if model_parameter_name in n:
				p.requires_grad = False
	elif adaptation_method == 'bitfit':
		for n, p in model.named_parameters():
			if 'bias' in n and model_parameter_name in n:
				p.requires_grad = False
	else:
		raise NotImplementedError
	return model


def format_predictions(predictions_tuple, test_dataset):
	predictions, labels, metrics = predictions_tuple
	predictions = [np.argmax(row) for row in predictions]
	predictions_test_dataset = test_dataset.add_column('predictions', predictions)
	return predictions_test_dataset


def train(dataset_name, train_dataset, test_dataset, num_labels, adaptation_method, seed, output_dir, cache_dir = None, model_name = 'roberta-base'):
	model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels, cache_dir = cache_dir)
	model.to(device)
	model = adaptation(model, adaptation_method, model_name)
	
	training_args = TrainingArguments("test_trainer")
	training_args.seed = seed
	training_args.learning_rate = 0.00002
	training_args.output_dir = '{}/{}/{}/{}_{}'.format(output_dir, dataset_name, adaptation_method, model_name, seed)
	training_args.save_strategy = 'epoch'
	training_args.disable_tqdm = True

	tokenizer = AutoTokenizer.from_pretrained(model_name)

	metric = load_metric("accuracy")
	def compute_metrics(eval_pred):
		logits, labels = eval_pred
		predictions = np.argmax(logits, axis=-1)
		return metric.compute(predictions=predictions, references=labels)

	trainer = Trainer(
	model = model,
	args=training_args,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	tokenizer=tokenizer,
	compute_metrics=compute_metrics,
	)

	trainer.train()
	print('Saving model')
	trainer.save_model('checkpoints/{}/{}/{}_{}'.format(dataset_name, adaptation_method, model_name, seed))
	predictions_tuple = trainer.predict(test_dataset=test_dataset)
	predictions_test_dataset = format_predictions(predictions_tuple, test_dataset)
	predictions_test_dataset.save_to_disk('predictions/{}/{}/{}_{}'.format(dataset_name, adaptation_method, model_name, seed))
	return predictions_test_dataset
