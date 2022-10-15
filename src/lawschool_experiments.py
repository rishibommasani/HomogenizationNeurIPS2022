import random
import numpy as np
import pandas as pd
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm, ensemble
from sklearn.neural_network import MLPClassifier
from tempeh.configurations import datasets
from homogenization import measure_homogenization, aggregate_measurements
from tqdm import tqdm
import pickle
import argparse


def format_data(lawschool_data, applications_list, seed):
	applications_data = {}
	# Construct data for each application
	for application_name in applications_list:
		dataset = lawschool_data[application_name]
		X_train, X_test = dataset.get_X()
		
		y_train, y_test = dataset.get_y()
		if application_name == 'gpa':
			mean_gpa = (sum(y_train) + sum(y_test)) / (len(y_train) + len(y_test))
			y_train, y_test = y_train > mean_gpa, y_test > mean_gpa
			y_train, y_test = y_train.astype(int), y_test.astype(int)
		y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
		
		# Skipping group metadata for now since splits are already given and its hard to recover a split on z
		# z_train, z_test = dataset.get_sensitive_features()
		
		applications_data[application_name] = {'X_tr' : X_train, 'X_test' : X_test, 'y_tr': y_train, 'y_test' : y_test}

	return applications_data


def fixed_partition(applications_data, applications_list, seed, data_scale):
	rng = random.Random(seed)
	index = rng.randint(0, data_scale - 1)
	
	for name, entries in applications_data.items():
		X_train, y_train = entries['X_tr'], entries['y_tr']
		# y_train = np.expand_dims(y_train, axis = 1)
		# data = np.concatenate((X_train, y_train), axis = 1)
		# np.random.RandomState(seed=seed).shuffle(data)
		# X_train, y_train = data[:, : -1], data[:, -1]
		
		N = len(y_train)
		block_length = N // data_scale
		start, end = block_length * index, block_length * index + block_length
		entries['X_tr'], entries['y_tr'] = X_train[start : end], y_train[start : end]
		
	return applications_data


def disjoint_partition(applications_data, applications_list, seed, data_scale):
	k = len(applications_list)
	permutation = np.random.RandomState(seed=seed).permutation(data_scale)
	
	for position, (name, entries) in enumerate(applications_data.items()):
		X_train, y_train = entries['X_tr'], entries['y_tr']
		# y_train = np.expand_dims(y_train, axis = 1)
		# data = np.concatenate((X_train, y_train), axis = 1)
		# np.random.RandomState(seed=seed).shuffle(data)
		# X_train, y_train = data[:, : -1], data[:, -1]
		
		N = len(y_train)
		block_length = N // data_scale
		index = permutation[position % data_scale]
		start, end = block_length * index, block_length * index + block_length
		entries['X_tr'], entries['y_tr'] = X_train[start : end], y_train[start : end]
	
	return applications_data


def generate_predictions(data, method, seed, predict_train=False):
	if method == 'logistic':
		core_model = LogisticRegression
	elif method == 'gbm':
		core_model = ensemble.GradientBoostingClassifier
	elif method == 'svm':
		core_model = svm.SVC
	elif method == 'nn':
		core_model = MLPClassifier
	else:
		raise NotImplementedError
	
	model = make_pipeline(StandardScaler(), core_model(random_state=seed))
	model.fit(data['X_tr'], data['y_tr'])
	test_predictions = model.predict(data['X_test'])
	if predict_train:
		train_predictions = model.predict(data['X_tr'])
		return {'train_predictions' : train_predictions, 'test_predictions' : test_predictions}

	return test_predictions


def group_and_format(applications_data, grouping):
	reformatted_applications_data = {}
	groups = set()
	for name, data in applications_data.items():
		reformatted_data = {}
		X, y, yhat = data['X_test'], data['y_test'], data['predictions']
		assert X.shape[0] == y.shape[0] == yhat.shape[0]

		for i in range(X.shape[0]):
			if grouping == 'individual':
				group = i
			else:
				raise NotImplementedError

			entry = {'input' : X[i], 'label' : y[i], 'prediction' : yhat[i], 'group' : group}
			
			# Entry id is index of entry in this case
			reformatted_data[i] = entry 
			groups.add(group)

		reformatted_applications_data[name] = reformatted_data

	return reformatted_applications_data, groups


# Experiment to test role of data partition
def partition_experiment(lawschool_data, applications_list, data_scale, model_seeds, partition_seeds, method = 'logistic', groupings = ['individual']):
	print('Running partition lawschool experiment for data scale {}'.format(data_scale))

	data_seed = 0
	model_seeds = list(range(model_seeds))
	partition_seeds = list(range(partition_seeds))

	partition_table = {}
	# Results for fixed 1/k training data where k = num. applications
	print('Training fixed models')
	
	seed2measurements = {grouping : {} for grouping in groupings}
	for model_seed in tqdm(model_seeds):
		for partition_seed in partition_seeds:
			# Generate predictions
			applications_data = format_data(lawschool_data, applications_list, data_seed)
			applications_data = fixed_partition(applications_data, applications_list, partition_seed, data_scale)

			for name, data in applications_data.items():
				predictions = generate_predictions(data, method, model_seed)
				data['predictions'] = predictions 
			
			for grouping in groupings:
				# Group inputs and reformat data to prepare for homogenization measurement
				reformatted_applications_data, groups = group_and_format(applications_data, grouping)
				homogenization_measurements = measure_homogenization(reformatted_applications_data, groups)
				seed2measurements[grouping][(model_seed, partition_seed)] = homogenization_measurements
	for grouping in groupings:
		aggregate_homogenization_measurements = aggregate_measurements(seed2measurements[grouping])
		partition_table[('fixed', grouping)] = aggregate_homogenization_measurements

	# Results for disjoint 1/k training data where k = num. applications
	print('Training disjoint models')
	seed2measurements = {grouping : {} for grouping in groupings}
	for model_seed in tqdm(model_seeds):
		for partition_seed in partition_seeds:
			# Generate predictions
			applications_data = format_data(lawschool_data, applications_list, data_seed)
			applications_data = disjoint_partition(applications_data, applications_list, partition_seed, data_scale)

			for name, data in applications_data.items():
				predictions = generate_predictions(data, method, model_seed)
				data['predictions'] = predictions 
				
			
			for grouping in groupings:
				# Group inputs and reformat data to prepare for homogenization measurement
				reformatted_applications_data, groups = group_and_format(applications_data, grouping)
				
				homogenization_measurements = measure_homogenization(reformatted_applications_data, groups)
				seed2measurements[grouping][(model_seed, partition_seed)] = homogenization_measurements
	for grouping in groupings:
		aggregate_homogenization_measurements = aggregate_measurements(seed2measurements[grouping])
		partition_table[('disjoint', grouping)] = aggregate_homogenization_measurements	

	return partition_table


if __name__ == '__main__':
	print('Loading data')
	passbar = datasets['lawschool_passbar']()
	gpa = datasets['lawschool_gpa']()

	lawschool_data = {'passbar' : passbar, 'gpa' : gpa}
	
	applications_list = ['passbar', 'gpa']
	
	model_seeds, partition_seeds = 5, 5

	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--methods', help='List of methods/learning algorithms', type=str)
	args = parser.parse_args()
	methods = args.methods
	if not methods:
		methods = ['nn', 'logistic']
	else:
		methods = methods.split(',')
	
	print('Running lawschool experiments')
	for i, method in enumerate(methods): 
		print(method)
		for data_scale in tqdm(list(set(range(30, 5, -1)))):
			results = partition_experiment(lawschool_data, applications_list, data_scale, model_seeds, partition_seeds, method = method)
			print('\n')
			print(data_scale)
			print('\n')
			pickle.dump(results, open("results/lawschool_rush_partition_{}_{}x{}_{}.pkl".format(method, model_seeds, partition_seeds, data_scale), "wb"))