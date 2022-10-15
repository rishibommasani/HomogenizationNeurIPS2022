import random
import numpy as np
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm, ensemble
from sklearn.neural_network import MLPClassifier
from folktables import ACSDataSource, ACSEmployment, ACSIncomePovertyRatio, ACSHealthInsurance
from homogenization import measure_homogenization, aggregate_measurements
from tqdm import tqdm
import pickle
import argparse


def format_data(acs_data, applications_list, seed):
	name2application = {'employment' : ACSEmployment, 'income-poverty' : ACSIncomePovertyRatio, 'health-insurance' : ACSHealthInsurance}
	data_matrices = []

	# Construct data for each application
	for application_name in applications_list:
		application = name2application[application_name]
		application_matrices = application.df_to_numpy(acs_data)
		data_matrices.extend(application_matrices)
	
	# Unified train/test split across applications
	split_matrices = train_test_split(*data_matrices, test_size = 0.2, random_state = seed) 
	assert len(split_matrices) % 6 == 0
	
	# Package data into applications_data = {name: {X_tr, X_test, y_tr, y_test, z_tr, z_test}}
	applications_data = {}
	for i in range(0, len(split_matrices), 6):
		name = applications_list[i // 6]
		X_train, X_test = split_matrices[i], split_matrices[i+1]
		y_train, y_test = split_matrices[i+2], split_matrices[i+3]
		z_train, z_test = split_matrices[i+4], split_matrices[i+5]
		applications_data[name] = {'X_tr' : X_train, 'X_test' : X_test, 'y_tr': y_train, 'y_test' : y_test, 'z_tr' : z_train, 'z_test' : z_test}
	return applications_data


def fixed_partition(applications_data, applications_list, seed, data_scale):
	rng = random.Random(seed)
	index = rng.randint(0, data_scale - 1)
	
	for name, entries in applications_data.items():
		X_train, y_train, z_train = entries['X_tr'], entries['y_tr'], entries['z_tr']		
		y_train, z_train = np.expand_dims(y_train, axis = 1), np.expand_dims(z_train, axis = 1)
		data = np.concatenate((X_train, y_train), axis = 1)
		data = np.concatenate((data, z_train), axis = 1)
		np.random.RandomState(seed=seed).shuffle(data)
		X_train, y_train, z_train = data[:, : -2], data[:, -2], data[:, -1]

		N = len(y_train)
		block_length = N // data_scale
		start, end = block_length * index, block_length * index + block_length
		entries['X_tr'], entries['y_tr'], entries['z_tr'] = X_train[start : end], y_train[start : end], z_train[start : end]
		
	return applications_data


def disjoint_partition(applications_data, applications_list, seed, data_scale):
	k = len(applications_list)
	permutation = np.random.RandomState(seed=seed).permutation(data_scale)
	
	for position, (name, entries) in enumerate(applications_data.items()):
		X_train, y_train, z_train = entries['X_tr'], entries['y_tr'], entries['z_tr']
		y_train, z_train = np.expand_dims(y_train, axis = 1), np.expand_dims(z_train, axis = 1)
		data = np.concatenate((X_train, y_train), axis = 1)
		data = np.concatenate((data, z_train), axis = 1)
		np.random.RandomState(seed=seed).shuffle(data)
		X_train, y_train, z_train = data[:, : -2], data[:, -2], data[:, -1]

		N = len(y_train)
		block_length = N // data_scale
		index = permutation[position % data_scale]
		start, end = block_length * index, block_length * index + block_length
		entries['X_tr'], entries['y_tr'], entries['z_tr'] = X_train[start : end], y_train[start : end], z_train[start : end]
	
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
		X, y, z, yhat = data['X_test'], data['y_test'], data['z_test'], data['predictions']
		assert X.shape[0] == y.shape[0] == z.shape[0] == yhat.shape[0]

		for i in range(X.shape[0]):
			if grouping == 'individual':
				group = i
			elif grouping == 'race':
				group = z[i]
			else:
				raise NotImplementedError

			entry = {'input' : X[i], 'label' : y[i], 'prediction' : yhat[i], 'group' : group}
			
			# Entry id is index of entry in this case
			reformatted_data[i] = entry 
			groups.add(group)

		reformatted_applications_data[name] = reformatted_data

	return reformatted_applications_data, groups


# Experiment to test role of data partition
def partition_experiment(acs_data, applications_list, data_scale, model_seeds, partition_seeds, method = 'logistic', groupings = ['individual', 'race']):
	print('Running partition census experiment for data scale {}'.format(data_scale))

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
			applications_data = format_data(acs_data, applications_list, data_seed)
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
			applications_data = format_data(acs_data, applications_list, data_seed)
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
	# Fetch ACS data
	root_dir = 'data'
	data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', root_dir = root_dir)

	print('Loading data')
	acs_data = data_source.get_data(download=False)

	applications_list = ['employment', 'income-poverty', 'health-insurance']
	
	model_seeds, partition_seeds = 5, 5

	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--methods', help='List of methods/learning algorithms', type=str)
	args = parser.parse_args()
	methods = args.methods
	if not methods:
		methods = ['logistic']
	else:
		methods = methods.split(',')

	print('Running experiments')
	for i, method in enumerate(methods): 
		print(method)
		for data_scale in tqdm([25000, 10000, 7500, 5000, 2500, 1000, 750, 500, 250, 100, 50, 10]):
			results = partition_experiment(acs_data, applications_list, data_scale, model_seeds, partition_seeds, method = method)
			print('\n')
			print(data_scale)
			print('\n')
			pickle.dump(results, open("results/census_partition_{}_{}x{}_{}.pkl".format(method, model_seeds, partition_seeds, data_scale), "wb"))