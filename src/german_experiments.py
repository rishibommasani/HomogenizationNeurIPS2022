import random
import numpy as np
import pandas as pd
import statistics
import sklearn
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

TRAIN_DATASET_SIZE = 800
EVAL_DATASET_SIZE = 200
EMBEDDING_SIZE = 1
NUM_FEATURE_COLS = [
	'duration_in_month', 'installment_rate', 'resident_since', 'age',
	'num_credits', 'num_liable'
]
CAT_FEATURE_COLS = [
	'status_checking_account', 'gender', 'credit_history', 'purpose', 'savings',
	'employement_since', 'debters', 'property', 'other_installments', 'housing',
	'job', 'telephone', 'foreign_worker'
]
# Used for creating one-hot encoding for categorical features:
VOCAB_SIZE_DICT = {
	'status_checking_account': 4,
	'duration_in_month': 33,
	'credit_history': 5,
	'purpose': 10,
	'credit_amount': 921,
	'savings': 5,
	'employement_since': 5,
	'installment_rate': 4,
	'gender': 2,
	'debters': 3,
	'resident_since': 4,
	'property': 4,
	'age': 53,
	'other_installments': 3,
	'housing': 3,
	'num_credits': 4,
	'job': 4,
	'num_liable': 2,
	'telephone': 2,
	'foreign_worker': 2
}

features = [
	'status_checking_account',
	'duration_in_month',
	'credit_history',
	'purpose',
	'savings',
	'employement_since',
	'installment_rate',
	'gender',
	'debters',
	'resident_since',
	'property',
	'age',
	'other_installments',
	'housing',
	'num_credits',
	'job',
	'num_liable',
	'telephone',
	'foreign_worker'
]


def load_dataset():
	data_df = pd.read_csv(open('data/german.data'), header=None, delimiter=' ')  
	data_df.rename(
		columns={
			0: 'status_checking_account',
			1: 'duration_in_month',
			2: 'credit_history',
			3: 'purpose',
			4: 'credit_amount',
			5: 'savings',
			6: 'employement_since',
			7: 'installment_rate',
			8: 'gender',
			9: 'debters',
			10: 'resident_since',
			11: 'property',
			12: 'age',
			13: 'other_installments',
			14: 'housing',
			15: 'num_credits',
			16: 'job',
			17: 'num_liable',
			18: 'telephone',
			19: 'foreign_worker',
			20: 'is_good_loan'
		},
		inplace=True)
	data_df.replace(
		{
			'gender': {
			'A91': 'male',
			'A92': 'female',
			'A93': 'male',
			'A94': 'male',
			'A95': 'female'
			}
		},
		inplace=True)
	data_df['is_good_loan'] = data_df['is_good_loan'] - 1  # Make it a binary response.
	data_df['is_high_credit'] = 1 * (data_df['credit_amount'] > 2000)

	# Normalize continuous features.
	for col in NUM_FEATURE_COLS:
		data_df[col] = (data_df[col] - data_df[col].mean()) / data_df[col].std()

	for col in CAT_FEATURE_COLS:  # convert categorical features to int
		data_df[col] = pd.Categorical(data_df[col], categories=data_df[col].unique()).codes

	return data_df


def format_data(german_data, applications_list, seed):
	df_train, df_test = sklearn.model_selection.train_test_split(german_data, test_size=0.2)

	applications_data = {}
	for application_name in applications_list:
		X_train, X_test = df_train[features].to_numpy(), df_test[features].to_numpy()
		y_train, y_test = df_train[application_name].to_numpy(), df_test[application_name].to_numpy()
		applications_data[application_name] = {'X_tr' : X_train, 'X_test' : X_test, 'y_tr': y_train, 'y_test' : y_test}

	return applications_data


def fixed_partition(applications_data, applications_list, seed, data_scale):
	rng = random.Random(seed)
	index = rng.randint(0, data_scale - 1)
	
	for name, entries in applications_data.items():
		X_train, y_train = entries['X_tr'], entries['y_tr']
		y_train = np.expand_dims(y_train, axis = 1)
		data = np.concatenate((X_train, y_train), axis = 1)
		np.random.RandomState(seed=seed).shuffle(data)
		X_train, y_train = data[:, : -1], data[:, -1]

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
		y_train = np.expand_dims(y_train, axis = 1)
		data = np.concatenate((X_train, y_train), axis = 1)
		np.random.RandomState(seed=seed).shuffle(data)
		X_train, y_train = data[:, : -1], data[:, -1]

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
def partition_experiment(german_data, applications_list, data_scale, model_seeds, partition_seeds, method = 'logistic', groupings = ['individual']):
	print('Running partition german experiment for data scale {}'.format(data_scale))

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
			applications_data = format_data(german_data, applications_list, data_seed)
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
			applications_data = format_data(german_data, applications_list, data_seed)
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
	# Fetch German Contracts data
	print('Loading data')
	german_data = load_dataset()

	applications_list = ['is_good_loan', 'is_high_credit']
	
	model_seeds, partition_seeds = 25, 25

	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--methods', help='List of methods/learning algorithms', type=str)
	args = parser.parse_args()
	methods = args.methods
	if not methods:
		methods = ['logistic', 'svm', 'gbm', 'nn']
	else:
		methods = methods.split(',')

	print('Running german contract experiments')
	for i, method in enumerate(methods): 
		print(method)
		for data_scale in tqdm([20, 16, 10, 9, 8, 7, 6, 5, 4, 3, 2]):
			results = partition_experiment(german_data, applications_list, data_scale, model_seeds, partition_seeds, method = method)
			print('\n')
			print(data_scale)
			print('\n')
			pickle.dump(results, open("results/german_new_sample_partition_{}_{}x{}_{}.pkl".format(method, model_seeds, partition_seeds, data_scale), "wb"))