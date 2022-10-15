from datasets import load_from_disk
from word_list import all_names, male_words, female_words
from nlp_groupings import group_data 
from homogenization import measure_homogenization, aggregate_measurements
from tqdm import tqdm
import pickle

global_measurements = {"avg" : [], "unif" : [], "worst" : [], 'error': [], "expected_avg" : [], "expected_unif" : [], "expected_worst" : [], 'expected_errors' : [], 'var_over_joint' : [], 'var_over_expected' : []}


def load_predictions(dataset_name, adaptation_method, adaptation_seed):
	return load_from_disk('predictions/{}/{}/{}'.format(dataset_name, adaptation_method, adaptation_seed))


def predictions_per_method(dataset_names, adaptation_entry, random_seed=None):
	if adaptation_entry in {'probing', 'bitfit', 'finetuning'}:
		adaptation_methods = [adaptation_entry] * len(dataset_names)
	elif adaptation_entry == 'random':
		raise NotImplementedError
	else:
		raise NotImplementedError

	predictions_dict = {adaptation_seed : {} for adaptation_seed in range(1, 5)}
	for dataset_name, adaptation_method in zip(dataset_names, adaptation_methods):
		for adaptation_seed in range(1, 5):
			predictions_dict[adaptation_seed][dataset_name] =  load_predictions(dataset_name, adaptation_method, adaptation_seed)
	return predictions_dict


# Formats applications_data into dicts expected by measure_homogenization
# Input: applications_data
# Returns: Appropriately formatted applications_data
def format_applications_data(applications_data):	
	reformatted_applications_data = {}
	groups = set()
	for dataset_name, application_data in applications_data.items():
		formatted_dataset = {}
		dataset, groups_list = application_data['dataset'], application_data['groups_list']
		for row_id, (row, group) in enumerate(zip(dataset, groups_list)):
			prediction, label = row['predictions'], row['label']
			groups.add(group)
			formatted_dataset[row_id] = {'prediction' : prediction, 'label' : label, 'group' : group}
		reformatted_applications_data[dataset_name] = formatted_dataset
	return reformatted_applications_data, groups


def homogenization_results(nlp_table, dataset_names, adaptation_methods, groupings, additional_args, groupings_names):
	assert len(groupings) == len(groupings_names)

	for adaptation_method in adaptation_methods:
		predictions_dict = predictions_per_method(dataset_names, adaptation_method)
		for G, G_name in zip(groupings, groupings_names):
			seed2measurements = {}
			for adaptation_seed in range(1, 5):
				applications_data = {}
				for dataset_name in tqdm(dataset_names):
					predictions = predictions_dict[adaptation_seed][dataset_name]
					groups_list = group_data(G, predictions, additional_args = additional_args)
					applications_data[dataset_name] = {'dataset' : predictions, 'groups_list' : groups_list}
				reformatted_applications_data, groups = format_applications_data(applications_data = applications_data)
				homogenization_measurements = measure_homogenization(reformatted_applications_data, groups)
				for metric_name in global_measurements.keys():
						global_measurements[metric_name].append(homogenization_measurements[metric_name])
				seed2measurements[adaptation_seed] = homogenization_measurements
			aggregate_homogenization_measurements = aggregate_measurements(seed2measurements)
			nlp_table[(adaptation_method, G_name)] = aggregate_homogenization_measurements

	return nlp_table


if __name__ == '__main__':
	nlp_table = {}
	dataset_names = ['agnews', 'imdb', 'hatespeech', 'yahoo']
	adaptation_methods = ['probing', 'bitfit', 'finetuning']

	groupings = ['race', 'name', 'gender']
	additional_args = {'names' : all_names, 'male' : male_words, 'female' : female_words, 'hispanic' : hispanic_names, 'white' : white_names, 'black' : black_names, 'api' : api_names}
	nlp_table = homogenization_results(nlp_table, dataset_names, adaptation_methods, groupings, additional_args, groupings)

	groupings = ['length']
	for num_bins in tqdm({2, 3, 4, 5, 10, 20, 50, 100}):
		additional_args['length_bins'] = num_bins
		groupings_names = ['lengths-{}'.format(num_bins)]
		nlp_table = homogenization_results(nlp_table, dataset_names, adaptation_methods, groupings, additional_args, groupings_names)

	pickle.dump(nlp_table, open("results/nlp_experiments_race.pkl", "wb"))

	print('\n \n \n')
	print('Global NLP correlations between metrics')
	print([len(measurements) for measurements in global_measurements.values()])
	print('\n')
	global_correlations = {}
	for row_metric_name, row_measurements in global_measurements.items():
		for column_metric_name, column_measurements in global_measurements.items():
			assert len(row_measurements) == len(column_measurements)
			global_correlations[(row_metric_name, column_metric_name)] = compute_correlations(row_measurements, column_measurements)
			print(row_metric_name, column_metric_name)
			print(global_correlations[(row_metric_name, column_metric_name)])

	pickle.dump(global_correlations, open("results/nlp_correlations_race.pkl", "wb"))
