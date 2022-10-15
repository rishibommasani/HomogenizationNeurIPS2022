import os
import pickle
from cv_groupings import group_data 
from homogenization import measure_homogenization, aggregate_measurements, compute_correlations
from tqdm import tqdm

global_measurements = {"avg" : [], "unif" : [], "worst" : [], 'error': [], "expected_avg" : [], "expected_unif" : [], "expected_worst" : [], 'expected_errors' : [], 'var_over_joint' : [], 'var_over_expected' : []}


def load_inputs():
	filename = 'predictions/celeba/attribute_values.pkl'
	_, _, inputs, attr2idx = pickle.load(open(filename, "rb"))
	return inputs, attr2idx


def load_predictions_per_epoch(folder, adaptation_seed):
	root_dir = 'predictions/celeba/{}'.format(folder)
	predictions_per_epoch = {}
	for seed_folder in os.listdir(root_dir):
		if 'seed-{}'.format(adaptation_seed) in seed_folder:
			run_folder = root_dir + '/{}/model_preds'.format(seed_folder)
			for epoch in range(0, 10):
				filename = run_folder + '/test_{}_preds.pkl'.format(epoch)
				predictions, labels = pickle.load(open(filename, "rb"))
				predictions_per_epoch[epoch] = {'predictions' : predictions, 'labels' : labels}
	return predictions_per_epoch


def predictions_per_method(dataset_names, adaptation_method, random_seed=None):
	if adaptation_method == 'probing':
		pattern = 'torch_linprobe_{}_celeba_clip_vit_b16'
	elif adaptation_method == 'finetuning':
		pattern = 'full_ft_{}_celeba_clip_vit_b16'
	elif adaptation_method == 'scratch':
		pattern = 'full_ft_{}_celeba_scratch_vit_b16_clipstyle'
	else:
		raise NotImplementedError
	
	patterns = [pattern] * len(dataset_names)
	predictions_dict = {adaptation_seed : {epoch : {} for epoch in range(0, 10)} for adaptation_seed in range(0, 5)}
	
	for dataset_name, pattern in zip(dataset_names, patterns):
		folder_name = pattern.format(dataset_name)
		for adaptation_seed in range(0, 5):
			predictions_per_epoch = load_predictions_per_epoch(folder_name, adaptation_seed)
			for epoch, predictions in predictions_per_epoch.items():
				predictions_dict[adaptation_seed][epoch][dataset_name] = predictions 
	return predictions_dict


# Formats predictions and group info into dicts expected by measure_homogenization
# Input: predictions - {dataset_name : {'predictions' : predictions_list, 'labels' : labels_list}}
# Input: groups_list - list of groups per example
# Input: dataset_names - list of dataset_names
# Returns: Appropriately formatted applications_data
def format_applications_data(predictions, groups_list, dataset_names):	
	reformatted_applications_data = {}
	groups = set()

	for dataset_name in dataset_names:
		predictions_list, labels_list = predictions[dataset_name]['predictions'], predictions[dataset_name]['labels']
		assert len(predictions_list) == len(labels_list) == len(groups_list)

		formatted_dataset = {}
		for entry_id, (prediction, label, group) in enumerate(zip(predictions_list, labels_list, groups_list)):
			groups.add(group)
			formatted_dataset[entry_id] = {'prediction' : prediction, 'label' : label, 'group' : group}
		reformatted_applications_data[dataset_name] = formatted_dataset
	return reformatted_applications_data, groups


def results_per_datasets(dataset_names, pickle_file):
	groupings = ['individual', 'hair', 'beard']
	cv_table = {}

	inputs, attr2idx = load_inputs()

	for adaptation_method in tqdm(['probing', 'finetuning', 'scratch']):
		all_applications_data = predictions_per_method(dataset_names, adaptation_method)
		for G in groupings:
			groups_list = group_data(G, inputs, attr2idx)
			for epoch in tqdm([9]):
				seed2measurements = {}
				for adaptation_seed in range(0, 5):
					reformatted_applications_data, groups = format_applications_data(all_applications_data[adaptation_seed][epoch], groups_list, dataset_names)
					homogenization_measurements = measure_homogenization(reformatted_applications_data, groups)
					seed2measurements[adaptation_seed] = homogenization_measurements
					for metric_name in global_measurements.keys():
						global_measurements[metric_name].append(homogenization_measurements[metric_name])
				aggregate_homogenization_measurements = aggregate_measurements(seed2measurements)
				cv_table[(adaptation_method, G, epoch)] = aggregate_homogenization_measurements
	pickle.dump(cv_table, open(pickle_file, 'wb'))
	return cv_table


if __name__ == '__main__':
	for apparel_names in [['Earrings', 'Necklace']]:
		pickle_file = "results/cv_experiments_{}.pkl".format("_".join(apparel_names))
		dataset_names = ['Wearing_' + apparel for apparel in apparel_names]
		results_per_datasets(dataset_names, pickle_file)			
