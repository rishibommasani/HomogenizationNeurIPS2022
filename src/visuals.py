import pickle 
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma']
plt.rcParams.update({'font.size': 11})
from os.path import exists


def generate_census_partition_subfigure(fig, table, metrics, grouping, data_scales_list, abbreviated_metrics, title_pattern):
	x = [3600000 // scale for scale in data_scales_list]
	width = 0.35 / (2 * len(metrics) - 1)  
	x_pos = [i + 0.175 for i in range(len(x))]
	
	bar_pos = 0
	for metric, metric_name, linestyle in zip(metrics, abbreviated_metrics, ['solid', 'dotted', 'dashed']):
		partition = 'disjoint'
		measurements = [table[scale][(partition, grouping)][metric] for scale in data_scales_list]
		means = [measurement['mean'] for measurement in measurements]
		stdevs = [measurement['stdev'] for measurement in measurements]
		x_pos = [i + width * bar_pos for i in range(len(x))]
		if grouping == 'individual':
			label = '{}'.format(partition)
		else:
			label = '{}, {}'.format(metric_name, partition)
		fig.errorbar(x, means,  color='red', linestyle = linestyle, marker = 'o', label = label)
		bar_pos += 1

		partition = 'fixed'
		measurements = [table[scale][(partition, grouping)][metric] for scale in data_scales_list]
		means = [measurement['mean'] for measurement in measurements]
		stdevs = [measurement['stdev'] for measurement in measurements]
		x_pos = x_pos = [i + width * bar_pos for i in range(len(x))]
		if grouping == 'individual':
			label = '{}'.format(partition)
		else:
			label = '{}, {}'.format(metric_name, partition)
		fig.errorbar(x, means, color='blue', linestyle = linestyle, marker = 'o', label = label)
		bar_pos += 1
	fig.legend(loc='best')
	return fig


def generate_census_partition_figure(filename, metrics, abbreviated_metrics, title_pattern, figure_tag=''):
	table = pickle.load(open(filename, "rb"))
	data_scales_list = sorted([25000, 10000, 7500, 5000, 2500, 1000, 750])
	
	f, axs = plt.subplots(1, 2, figsize=(14.31, 4.41))
	ax1, ax2 = axs
	generate_census_partition_subfigure(ax2, table, metrics, 'race', data_scales_list, abbreviated_metrics, title_pattern)
	metrics, abbreviated_metrics = metrics[:1], abbreviated_metrics[:1]
	generate_census_partition_subfigure(ax1, table, metrics, 'individual', data_scales_list, abbreviated_metrics, title_pattern)

	ax1.set_title('Outcome Homogenization for Individuals in ACS PUMS ({})'.format(figure_tag))
	ax2.set_title('Outcome Homogenization for Racial Groups in ACS PUMS ({})'.format(figure_tag))

	xlabel = 'Number of examples in training data'
	ylabel = 'Homogenization'
	for ax in axs.flat:
		ax.set(xlabel=xlabel)
	ax1.set_ylabel(ylabel)

	f.savefig('figures/census/census_partition_neurips_{}'.format(figure_tag.lower()), dpi = 100)


def generate_lawschool_partition_subfigure(fig, table, metrics, grouping, data_scales_list, abbreviated_metrics, title_pattern):
	x = [4500 // scale for scale in data_scales_list]
	width = 0.35 / (2 * len(metrics) - 1)  
	x_pos = [i + 0.175 for i in range(len(x))]
	
	bar_pos = 0
	for metric, metric_name, linestyle in zip(metrics, abbreviated_metrics, ['solid', 'dotted', 'dashed']):
		partition = 'disjoint'
		measurements = [table[scale][(partition, grouping)][metric] for scale in data_scales_list]
		means = [measurement['mean'] for measurement in measurements]
		stdevs = [measurement['stdev'] for measurement in measurements]
		x_pos = [i + width * bar_pos for i in range(len(x))]
		if grouping == 'individual':
			label = '{}'.format(partition)
		fig.errorbar(x, means,  color='red', linestyle = linestyle, marker = 'o', label = label)
		bar_pos += 1

		partition = 'fixed'
		measurements = [table[scale][(partition, grouping)][metric] for scale in data_scales_list]
		means = [measurement['mean'] for measurement in measurements]
		stdevs = [measurement['stdev'] for measurement in measurements]
		x_pos = x_pos = [i + width * bar_pos for i in range(len(x))]
		if grouping == 'individual':
			label = '{}'.format(partition)
		fig.errorbar(x, means, color='blue', linestyle = linestyle, marker = 'o', label = label)
		bar_pos += 1
	fig.legend(loc='best')
	return fig


def generate_lawschool_partition_figure(filename, metrics, abbreviated_metrics, title_pattern, figure_tag=''):
	table = pickle.load(open(filename, "rb"))
	# data_scales_list = sorted(list(set(range(45, 30, -5)) | set(range(30, 5, -1))))
	data_scales_list = sorted(list(set(range(30, 5, -1))))

	f, axs = plt.subplots(1, 2, figsize=(14.31, 4.41))
	ax1, ax2 = axs
	homogenization_metrics, homogenization_abbreviated_metrics = metrics[:1], abbreviated_metrics[:1]
	generate_lawschool_partition_subfigure(ax1, table, homogenization_metrics, 'individual', data_scales_list, homogenization_abbreviated_metrics, title_pattern)
	error_metrics, error_abbreviated_metrics = metrics[-1:], abbreviated_metrics[-1:]
	generate_lawschool_partition_subfigure(ax2, table, error_metrics, 'individual', data_scales_list, error_abbreviated_metrics, title_pattern)

	ax1.set_title('Outcome Homogenization for Individuals in LSAC ({})'.format(figure_tag))
	ax2.set_title('Expected Systemic Failure Rate for Individuals in LSAC ({})'.format(figure_tag))

	xlabel = 'Number of examples in training data'
	homogenization_ylabel, error_ylabel = 'Homogenization',  'Systemic Failure'
	for ax in axs.flat:
		ax.set(xlabel=xlabel)
	ax1.set_ylabel(homogenization_ylabel)
	ax2.set_ylabel(error_ylabel)

	f.savefig('figures/lsac/lsac_partition_neurips_{}'.format(figure_tag.lower()), dpi = 100)


def generate_german_partition_subfigure(fig, table, metrics, grouping, data_scales_list, abbreviated_metrics, title_pattern):
	x = [800 // scale for scale in data_scales_list]
	width = 0.35 / (2 * len(metrics) - 1)  
	x_pos = [i + 0.175 for i in range(len(x))]
	
	bar_pos = 0
	for metric, metric_name, linestyle in zip(metrics, abbreviated_metrics, ['solid', 'dotted', 'dashed']):
		partition = 'disjoint'
		measurements = [table[scale][(partition, grouping)][metric] for scale in data_scales_list]
		means = [measurement['mean'] for measurement in measurements]
		stdevs = [measurement['stdev'] for measurement in measurements]
		x_pos = [i + width * bar_pos for i in range(len(x))]
		if grouping == 'individual':
			label = '{}'.format(partition)
		fig.errorbar(x, means,  color='red', linestyle = linestyle, marker = '.', label = label)
		bar_pos += 1

		partition = 'fixed'
		measurements = [table[scale][(partition, grouping)][metric] for scale in data_scales_list]
		means = [measurement['mean'] for measurement in measurements]
		stdevs = [measurement['stdev'] for measurement in measurements]
		x_pos = x_pos = [i + width * bar_pos for i in range(len(x))]
		if grouping == 'individual':
			label = '{}'.format(partition)
		fig.errorbar(x, means, color='blue', linestyle = linestyle, marker = '.', label = label)
		bar_pos += 1
	fig.legend(loc='best')
	return fig


def generate_german_partition_figure(filename, metrics, abbreviated_metrics, title_pattern, figure_tag=''):
	table = pickle.load(open(filename, "rb"))
	data_scales_list = [20, 16, 10, 9, 8, 7, 6, 5, 4, 3, 2]
	
	f, axs = plt.subplots(1, 2, figsize=(14.31, 4.41))
	ax1, ax2 = axs
	homogenization_metrics, homogenization_abbreviated_metrics = metrics[:1], abbreviated_metrics[:1]
	generate_german_partition_subfigure(ax1, table, homogenization_metrics, 'individual', data_scales_list, homogenization_abbreviated_metrics, title_pattern)
	error_metrics, error_abbreviated_metrics = metrics[-1:], abbreviated_metrics[-1:]
	generate_german_partition_subfigure(ax2, table, error_metrics, 'individual', data_scales_list, error_abbreviated_metrics, title_pattern)

	ax1.set_title('Outcome Homogenization for Individuals in GC ({})'.format(figure_tag))
	ax2.set_title('Expected Systemic Failure Rate for Individuals in GC ({})'.format(figure_tag))

	xlabel = 'Number of examples in training data'
	homogenization_ylabel, error_ylabel = 'Homogenization',  'Systemic Failure'
	for ax in axs.flat:
		ax.set(xlabel=xlabel)
	ax1.set_ylabel(homogenization_ylabel)
	ax2.set_ylabel(error_ylabel)

	f.savefig('figures/german/german_partition_neurips_{}'.format(figure_tag.lower()), dpi = 100)


def generate_cv_experiments_epochs_subfigure(table, metrics, num_epochs, method, grouping):
	x_axis = 'Epochs'
	y_axis = 'Measurement'
	title = 'Homogenization across training for {} grouped by {}'.format(method, grouping)

	plt.figure()
	epochs = list(range(num_epochs))

	all_measurements = [table[(method, grouping, epoch)] for epoch in epochs]
	for metric in metrics:
		measurements = [all_measurement[metric] for all_measurement in all_measurements]
		means = [measurement['mean'] for measurement in measurements]
		stdevs = [measurement['stdev'] for measurement in measurements]
		label = metric
		plt.errorbar(epochs, means, stdevs, label = label)

	plt.xlabel(x_axis)
	plt.ylabel(y_axis)
	plt.title(title)
	plt.legend(loc='best')
	plt.show()


def generate_cv_experiments_epochs_figure(filename, metrics, num_epochs):
	table = pickle.load(open(filename, "rb"))
	for method in ['scratch', 'probing', 'finetuning']:
		for grouping in ['individual', 'hair', 'beard']:
			generate_cv_experiments_epochs_subfigure(table = table, metrics = metrics, num_epochs = num_epochs, method = method, grouping = grouping)


def generate_cv_experiments_subfigure(fig, table, metrics, epoch, methods, grouping, abbreviated_metrics, title_pattern):
	x_axis = 'Group Homogenization Metrics'
	y_axis = 'Measurement'
	title = '{} homogenization for vision models grouped by {}'.format(title_pattern, grouping)

	width = 0.8 / (len(methods)) 
	x_pos = [i + 0.275 for i in range(len(metrics))]
	if grouping == 'individual':
		fig.set_xticks([], [])
	else:
		fig.set_xticks(x_pos)
		fig.set_xticklabels(abbreviated_metrics)
		fig.set_xlabel(x_axis)

	bar_pos = 0
	for method, color in zip(methods, ['red', 'blue', 'green']):
		measurements = [table[(method, grouping, epoch)][metric] for metric in metrics]
		means = [measurement['mean'] for measurement in measurements]
		stdevs = [measurement['stdev'] for measurement in measurements]
		label = method  
		x_pos = [i + width * bar_pos for i in range(len(metrics))]
		fig.bar(x_pos, means,  width, color=color, yerr=stdevs, label = label)
		bar_pos += 1

	if grouping == 'individual':
		fig.legend(loc='lower center')
	else:
		fig.legend(loc='best')


def generate_cv_experiments_figure(filename, full_metrics, methods, full_abbreviated_metrics, title_pattern):
	table = pickle.load(open(filename, "rb"))
	f, axs = plt.subplots(1, 3, figsize=(14.31, 4.41))
	titles = ['for Individuals', 'by Hair Color', 'by Beard']
	for ax, grouping, title in zip(axs, ['individual', 'hair', 'beard'], titles):
		if grouping == 'individual':
			metrics, abbreviated_metrics = full_metrics[:1], full_abbreviated_metrics[:1]
			ylabel = 'Homogenization'
			ax.set_ylabel(ylabel)
		else:
			metrics, abbreviated_metrics = full_metrics, full_abbreviated_metrics
		ax.set_title('Outcome Homogenization {}'.format(title))
		generate_cv_experiments_subfigure(fig = ax, table = table, metrics = metrics, epoch = 9, methods = methods, grouping = grouping, abbreviated_metrics = abbreviated_metrics, title_pattern = title_pattern)
	plt.show()
	f.savefig('figures/cv/cv_experiment')
	

def generate_nlp_experiments_subfigure(table, metrics, methods, grouping, abbreviated_metrics, title_pattern):
	x_axis = 'Group Homogenization Metrics'
	y_axis = 'Homogenization'
	title = 'Outcome Homogenization for {} Groups'.format(grouping.title())

	fig = plt.figure()
	plt.rcParams["figure.figsize"] = (4.41, 4.41)
	width = 0.8 / (len(methods)) 
	x_pos = [i + 0.275 for i in range(len(metrics))]
	plt.xticks(x_pos, abbreviated_metrics)

	bar_pos = 0
	for method, color in zip(methods, ['purple', 'blue', 'green']):
		measurements = [table[(method, grouping)][metric] for metric in metrics]
		means = [measurement['mean'] for measurement in measurements]
		stdevs = [measurement['stdev'] for measurement in measurements]
		label = method  
		x_pos = [i + width * bar_pos for i in range(len(metrics))]
		plt.bar(x_pos, means,  width, color=color, yerr=stdevs, label = label)
		bar_pos += 1

	plt.xlabel(x_axis)
	plt.ylabel(y_axis)
	plt.title(title)
	plt.legend(loc='best')
	plt.show()
	fig.savefig('figures/nlp/nlp_experiment_{}'.format(grouping))


def generate_nlp_experiments_figure(filename, metrics, methods, abbreviated_metrics, title_pattern):
	table = pickle.load(open(filename, "rb"))
	grouping = 'gender'
	generate_nlp_experiments_subfigure(table = table, metrics = metrics, methods = methods, grouping = grouping, abbreviated_metrics = abbreviated_metrics, title_pattern = title_pattern)


def generate_nlp_experiments_table(filename):
	table = pickle.load(open(filename, "rb"))
	for k, v in table.items():
		print(k)
		print(v)
		print('\n')


def generate_correlations_table(filename, row_names, column_names):
	table = pickle.load(open(filename, "rb"))
	print(' & '.join([''] + list(column_names)) + '\\\\ ' + '\n')
	for row_name in  row_names:
		string = ''
		for column_name in column_names:
			correlations = table[(row_name, column_name)]
			r2, lin_p, rho, mon_p  = correlations['R^2'], correlations['linear_p'], correlations['rho'], correlations['spearman_p']
			r2, rho = str(round(r2, 2)), str(round(rho, 2))
			if lin_p < 0.001:
				r2 = r2 + '**'
			elif lin_p < 0.05:
				r2 =  r2 + '*'
			if mon_p < 0.001:
				rho = rho + '**'
			elif mon_p < 0.05:
				rho = rho + '*'
			string += '({}, {})'.format(r2, rho)
			string += ' & '
		string += '\\\\ \n'
		print(string)


if __name__ == '__main__':
	visualize = {'census'}

	tracked = {"avg" : [], "unif" : [], "worst" : [], 'error': [], 'var_over_joint' : []}
	metrics = list(tracked.keys())
	abbreviated_metrics = ['avg', 'unif', 'worst', 'err', 'V_err']
	row_names, column_names = metrics, metrics
	
	# Census
	if 'census' in visualize:
		for method in ['GBM', 'SVM', 'NN', 'Logistic']:	
			model_seeds, partition_seeds = 5, 5			
			filename = 'results/census_new_sample_partition_neurips_{}x{}_{}.pkl'.format(model_seeds, partition_seeds, method.lower())

			if exists(filename):
				metrics, abbreviated_metrics = ['avg', 'unif', 'worst'], ['avg', 'unif', 'worst']
				title_pattern = 'Systemic'
				generate_census_partition_figure(filename, metrics, abbreviated_metrics, title_pattern, figure_tag = method) 
			else:
				print('No file:', filename)

	# LSAC
	if 'lsac' in visualize:
		for method in ['SVM', 'Logistic', 'GBM', 'NN']:	
			model_seeds, partition_seeds = 5, 5	
			filename = 'results/lawschool_new_sample_partition_neurips_{}x{}_{}.pkl'.format(model_seeds, partition_seeds, method.lower())
			if exists(filename):
				metrics, abbreviated_metrics = ['avg', 'unif', 'worst', 'error'], ['avg', 'unif', 'worst', 'error']
				title_pattern = 'Systemic'
				generate_lawschool_partition_figure(filename, metrics, abbreviated_metrics, title_pattern, figure_tag = method) 
			else:
				print('No file: ', filename)

	# German contracts
	if 'german' in visualize:
		for method in ['SVM', 'Logistic', 'GBM', 'NN']:			
			model_seeds, partition_seeds = 25, 25	
			filename = 'results/german_new_sample_partition_neurips_{}x{}_{}.pkl'.format(model_seeds, partition_seeds, method.lower())
			if exists(filename):
				metrics, abbreviated_metrics = ['avg', 'unif', 'worst', 'error'], ['avg', 'unif', 'worst', 'error']
				title_pattern = 'Systemic'
				generate_german_partition_figure(filename, metrics, abbreviated_metrics, title_pattern, figure_tag = method) 
			else:
				print('No file: ', filename)

	# CV
	if 'cv' in visualize:
		filename = 'results/cv_experiments_Earrings_Necklace.pkl'
		methods = ['scratch', 'probing', 'finetuning']

		metrics, abbreviated_metrics = ['avg', 'unif', 'worst'], ['avg', 'unif', 'worst']
		title_pattern = 'Systemic'
		generate_cv_experiments_figure(filename, metrics, methods, abbreviated_metrics, title_pattern)

	# NLP
	if 'nlp' in visualize:
		filename = 'results/nlp_experiments.pkl'
		methods = ['bitfit', 'probing','finetuning']

		metrics, abbreviated_metrics = ['avg', 'unif', 'worst'], ['avg', 'unif', 'worst']
		title_pattern = 'Systemic'
		generate_nlp_experiments_figure(filename, metrics, methods, abbreviated_metrics, title_pattern)

	
	# Correlations
	if 'correlations' in visualize:
		row_names = ['avg', 'unif', 'worst']
		column_names = ["avg", "unif", "worst", 'error', 'var_over_joint']
		generate_correlations_table('results/cv_correlations.pkl', row_names, column_names)
		generate_correlations_table('results/nlp_correlations.pkl', row_names, column_names)
