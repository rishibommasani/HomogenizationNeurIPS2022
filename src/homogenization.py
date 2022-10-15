import math 
import statistics
from sklearn.metrics import r2_score
from scipy.stats import linregress, spearmanr


def mean(data):
	return sum(data) / len(data)


def compute_correlations(list1, list2):
	assert len(list1) == len(list2)
	slope, intercept, r_value, linear_p_value, std_err = linregress(list1, list2)
	rho, spearman_p_value = spearmanr(list1, list2)
	return_value = {'R^2' : r_value ** 2, 'linear_p': linear_p_value, 'linear_std_err': std_err, 'rho': rho, 'spearman_p': spearman_p_value}
	return_value = {k : round(v, 4) for k, v in return_value.items()}
	return return_value


# Compute the statistics for a specific deployment i
# Input: [data] - {entry_id : entry}
# entry = {'prediction' : pred, 'label' : label, 'group' : group}
# Input: [groups] - {groups}
# Output: N - total number of examples for deployment i
# Output: error_rate - err for h^i
# Output: group_statistics - {group : group_statistic}
# group_statistic = {'count' : count, 'errors' : errors, 'error_rate' : error_rate}
def compute_statistics(data, groups):
	N = len(data)
	group_statistics = {group : {'group_count' : 0, 'group_errors' : 0} for group in groups}
	total_errors = 0
	for entry_id, entry in data.items():
		prediction, label, group = entry['prediction'], entry['label'], entry['group']
		assert group in groups, "For entry ID: {}, the group {} is not in the provided list of groups {}".format(entry_id, group, groups)
		
		error = prediction != label
		 
		group_statistics[group]['group_count'] += 1 
		group_statistics[group]['group_errors'] += error 
		total_errors += error

	error_rate = total_errors / N 
	for group, group_statistic in group_statistics.items():
		group_count, group_errors = group_statistic['group_count'], group_statistic['group_errors']
		group_statistic['group_frequency'] = group_count / N 
		if group_count == 0:
			group_statistic['group_error_rate'] = 1
		else:
			group_statistic['group_error_rate'] = group_errors / group_count
	return {'N' : N, 'error_rate' : error_rate, 'group_statistics' : group_statistics}


# Output: \prod_{i} fail(h^i)
def compute_global_error_rate(application_statistics):
	return math.prod([application_statistic['error_rate'] for application_statistic in application_statistics.values()])


# def compute_best_error_rate(application_statistics):
# 	return min([application_statistic['error_rate'] for application_statistic in application_statistics.values()])	


# def compute_average_error_rate(application_statistics):
# 	return mean([application_statistic['error_rate'] for application_statistic in application_statistics.values()])	


# Dictionary mapping groups to frequency-weighted systemic failure rate.
# For a group g, this is \prod_i freq(g, i) * fail_g(h^i)
# Output: {group : frequency-weighted systemic failure rate for group}
def compute_group2systemic_failure(application_statistics, groups):
	H = {}
	for g in groups:
		H_g = 1
		for _, application_statistic in application_statistics.items():
			g_statistic = application_statistic['group_statistics'][g]
			g_error_rate, g_frequency = g_statistic['group_error_rate'], g_statistic['group_frequency'] 
			g_joint_probability = g_frequency * g_error_rate
			H_g *= g_joint_probability
		H[g] = H_g
	return H


# Dictionary mapping groups to unweighted systemic failure rate.
# For a group g, this is \prod_i fail_g(h^i)
# Output: {group : systemic failure rate for group}
def compute_global_group_error_rates(application_statistics, groups):
	global_group_error_rates = {}
	for g in groups:
		global_g_error_rate = math.prod([stat['group_statistics'][g]['group_error_rate'] for stat in application_statistics.values()])
		global_group_error_rates[g] = global_g_error_rate
	return global_group_error_rates


# Dictionary mapping groups to joint group frequency.
# For a group g, this is \prod_i freq(g, i)
# Output: {group : joint frequency for group}
def compute_global_group_frequencies(application_statistics, groups):
	global_group_frequencies = {}
	for g in groups:
		global_g_frequency = math.prod([stat['group_statistics'][g]['group_frequency'] for stat in application_statistics.values()])
		global_group_frequencies[g] = global_g_frequency
	return global_group_frequencies


def compute_average_case_homogenization(application_statistics, groups):
	H = compute_group2systemic_failure(application_statistics, groups)
	# \sum_g \prod_i freq(g, i) * fail_g(h^i)
	numerator = sum([H[g] for g in H])
	# \prod_i fail(h^i)
	global_error_rate = compute_global_error_rate(application_statistics)
	global_group_frequencies = compute_global_group_frequencies(application_statistics, groups)
	# \sum_g \prod_i freq(g, i) * fail(h^i)
	denominator = global_error_rate * sum([global_group_frequencies[g] for g in global_group_frequencies])
	assert denominator != 0, "Average case homogenization is not defined; see the global error rate: {} and global group frequencies: {}".format(global_error_rate, global_group_frequencies)
	return numerator / denominator 


def compute_uniform_case_homogenization(application_statistics, groups):
	global_group_error_rates = compute_global_group_error_rates(application_statistics, groups)
	# \E_g \prod_i fail_g(h^i)
	numerator = mean([global_group_error_rates[g] for g in global_group_error_rates])
	global_error_rate = compute_global_error_rate(application_statistics)
	# \prod_i fail_g(h^i)
	denominator = global_error_rate 
	assert denominator != 0, "Uniform case homogenization is not defined; see the global error rate: {} and global group error rates: {}".format(global_error_rate, global_group_error_rates)
	return numerator / denominator 


def compute_worst_case_homogenization(application_statistics, groups):
	global_group_error_rates = compute_global_group_error_rates(application_statistics, groups)
	# \max_g \prod_i fail_g(h^i)
	worst_group_error_rate = max([global_group_error_rates[g] for g in global_group_error_rates])
	numerator = worst_group_error_rate
	# \prod_i fail_g(h^i)
	global_error_rate = compute_global_error_rate(application_statistics)
	denominator = global_error_rate
	assert denominator != 0, "Worst case homogenization is not defined; see the global error rate: {} and worst group error rates: {}".format(global_error_rate, worst_group_error_rate)
	return numerator / denominator 


def compute_variance_over_joint_errors(application_statistics, groups):
	global_group_error_rates = compute_global_group_error_rates(application_statistics, groups)
	return statistics.variance([global_group_error_rates[g] for g in global_group_error_rates])


def aggregate_measurements(seed2measurements):
	per_seed_measurements = {}
	for measurements in seed2measurements.values():
		for measurement_name, measurement in measurements.items():
			if measurement_name in per_seed_measurements:
				per_seed_measurements[measurement_name].append(measurement)
			else:
				per_seed_measurements[measurement_name] = [measurement]
	aggregates = {}
	for measurement_name, measurements in per_seed_measurements.items():
		mean = statistics.mean(measurements)
		stdev = statistics.stdev(measurements)
		seeds = len(measurements)
		aggregates[measurement_name] = {'mean' : mean, 'stdev': stdev, 'seeds' : seeds}
		# aggregates[measurement_name] = {'mean' : round(mean, 3), 'stdev': round(stdev, 3), 'seeds' : seeds}
	return aggregates


def measure_homogenization(applications_data, groups, verbose = False):
	"""Measures homogenization given data about applications and groups
	This function assumes both input groupings and model predictions are already computed in applications

	Args:
	applications: Dictionary from application name (str) to application data (dict)
	application data: dict indexed by entry id with each entry itself a datum (dict)
	entry: dict containing prediction, label, and group for each example
	groups: valid groups under grouping function being used
	
	Returns:
	Dict containing homogenization values for "avg", "unif", and "worst" definitions
	Dict also contains global error rate as "error" and application-specific error rates by application name
	Verbose flag = true also returns additional statistics 
	
    """
	application_statistics = {name : compute_statistics(data, groups) for name, data in applications_data.items()}
	
	global_error_rate = compute_global_error_rate(application_statistics)
	average_case_homogenization = compute_average_case_homogenization(application_statistics, groups)
	uniform_case_homogenization = compute_uniform_case_homogenization(application_statistics, groups)
	worst_case_homogenization = compute_worst_case_homogenization(application_statistics, groups)

	return_value = {"avg" : average_case_homogenization, "unif" : uniform_case_homogenization, "worst" : worst_case_homogenization, 'error' : global_error_rate}
	return_value['var_over_joint'] = compute_variance_over_joint_errors(application_statistics, groups)

	for name, statistics in application_statistics.items():
		return_value[name] = statistics['error_rate']

	if verbose:
		return return_value, application_statistics
	else:
		return return_value
