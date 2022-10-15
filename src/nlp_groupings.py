import numpy as np
import nltk
from nltk.tokenize import word_tokenize 


def get_tokenized_text(row):
	return word_tokenize(row['text'])


def get_top_topic(text, model):
	raise NotImplementedError


def group_by_topic(inputs, additional_args):
	groups_list = []
	topic_model = additional_args['topic_model']

	for row in inputs:
		tokenized_text = get_tokenized_text(row)
		top_topic = get_top_topic(tokenized_text, topic_model)
		groups_list.append(top_topic)
	return groups_list


def group_by_name(inputs, additional_args):
	groups_list = []
	names = additional_args['names']
	names = {name.lower() for name in names}

	for row in inputs:
		tokenized_text = get_tokenized_text(row)
		tokens = {token.lower() for token in tokenized_text}
		seen_names = names & tokens
		if len(seen_names) == 1:
			(group,) = seen_names
		else:
			# Indicates either zero or > 1 name appears in input
			group = 'Other'
		groups_list.append(group)					
	return groups_list


def group_by_race(inputs, additional_args):
	groups_list = []
	races = ['hispanic', 'white', 'black', 'api']
	race2index = {race : i for i, race in enumerate(races)}

	for row in inputs:
		tokenized_text = get_tokenized_text(row)
		tokens = {token.lower() for token in tokenized_text}
		race_counts = [0] * len(races)

		for token in tokens:
			for race in races:
				if token in additional_args[race]:
					race_counts[race2index[race]] += 1

		if all(count == race_counts[0] for count in race_counts):
			group == 'Other'
		else:
			group = races[np.argmax(race_counts)]
		groups_list.append(group)					
	return groups_list


def group_by_gender(inputs, additional_args):
	groups_list = []
	male_words, female_words = additional_args['male'], additional_args['female']
	male_words, female_words = {word.lower() for word in male_words}, {word.lower() for word in female_words}
	assert len(male_words & female_words) == 0

	for row in inputs:
		tokenized_text = get_tokenized_text(row)
		tokens = {token.lower() for token in tokenized_text}
		male_count, female_count = 0, 0 

		for token in tokens:
			if token in male_words:
				male_count += 1
			elif token in female_words:
				female_count += 1
		
		if male_count > female_count:
			group = 'male'
		elif female_count > male_count:
			group = 'female'
		else:
			group = 'Other'
		
		groups_list.append(group)					
	return groups_list


def group_by_length(inputs, additional_args):
	groups_list = []
	lengths = []
	num_bins = additional_args['length_bins']

	for row in inputs:
		tokenized_text = get_tokenized_text(row)
		lengths.append(len(tokenized_text))

	ordered_lengths = sorted(lengths)
	N = len(ordered_lengths)
	for length in lengths:
		group = 1
		while(group < num_bins and length > ordered_lengths[group * (N // num_bins)]):
			group += 1
		assert group in range(1, num_bins + 1)
		groups_list.append(group)
	return groups_list


# Return group specified by grouping function G for each input
# Input: G - Name of grouping function
# Input: inputs - HF dataset 
# Input: additional_args - additional arguments needed by grouping function
def group_data(G, inputs, additional_args):
	name2function = {'topic' : group_by_topic, 'name' : group_by_name, 'gender' : group_by_gender, 'length' : group_by_length, 'race' : group_by_race}
	group_function = name2function[G]
	groups_list = group_function(inputs, additional_args)
	return groups_list
