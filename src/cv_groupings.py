def group_by_individual(row, i, attr2idx):
	return i


def group_by_hair(row, i, attr2idx):
	hair_colors = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
	hair_color_idxs = [attr2idx[attr] for attr in hair_colors]
	hair_color_vals = [row[idx] for idx in hair_color_idxs]
	
	if sum(hair_color_vals) != 1:
		# Individual is bald, has hair obfuscated, or has > 1 hair color annotated
		return 'Other'
	else:
		for hair_color, hair_color_val in zip(hair_colors, hair_color_vals):
			if hair_color_val == 1:
				return hair_color


def group_by_beard(row, i, attr2idx):
	no_beard_idx = attr2idx['No_Beard']
	val = row[no_beard_idx]
	val2str = {0 : 'Beard', 1: 'No_Beard'}
	return val2str[val]


# Return group specified by grouping function G for each input
# Input: G - Name of grouping function
# Input: inputs - matrix of (num. examples, num. attr)
# attr2idx: Mapping from attribute names to attribute column index
def group_data(G, inputs, attr2idx):
	name2function = {'individual' : group_by_individual, 'hair' : group_by_hair, 'beard' : group_by_beard}
	group_function = name2function[G]
	groups_list = [group_function(row, i, attr2idx) for i, row in enumerate(inputs)]
	return groups_list