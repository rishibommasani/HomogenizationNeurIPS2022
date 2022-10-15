import pandas as pd
import numpy as np


if __name__ == '__main__':
	df = pd.read_csv('firstnames.csv') 
	names = {name.lower().replace("'", "") for name in df['firstname']}
	races = ['hispanic', 'white', 'black', 'api']
	race2names = {race : set() for race in races}
	for index, row in df.iterrows():
		race_percents = [row['pct' + race] for race in races]
		statistically_associated_race_position = np.argmax(race_percents)
		statistically_associated_race = races[statistically_associated_race_position]
		name = row['firstname'].lower().replace("'", "")
		race2names[statistically_associated_race].add(name)
	print('names = ', names)
	print('\n')
	for race, names_list in race2names.items():
		print(race, ' = ', names_list, '\n')
