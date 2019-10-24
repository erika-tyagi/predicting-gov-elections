import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
import us  

pd.options.mode.chained_assignment = None


def get_abbr(state_name): 
    states_dict = us.states.mapping('name', 'abbr')
    state_abbr =  states_dict[state_name]
    return state_abbr


def wrangle_dem_vtshr(filename, newvar, dem_var, rep_var, state_var): 
	results = pd.read_csv(filename)
	results[newvar] = results[dem_var] / (results[dem_var] + results[rep_var])
	results = results[[state_var, newvar]]
	return results 


def append_rows(df, row_dicts): 
	for r in row_dicts: 
		df = df.append(r, ignore_index = True) 
	return df 


def wrangle_incumbents(filename): 
	incumbents = pd.read_csv(filename)
	incumbents = incumbents[incumbents['incumbent']]
	incumbents['dem_incumbent'] = np.where(incumbents['party'] == 'D', 1, 0)
	incumbents['rep_incumbent'] = np.where(incumbents['party'] == 'R', 1, 0)
	incumbents = incumbents[['state', 'dem_incumbent', 'rep_incumbent']].drop_duplicates()
	return incumbents


def wrangle_dem_poll(filename): 
	polls = pd.read_csv(filename)
	polls = polls[polls['candidate_party'].isin(['DEM', 'REP'])]
	polls['two_party_vtsh'] = polls['pct'].groupby(polls['question_id']).transform('sum')
	polls['dem_vtsh_poll'] = polls['pct'] / polls['two_party_vtsh']
	polls['start_date'] = pd.to_datetime(polls['start_date'])
	polls = polls[polls['candidate_party'] == 'DEM'].sort_values(by = ['start_date']).drop_duplicates(subset = 'state', keep = 'last')
	polls = polls[['state', 'dem_vtsh_poll']]
	return polls 


def split(df, target, features): 
	train = df[(df[target].notna())].dropna()
	test = df[(df[target].isna())]
	X_train = train[features]
	y_train = train[[target]]
	X_test = test[features]
	return X_train, y_train, X_test, test 



