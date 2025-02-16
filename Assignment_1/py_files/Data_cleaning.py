import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('dataset_mood_smartphone.csv')

# Check for missing values in the attributes
# missing_values_per_attribute = df.groupby('variable').apply(lambda x: x.isna().sum())
# print("\nMissing values per attribute in the 'variable' column:")
# print(missing_values_per_attribute)
# Outcome: circumplex.valence and circumplex.arousal have some missing (NA) values

# Backward-Fill for the NA values in 'value' column
df['value'] = df['value'].fillna(method='bfill')

# Rolling statistics imputation
# window_size = 10
# rolling_mean = df['value'].rolling(window=window_size, min_periods=1).mean()
# df['value_filled'] = df['value'].fillna(rolling_mean)

# Drop rows with negative values for appCat.entertainment and appCat.builtin (they contain negative values which they
# shouldn't have
df = df[(df['variable'] != 'appCat.entertainment') | (df['value'] >= 0)]
df = df[(df['variable'] != 'appCat.builtin') | (df['value'] >= 0)]

# Change the df from all variables in one column to individual columns for every variable
df['time'] = pd.to_datetime(df['time'])
df = pd.pivot_table(df, values='value', index=['id','time'], columns='variable').reset_index()

# Since most variables started 2014-03-20, we will drop the rows before that time
df = df[df['time'] >= '2014-03-20']
df.reset_index(drop=True, inplace=True)  # Reset index

# Since we have a high number of values above the upper bound (right skewness), we can reduce their effect by
# log-transforming the data
cols_to_transform = ['activity', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment', 'appCat.finance',
                     'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown',
                     'appCat.utilities', 'appCat.weather', 'screen']
df[cols_to_transform] = df[cols_to_transform].apply(np.log1p)

# Convert to CSV file
df.to_csv('dataset_mood.csv', index=False)

