import pandas as pd
import numpy as np


def filter_rows(group):
    freq_count = group.sum(axis=1)
    non_zero_index = freq_count.ne(0).idxmax()
    if (group.index.get_level_values('id') == 'AS14.01').any():
        return group.loc[('AS14.01', pd.Timestamp('2014-03-21 00:00:00')):]
    elif (group.index.get_level_values('id') == 'AS14.02').any():
        return group.loc[('AS14.02', pd.Timestamp('2014-03-16 00:00:00')):]
    elif (group.index.get_level_values('id') == 'AS14.12').any():
        return group.loc[('AS14.12', pd.Timestamp('2014-03-27 00:00:00')):]
    elif (group.index.get_level_values('id') == 'AS14.17').any():
        return group.loc[('AS14.17', pd.Timestamp('2014-03-21 00:00:00')):]
    elif (group.index.get_level_values('id') == 'AS14.26').any():
        return group.loc[('AS14.26', pd.Timestamp('2014-04-13 00:00:00')):]
    else:
        return group.loc[non_zero_index:]


FILL_MODE = 'window'
assert FILL_MODE in ['ffil', 'window']

# Load the data
df = pd.read_csv('dataset_mood_smartphone.csv', index_col=0)

# First round of filling, before the pivot
if FILL_MODE == 'ffil':  # Forward-Fill for the NA values in 'value' column
    df['value'] = df['value'].ffill()
elif FILL_MODE == 'window':  # Rolling statistics imputation
    window_size = 10
    rolling_mean = df['value'].rolling(window=window_size, min_periods=1).mean()
    df['value'] = df['value'].fillna(rolling_mean)

# Change the df from all variables in one column to individual columns for every variable
df['time'] = pd.to_datetime(df['time'])
df = pd.pivot_table(df, values='value', index=['id','time'], columns='variable')

# Drop rows with negative values for appCat.entertainment and appCat.builtin (they contain negative values which they
# shouldn't have
non_negative_cols = [col for col in df.columns if col.startswith('appCat')]
rows_to_drop = df[(df[non_negative_cols] < 0).any(axis=1)]
df = df.drop(rows_to_drop.index)

# Since we have a high number of values above the upper bound (right skewness), we can reduce their effect by
# log-transforming the data
cols_to_transform = ['activity', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment', 'appCat.finance',
                     'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown',
                     'appCat.utilities', 'appCat.weather', 'screen']

df[cols_to_transform] = df[cols_to_transform].apply(np.log1p)

# Calculate frequencies
frequencies = df.groupby([pd.Grouper(level='id'), pd.Grouper(level='time', freq='1D')]).count()
frequencies.columns = ['frequency_' + col for col in frequencies.columns]
frequencies = frequencies.drop(columns=['frequency_call', 'frequency_sms'])
frequencies = frequencies.groupby(level='id', group_keys=False).apply(filter_rows)
# frequencies.to_csv('filtered_frequencies.csv')

# Define the aggregation functions for different columns
agg_functions = {
    'mood': 'mean',
    'circumplex.arousal': 'mean',
    'circumplex.valence': 'mean',
    'activity': 'mean',
    'call': 'sum',
    'sms': 'sum',
    'appCat.builtin': 'sum',
    'appCat.communication': 'sum',
    'appCat.entertainment': 'sum',
    'appCat.finance': 'sum',
    'appCat.game': 'sum',
    'appCat.office': 'sum',
    'appCat.other': 'sum',
    'appCat.social': 'sum',
    'appCat.travel': 'sum',
    'appCat.unknown': 'sum',
    'appCat.utilities': 'sum',
    'appCat.weather': 'sum'
}

df = df.groupby([pd.Grouper(level='id'), pd.Grouper(level='time', freq='1D')]).agg(agg_functions)

# df.to_csv('test.csv')
df = frequencies.join(df)

# Second round of filling, after the pivot (much more datapoints are created, so a lot of new NA values)
df['activity'] = df['activity'].fillna(0)
if FILL_MODE == 'ffil':  # Forward-Fill for the NA values in 'value' column
    df['mood'] = df['mood'].ffill()
    df['circumplex.arousal'] = df['circumplex.arousal'].ffill()
    df['circumplex.valence'] = df['circumplex.valence'].ffill()
elif FILL_MODE == 'window':  # Rolling statistics imputation
    window_size = 10
    rolling_mean_mood = df['mood'].rolling(window=window_size, min_periods=1).mean()
    df['mood'] = df['mood'].fillna(rolling_mean_mood)
    rolling_mean_arousal = df['circumplex.arousal'].rolling(window=window_size, min_periods=1).mean()
    df['circumplex.arousal'] = df['circumplex.arousal'].fillna(rolling_mean_arousal)
    rolling_mean_valence = df['circumplex.valence'].rolling(window=window_size, min_periods=1).mean()
    df['circumplex.valence'] = df['circumplex.valence'].fillna(rolling_mean_valence)

print(df.isna().sum())

df.to_csv('dataset_cleaned.csv')
