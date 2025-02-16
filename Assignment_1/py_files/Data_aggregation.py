import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('dataset_mood.csv')

# Display all columns when printing
pd.set_option('display.max_columns', None)

# Convert the 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])

# Extract the date component from the 'time' column
df['date'] = df['time'].dt.date

# Count the frequency of all occurrences per id per day
# frequency_per_day = df.groupby(pd.Grouper(key='time', freq='D')).size().reset_index(name='frequency')
# frequency_per_day = df.groupby(['id', pd.Grouper(key='time', freq='D')]).size().reset_index(name='frequency')

# Count the frequency of occurrences of individual variables per id per day
frequencies = df.groupby(['id', 'date']).count().reset_index()
frequencies.columns = ['frequency_' + col if col not in ['id', 'date', 'time'] else col for col in frequencies.columns]
frequencies = frequencies.drop(columns=['id', 'date', 'time'])

# Convert the 'time' column in frequency_per_day DataFrame to datetime if it's not already
# frequency_per_day['time'] = pd.to_datetime(frequency_per_day['time'])

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

# Group by 'id' and day ('time'), then apply aggregation functions
df_aggregated = df.groupby(['id', df['time'].dt.date]).agg(agg_functions).reset_index()

# Merge the frequency_per_day df with the aggregated df
# df_aggregated_freq = pd.concat([df_aggregated, frequency_per_day['frequency']], axis=1)

# Merge the frequencies df with the aggregated df
df_aggregated_freq = pd.concat([df_aggregated, frequencies], axis=1)

# Backward filling the NA values that came up after aggregation
ids_df = df_aggregated_freq[['id']].copy()
df_aggregated_freq = df_aggregated_freq.groupby(['id']).bfill()
df_aggregated_freq = pd.merge(ids_df, df_aggregated_freq, left_index=True, right_index=True)

# Forward filling the NA values that came up after aggregation
df_aggregated_freq = df_aggregated_freq.groupby(['id']).ffill()
df_aggregated_freq = pd.merge(ids_df, df_aggregated_freq, left_index=True, right_index=True)

# Interpolation might be an option?
# df_aggregated_freq = df_aggregated_freq.reset_index()
# df_aggregated_freq = df_aggregated_freq.groupby('id').apply(lambda group: group.interpolate(method='linear'))
# df_aggregated_freq = df_aggregated_freq.droplevel(0)
# df_aggregated_freq = df_aggregated_freq.reset_index(drop=True)

# Rolling statistics imputation for the NA values that came up after aggregation
# rolling_window_size = 10
# numeric_columns = df_aggregated_freq.select_dtypes(include=[np.number]).columns
# df_aggregated_freq[numeric_columns] = df_aggregated_freq.groupby('id')[numeric_columns].transform(lambda x: x.fillna(
#     x.rolling(window=rolling_window_size, min_periods=1).mean()))
# df_aggregated_freq = df_aggregated_freq.reset_index(drop=True)

# Convert to csv
df_aggregated_freq.to_csv('dataset_mood_bfill_aggregated.csv', index=False)