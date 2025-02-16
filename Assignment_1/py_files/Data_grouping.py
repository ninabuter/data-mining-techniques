import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('dataset_mood.csv', index_col='id')
print(df.shape)

# Convert 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'])

# Extract date from 'time' column
df['date'] = df['time'].dt.date

# Group the data by date and variable, then calculate the average value
result = df.groupby(['id', 'date', 'variable']).mean().reset_index()
# column_order = ['Unnamed: 0','time','variable','value']
# result = result[column_order]
result = result.sort_values(by=['id', 'variable', 'date'])
result = result.drop(columns=['Unnamed: 0', 'time'])
print(result.shape)
# (376912, 5)

result.to_csv('dataset_mood_grouped.csv', index=False)

# Display the result
print(result)


