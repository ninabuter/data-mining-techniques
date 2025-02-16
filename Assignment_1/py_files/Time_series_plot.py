import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Import data
df = pd.read_csv('dataset_mood_smartphone2.csv')

# Initialize MinMaxScaler
# scaler = MinMaxScaler()

# Fit and transform all numerical columns
# df['value'] = scaler.fit_transform(df['value'].values.reshape(-1, 1))

# Get unique variable names
variables = df['variable'].unique()

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

# Group by date and variable, and aggregate values
daily_data = df.groupby(['date', 'variable']).agg({'value': 'mean'}).reset_index()

# Perform exploratory analysis
# For example, you can calculate descriptive statistics for each variable
descriptive_stats = daily_data.groupby('variable')['value'].describe()

# Calculate moving averages for each variable
window_size = 1  # Adjust the window size as needed
for variable in variables:
    daily_data[variable + '_MA'] = daily_data[daily_data['variable'] == variable]['value'].rolling(window=window_size).mean()

# Visualize trends over time with moving averages
plt.figure(figsize=(12, 8))

for variable in df['variable'].unique():
    plt.plot(daily_data[daily_data['variable'] == variable]['date'],
             daily_data[daily_data['variable'] == variable][variable + '_MA'], label=variable + ' (MA)')
plt.title('Trends Over Time with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Average Value')
plt.xticks(rotation=45)
plt.legend()
plt.tick_params(axis='x', which='major', labelsize=8)  # Adjust tick size
# plt.savefig('trends_over_time.png')
plt.show()

# Correlation analysis
correlation_matrix = daily_data.pivot_table(index='date', columns='variable', values='value', aggfunc='mean').corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix between Variables')
plt.show()
