import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data and check the shape and such
df = pd.read_csv('dataset_mood_smartphone.csv')
print("Dimensions of the dataset:", df.shape)
print("\nFirst few rows of the dataset:")
print(df.head())
print("\nData types of each column:")
print(df.dtypes)

# Summary statistics
print("\nUnique categories and their frequency counts for categorical variables:")
print(df['variable'].value_counts())

# Group by 'variable' and calculate the average of 'value' for each group
average_values = df.groupby('variable')['value'].mean()
print("\nAverage value for each variable:")
print(average_values)

# Group by 'variable' and calculate the range of 'value' for each group
range_values = df.groupby('variable')['value'].agg(lambda x: x.max() - x.min())
print("\nRange of value for each variable:")
print(range_values)

# Group by 'variable' and calculate the min and max value of 'value' for each group
min_max_values = df.groupby('variable')['value'].agg(['min', 'max'])
print("\nMin and Max value for each variable:")
print(min_max_values)

# Create a boxplot for each group defined by the 'variable' column
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='variable', y='value', data=df)
# plt.title('Boxplot of Value Distribution for each Variable')
# plt.xlabel('Variable')
# plt.ylabel('Value')
# plt.show()

# Create a violin plot for each group defined by the 'variable' column
# plt.figure(figsize=(10, 6))
# sns.violinplot(x='variable', y='value', data=df)
# plt.title('Violin plot of Value Distribution for each Variable')
# plt.xlabel('Variable')
# plt.ylabel('Value')
# plt.show()

# Correlation analysis
# correlation_matrix = df.pivot_table(index='id', columns='variable', values='value')
# correlation_matrix = correlation_matrix.corr()
# print("Correlation Matrix:")
# print(correlation_matrix)

# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap')
# plt.xlabel('Variable')
# plt.ylabel('Variable')
# plt.show()

for group_name, group_data in df.groupby('variable'):
    plt.figure(figsize=(12, 6))

    # Plot histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data=group_data, x='value', kde=True)
    plt.title(f'Distribution of Values for {group_name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Plot boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=group_data, x='value')
    plt.title(f'Boxplot of Values for {group_name}')
    plt.xlabel('Value')

    plt.tight_layout()
    plt.show()

