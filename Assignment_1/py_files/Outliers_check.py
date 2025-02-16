import pandas as pd

# Load the data
df = pd.read_csv('dataset_mood.csv')

# Set option to display all rows
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Filter the df for each variable with 'value' higher than a certain point
# filtered_data = df[(df['variable'] == 'appCat.travel') & (df['value'] > 2000)]

# Display the filtered data
# print(filtered_data)


# Define a function to calculate IQR and boundaries for outliers
def calculate_outlier_boundaries(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return pd.Series([Q1, Q3, IQR, lower_bound, upper_bound], index=['Q1', 'Q3', 'IQR', 'Lower Bound', 'Upper Bound'])


# Calculate outlier boundaries for each numerical column except the first two columns
outlier_boundaries = df.iloc[:, 2:].apply(calculate_outlier_boundaries)

# Display the result
print(outlier_boundaries)

# Filter the DataFrame to select only the numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64'])

# Initialize dictionaries to store the counts of outliers for each category
high_outliers = {}
low_outliers = {}

# Iterate over each numerical column
for column in numerical_columns:
    # Get the lower and upper bounds for the current variable
    lower_bound = outlier_boundaries.loc['Lower Bound', column]
    upper_bound = outlier_boundaries.loc['Upper Bound', column]

    # Count the number of high outliers (above the upper bound) for the current variable
    high_outliers[column] = (df[column] > upper_bound).sum()

    # Count the number of low outliers (below the lower bound) for the current variable
    low_outliers[column] = (df[column] < lower_bound).sum()

# Convert the dictionaries to DataFrames
high_outliers_df = pd.DataFrame.from_dict(high_outliers, orient='index', columns=['high_outliers'])
low_outliers_df = pd.DataFrame.from_dict(low_outliers, orient='index', columns=['low_outliers'])

# Concatenate the DataFrames to combine the counts of high and low outliers
outlier_counts_df = pd.concat([high_outliers_df, low_outliers_df], axis=1)

# Display the result
print(outlier_counts_df)