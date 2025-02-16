import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv("dataset_mood_bfill_aggregated.csv")

# Split the data into features and target variable
# X = df.drop(columns=['id', 'time', 'mood'])
# y = df['mood']

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Time series data split (variation of K-fold)
unique_ids = df['id'].unique()  # Get unique user IDs

# Define the number of splits
n_splits = 5  # or any other number of splits you prefer

# Initialize the TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=n_splits)

# Iterate over the unique user IDs
for user_id in unique_ids:
    user_data = df[df['id'] == user_id]  # Get data for the current user ID
    X = user_data.drop(columns=['id', 'mood'])
    y = user_data['mood']

    # Iterate over the splits for the current user ID
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# # Initialize the Random Forest Regressor
# rf_regressor = RandomForestRegressor()
#
# # Fit the model
# rf_regressor.fit(X_train, y_train)
#
# # Predict
# y_pred = rf_regressor.predict(X_test)
#
# # Evaluate
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)
#
# # Define the parameter grid to search
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# # Initialize the RandomForestRegressor
# rf_regressor = RandomForestRegressor()
#
# # Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
#
# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)
#
# # Get the best parameters
# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)
#
# # Get the best model
# best_model = grid_search.best_estimator_
#
# # Evaluate the best model
# best_model_score = best_model.score(X_test, y_test)
# print("Best Model R2 Score:", best_model_score)