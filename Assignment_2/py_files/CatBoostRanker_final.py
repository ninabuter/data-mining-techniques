import sys
from catboost import CatBoostRanker, Pool
from copy import deepcopy
import numpy as np
import pandas as pd

# Default
np.random.seed(42)

# Load the dataset
df = pd.read_csv("training_set_VU_DM.csv")
df_test = pd.read_csv("test_set_VU_DM.csv")

# Split
unique_srch_id = np.unique(df['srch_id'])
evaluation_srch_id = np.random.choice(unique_srch_id, size=int(np.ceil(len(unique_srch_id)*0.2)), replace=False)
df_train = df[~df['srch_id'].isin(evaluation_srch_id)]
df_evaluation = df[df['srch_id'].isin(evaluation_srch_id)]

# Define features

# Beste score:
# features = ['site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'prop_starrating',
#             'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2',
#             'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay',
#             'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count',
#             'srch_saturday_night_bool', 'orig_destination_distance', 'random_bool']

# Thijs' data cleaning en preprocessing:
# features = df.drop(['srch_id', 'date_time', 'booking_bool', 'click_bool', 'gross_bookings_usd', 'position'], axis=1)

# Alle features (exclusief de features die niet in de testset zijn)
features = ["site_id", "visitor_location_country_id", "visitor_hist_starrating",
            "visitor_hist_adr_usd", "prop_country_id", "prop_id", "prop_starrating", "prop_review_score",
            "prop_brand_bool", "prop_location_score1", "prop_location_score2", "prop_log_historical_price",
            "price_usd", "promotion_flag", "srch_destination_id", "srch_length_of_stay",
            "srch_booking_window", "srch_adults_count", "srch_children_count", "srch_room_count",
            "srch_saturday_night_bool", "srch_query_affinity_score", "orig_destination_distance",
            "random_bool", "comp1_rate", "comp1_inv", "comp1_rate_percent_diff", "comp2_rate", "comp2_inv",
            "comp2_rate_percent_diff", "comp3_rate", "comp3_inv", "comp3_rate_percent_diff", "comp4_rate",
            "comp4_inv", "comp4_rate_percent_diff", "comp5_rate", "comp5_inv", "comp5_rate_percent_diff",
            "comp6_rate", "comp6_inv", "comp6_rate_percent_diff", "comp7_rate", "comp7_inv",
            "comp7_rate_percent_diff", "comp8_rate", "comp8_inv", "comp8_rate_percent_diff"]

# De beste 20 features na feature importance
# features = ["prop_location_score2", "random_bool", "prop_location_score1", "price_usd", "promotion_flag",
#             "prop_starrating", "prop_review_score", "prop_log_historical_price", "srch_room_count",
#             "comp8_rate_percent_diff", "srch_adults_count", "srch_children_count", "srch_booking_window",
#             "visitor_hist_adr_usd", "comp5_rate", "comp8_rate", "comp7_rate_percent_diff", "prop_country_id",
#             "comp5_rate_percent_diff", "prop_brand_bool"]

# Generate train, evaluation, and test datasets
X_train = df_train[features]  # select columns from the features dataframe
X_train_ID = df_train['srch_id']
y_train = np.maximum(5 * df_train['booking_bool'], df_train['click_bool'])
y_train_scaled = (y_train - y_train.min()) / (y_train.max() - y_train.min())  # Used for Grid Search

X_eval = df_evaluation[features]  # select columns from the features dataframe
X_eval_ID = df_evaluation['srch_id']
y_eval = np.maximum(5 * df_evaluation['booking_bool'], df_evaluation['click_bool'])

X_test = df_test[features]  # select columns from the features dataframe
X_test_ID = df_test['srch_id']

# Generate train and evaluation pools
train = Pool(
    data=X_train,
    label=y_train,  # y_train_scaled for grid search
    group_id=X_train_ID
)

evaluation = Pool(
    data=X_eval,
    label=y_eval,
    group_id=X_eval_ID
)

test = Pool(
    data=X_test,
    group_id=X_test_ID
)

# Grid search
# grid_model = CatBoostRanker()
# param_grid = {'learning_rate': [0.01, 0.03, 0.05, 0.1], 'depth': [4, 6, 8, 10], 'iterations': [50, 100, 150, 200]}
# grid_search_result = grid_model.grid_search(param_grid, train, plot=True)
# print(grid_search_result)
# print("Best parameters:", grid_search_result['params'])

# Initialize CBR model (with the best parameters after the grid search)
default_parameters = {
    'depth': 6,  # grid search
    'learning_rate': 0.1,  # grid search
    'iterations': 500,  # grid search
    'custom_metric': ['NDCG', 'PFound', 'AverageGain:top=10'],
    'verbose': True,
    'random_seed': 0,
}

parameters = {}


def fit_model(loss_function, additional_params=None, train_pool=train, eval_pool=evaluation):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function

    if additional_params is not None:
        parameters.update(additional_params)

    model = CatBoostRanker(**parameters)
    model.fit(train_pool, eval_set=eval_pool, plot=True)

    return model


# Train the model and get NDCG score after training
model = fit_model('RMSE', {'custom_metric': ['NDCG']})
print(model.get_evals_result())

# Get the feature importance from the training set
feature_importance = model.get_feature_importance(data=train, type="LossFunctionChange")
feature_importance_id = zip(X_train.columns, feature_importance)
sorted_feature_importance = sorted(feature_importance_id, key=lambda x: x[1], reverse=True)
for feature_name, importance_score in sorted_feature_importance:
    print(feature_name, importance_score)

# Make predictions of the test dataset with the CBR model
# predictions = model.predict(test)

# Combine predictions with search ID and property ID, sort by search ID and predictions, and save for submission
# submission = pd.DataFrame({'srch_id': df_test['srch_id'], 'prop_id': df_test['prop_id'], 'Prediction': predictions})
# sorted_submission = submission.sort_values(by=['srch_id', 'Prediction'], ascending=[True, False])
# sorted_submission.to_csv("VU-DM-2024-Group-61.csv", columns=['srch_id', 'prop_id'], index=False)

