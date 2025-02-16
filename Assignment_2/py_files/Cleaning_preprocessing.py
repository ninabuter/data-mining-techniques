import pandas as pd


"""mogge1 kies ff een input csv en een naam voor je output csv hierbeneden"""

input_csv = 'training_set_VU_DM.csv' #<<<< hier kiezen
output_csv = 'cleaned_training_set_VU_DM.csv' # <<< hier kiezen
df = pd.read_csv(input_csv)

country_score_averages = df.groupby('visitor_location_country_id')['prop_location_score2'].mean()

def calculate_weighted_score(counts):
    total_count = sum(counts)
    
    # Assigning weights (-1, 0, 1)
    weights = [-1, 0, 1]
    
    # Multiply counts by weights
    weighted_sum = sum(count * weight for count, weight in zip(counts, weights))
    
    # Calculate the score
    score = weighted_sum / total_count
    
    return score

def fillna_with_average(row):
    category_value = row['visitor_location_country_id']
    average = country_score_averages.get(category_value)
    if pd.isna(row['prop_location_score2']):
        return average
    else:
        return row['prop_location_score2']


####data cleaning#####

#0-fill cols
zero_fill_cols = ['visitor_hist_starrating','visitor_hist_adr_usd', 'gross_bookings_usd','srch_query_affinity_score']
df[zero_fill_cols] = df[zero_fill_cols].fillna(value=0)

#mean_fill_columns
mean_fill_cols = ['prop_review_score','orig_destination_distance']
df[mean_fill_cols] = df[mean_fill_cols].fillna(df[mean_fill_cols].mean())

#calculate competitor substitue values
#fill na values in competitor stats with weighted mean scores of -1, 0 an 1 for rate and inv
comp_stats_df = df[[col for col in df.columns if col.startswith('comp') and col.endswith(('rate','inv'))]]
comp_stats_counts = comp_stats_df.apply(pd.Series.value_counts)
comp_stats_subvalues = comp_stats_counts.apply(calculate_weighted_score)
df = df.fillna(comp_stats_subvalues)

#because percent_diff is already and absolute and calculated, use the average of the column to fill
rate_diff_cols = [col for col in df.columns if col.endswith('_rate_percent_diff')]
df[rate_diff_cols] = df[rate_diff_cols].fillna(df[rate_diff_cols].mean())

#fill missing location scores with average location score per country
df['prop_location_score2'] = df.apply(fillna_with_average, axis=1)

categorical_feats = ['site_id','visitor_location_country_id','prop_country_id','prop_id','srch_destination_id']


####feature engineering####
"""
add a column 'is_highest rated' 
based on wether the proposed hotel 
has the highest star rating of all hotels in the search.
"""
df['prop_is_highest_rated'] = df.groupby('srch_id')['prop_starrating'].transform(lambda x: x == x.max())

"""
add a column 'is_highest_reviewd' 
based on wether the proposed hotel 
has the highest review rating of all hotels in the search.
"""
df['prop_is_highest_reviewd'] = df.groupby('srch_id')['prop_review_score'].transform(lambda x: x == x.max())
"""
add a column 'mean_price_ranking' 
which is a ranked score from 0 to 1 on the mean price of the hotel
"""
df['prop_mean_price_ranking_historical'] = df.groupby('srch_id')['prop_log_historical_price'].transform(lambda x: (x.rank() - 1) / (x.nunique() - 1))
"""
add a column 'price_ranking' 
which is a ranked score from 0 to 1 on the mean price of the hotel
"""
df['prop_price_ranking'] = df.groupby('srch_id')['price_usd'].transform(lambda x: (x.rank() - 1) / (x.nunique() - 1))

# features = ["srch_id", "prop_id", "prop_location_score2", "random_bool", "prop_location_score1", "price_usd", "promotion_flag",
#             "prop_starrating", "prop_review_score", "prop_log_historical_price", "srch_room_count",
#             "comp8_rate_percent_diff", "srch_adults_count", "srch_children_count", "srch_booking_window",
#             "visitor_hist_adr_usd", "comp5_rate", "comp8_rate", "comp7_rate_percent_diff", "prop_country_id",
#             "comp5_rate_percent_diff", "prop_brand_bool", "booking_bool", "click_bool"]
#
# df = df[features]

df.to_csv(output_csv, index=False)