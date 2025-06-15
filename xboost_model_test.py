import pandas as pd
import joblib

# Load your feature dataframe for new pairs
df_features = pd.read_csv('../output/new_patient_pair_features.csv')
# Make sure this dataframe contains all the features used in training, e.g.:
# ['score', 'first_name_sim', 'last_name_sim', 'birthdate_match', 'height_match_score', 'height_min_diff', 'regimen_match_score']

# Load your saved model
model = joblib.load('xgb_patient_match_model.pkl')

# Select features columns as you used in training
features = [
    'score',
    'first_name_sim',
    'last_name_sim',
    'birthdate_match',
    'height_match_score',
    'height_min_diff',
    'regimen_match_score'
]
X_new = df_features[features]

# Predict match probabilities for each pair
df_features['match_probability'] = model.predict_proba(X_new)[:, 1]

# You can set thresholds to classify or prioritize for manual review
threshold_high = 0.75
threshold_low = 0.3

df_features['match_label'] = 'manual_review'
df_features.loc[df_features['match_probability'] >= threshold_high, 'match_label'] = 'match'
df_features.loc[df_features['match_probability'] <= threshold_low, 'match_label'] = 'no_match'

# Save output with predictions
df_features.to_csv('../output/new_patient_pair_predictions.csv', index=False)

print(f"Predictions done on {len(df_features)} pairs. Saved to 'new_patient_pair_predictions.csv'")
