import pandas as pd
import joblib

# Load test data (with 'score' already present)
df = pd.read_csv("test_model.csv")

# Feature engineering for binary matches
df['birthdate_1'] = pd.to_datetime(df['birthdate_1'], errors='coerce')
df['birthdate_2'] = pd.to_datetime(df['birthdate_2'], errors='coerce')

df['birthdate_match'] = (df['birthdate_1'] == df['birthdate_2']).astype(int)
df['first_name_match'] = (df['first_name_1'].str.lower() == df['first_name_2'].str.lower()).astype(int)
df['last_name_match'] = (df['last_name_1'].str.lower() == df['last_name_2'].str.lower()).astype(int)

# Load model
model = joblib.load('xgb_patient_match_model.pkl')

# Features to use, including score from the CSV file
features = ['score', 'birthdate_match', 'first_name_match', 'last_name_match']

# Predict
df['predicted_match'] = model.predict(df[features])
df['match_probability'] = model.predict_proba(df[features])[:, 1]

# Output results
print(df[['first_name_1', 'last_name_1', 'first_name_2', 'last_name_2', 'score', 'predicted_match', 'match_probability']])
df.to_csv("../match_predictions.csv", index=False)
