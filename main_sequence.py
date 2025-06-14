import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from difflib import SequenceMatcher
import itertools

# --- 1. Load Data ---
lower_neno = pd.read_csv('data/lower_neno_data.csv')
upper_neno = pd.read_csv('data/upper_neno_data.csv')

# Add db column for consistency
lower_neno['db'] = 'lower_neno'
upper_neno['db'] = 'upper_neno'

# --- 2. Clean and Normalize Data ---
def clean_names(df):
    df['first_name'] = df['first_name'].astype(str).str.strip().str.lower()
    df['last_name'] = df['last_name'].astype(str).str.strip().str.lower()
    df['gender'] = df['gender'].astype(str).str.upper().str.strip()
    df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
    df['art_first_line_regimen_start_date'] = pd.to_datetime(df['art_first_line_regimen_start_date'], errors='coerce')
    df['first_visit_date'] = pd.to_datetime(df['first_visit_date'], errors='coerce')
    df['last_visit_date'] = pd.to_datetime(df['last_visit_date'], errors='coerce')
    return df

lower_neno = clean_names(lower_neno)
upper_neno = clean_names(upper_neno)

# --- 3. Add Blocking Keys ---
def add_blocking_keys(df):
    df['birth_year'] = df['birthdate'].dt.year
    return df

lower_neno = add_blocking_keys(lower_neno)
upper_neno = add_blocking_keys(upper_neno)

# --- 4. Generate Candidate Pairs via Blocking ---
blocked_pairs = []
for gender in lower_neno['gender'].dropna().unique():
    for year in lower_neno['birth_year'].dropna().unique():
        block_l = lower_neno[(lower_neno['gender'] == gender) & (lower_neno['birth_year'] == year)]
        block_u = upper_neno[(upper_neno['gender'] == gender) & (upper_neno['birth_year'] == year)]
        pairs = list(itertools.product(block_l.to_dict('records'), block_u.to_dict('records')))
        blocked_pairs.extend(pairs)

print(f"Blocked candidate pairs: {len(blocked_pairs):,}")

# --- 5. Compute Similarity Features ---
def string_similarity(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()

def create_features(pair):
    rec1, rec2 = pair
    return {
        'first_name_sim': string_similarity(rec1['first_name'], rec2['first_name']),
        'last_name_sim': string_similarity(rec1['last_name'], rec2['last_name']),
        'gender_match': int(rec1['gender'] == rec2['gender']),
        'birthdate_match': int(rec1['birthdate'] == rec2['birthdate']),
        'regimen_match': int(rec1['art_first_line_regimen'] == rec2['art_first_line_regimen']),
        'location_sim': string_similarity(rec1.get('enrollment_location', ''), rec2.get('enrollment_location', '')),
    }

features = [create_features(pair) for pair in blocked_pairs]
df_features = pd.DataFrame(features)

# --- 6. Score Matching Pairs ---
df_features['score'] = (
    df_features['first_name_sim'] * 0.3 +
    df_features['last_name_sim'] * 0.3 +
    df_features['gender_match'] * 0.1 +
    df_features['birthdate_match'] * 0.2 +
    df_features['regimen_match'] * 0.1
)

# --- 7. Threshold-Based Filtering ---
threshold = 0.85
matched_indices = df_features[df_features['score'] >= threshold].index
matched_pairs = [blocked_pairs[i] for i in matched_indices]

# --- 8. Output Matched Pairs ---
matched_df = pd.DataFrame([
    {
        'lower_id': pair[0]['patient_id'],
        'upper_id': pair[1]['patient_id'],
        'first_name_1': pair[0]['first_name'],
        'first_name_2': pair[1]['first_name'],
        'last_name_1': pair[0]['last_name'],
        'last_name_2': pair[1]['last_name'],
        'birthdate_1': pair[0]['birthdate'],
        'birthdate_2': pair[1]['birthdate'],
        'score': df_features.iloc[i]['score']
    }
    for i, pair in zip(matched_indices, matched_pairs)
])

matched_df.to_csv('../matched_patients.csv', index=False)
print(f"✅ Matching complete. {len(matched_df)} matches saved to 'matched_patients.csv'.")
