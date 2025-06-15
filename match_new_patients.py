import pandas as pd
from rapidfuzz import fuzz
import itertools

# --- 1. Load Data ---
new_patients = pd.read_csv('../data/lower_neno_data.csv')
existing_patient = pd.read_csv('../data/upper_neno_data.csv')

# Add db column
new_patients['db'] = 'new_patient'
existing_patient['db'] = 'existing_patient'

# --- 2. Clean and Normalize ---
def clean(df):
    df['first_name'] = df['first_name'].astype(str).str.strip().str.lower()
    df['last_name'] = df['last_name'].astype(str).str.strip().str.lower()
    df['gender'] = df['gender'].astype(str).str.upper().str.strip()
    df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
    df['art_first_line_regimen_start_date'] = pd.to_datetime(df['art_first_line_regimen_start_date'], errors='coerce')
    df['first_visit_date'] = pd.to_datetime(df['first_visit_date'], errors='coerce')
    df['last_visit_date'] = pd.to_datetime(df['last_visit_date'], errors='coerce')
    return df

new_patients = clean(new_patients)
existing_patient = clean(existing_patient)

# --- 3. Add Blocking Key ---
new_patients['birth_year'] = new_patients['birthdate'].dt.year
existing_patient['birth_year'] = existing_patient['birthdate'].dt.year

# --- 4. Blocking ---
blocked_pairs = []
for gender in new_patients['gender'].dropna().unique():
    for year in new_patients['birth_year'].dropna().unique():
        n_block = new_patients[(new_patients['gender'] == gender) & (new_patients['birth_year'] == year)]
        e_block = existing_patient[(existing_patient['gender'] == gender) & (existing_patient['birth_year'] == year)]
        pairs = list(itertools.product(n_block.to_dict('records'), e_block.to_dict('records')))
        blocked_pairs.extend(pairs)

print(f"Blocked pairs: {len(blocked_pairs)}")

# --- 5. Feature Engineering ---
def string_similarity(a, b):
    return fuzz.token_sort_ratio(str(a), str(b)) / 100

def fuzzy_birthdate_match(bd1, bd2, tol=60):
    if pd.isnull(bd1) or pd.isnull(bd2):
        return 0
    return int(abs((bd1 - bd2).days) <= tol)

def compute_height_similarity(h1_first, h1_last, h2_first, h2_last, tolerance=2):
    heights_1 = [h1_first, h1_last]
    heights_2 = [h2_first, h2_last]
    match_count = 0
    diffs = []

    for h1 in heights_1:
        for h2 in heights_2:
            if h1 > 0 and h2 > 0:
                diff = abs(h1 - h2)
                diffs.append(diff)
                if diff <= tolerance:
                    match_count += 1

    match_score = match_count / 4
    min_diff = min(diffs) if diffs else None
    return match_score, min_diff

def compute_regimen_similarity(r1_first, r1_last, r2_first, r2_last):
    regimens_1 = [r1_first, r1_last]
    regimens_2 = [r2_first, r2_last]
    match_count = 0

    for r1 in regimens_1:
        for r2 in regimens_2:
            if pd.notnull(r1) and pd.notnull(r2) and r1 == r2:
                match_count += 1

    return match_count / 4

def create_features(pair):
    rec1, rec2 = pair

    try:
        h1_first = float(rec1.get('height_at_first_visit', 0))
        h1_last = float(rec1.get('height_at_last_visit', 0))
        h2_first = float(rec2.get('height_at_first_visit', 0))
        h2_last = float(rec2.get('height_at_last_visit', 0))
    except:
        h1_first = h1_last = h2_first = h2_last = 0

    height_match_score, height_min_diff = compute_height_similarity(h1_first, h1_last, h2_first, h2_last)

    regimen_match_score = compute_regimen_similarity(
        rec1.get('regimen_at_first_visit'),
        rec1.get('regimen_at_last_visit'),
        rec2.get('regimen_at_first_visit'),
        rec2.get('regimen_at_last_visit')
    )

    return {
        'first_name_sim': string_similarity(rec1['first_name'], rec2['first_name']),
        'last_name_sim': string_similarity(rec1['last_name'], rec2['last_name']),
        'gender_match': int(rec1['gender'] == rec2['gender']),
        'birthdate_match': fuzzy_birthdate_match(rec1['birthdate'], rec2['birthdate']),
        'regimen_match_score': regimen_match_score,
        'height_match_score': height_match_score,
        'height_min_diff': height_min_diff,
        'regimen_new_first': rec1.get('regimen_at_first_visit'),
        'regimen_new_last': rec1.get('regimen_at_last_visit'),
        'regimen_existing_first': rec2.get('regimen_at_first_visit'),
        'regimen_existing_last': rec2.get('regimen_at_last_visit'),
        'height_new_first': rec1.get('height_at_first_visit'),
        'height_new_last': rec1.get('height_at_last_visit'),
        'height_existing_first': rec2.get('height_at_first_visit'),
        'height_existing_last': rec2.get('height_at_last_visit'),
        'pair': (rec1, rec2)
    }

features = [create_features(p) for p in blocked_pairs]
df_features = pd.DataFrame(features)

# --- 6. Scoring ---
df_features['score'] = (
    df_features['first_name_sim'] * 0.3 +
    df_features['last_name_sim'] * 0.3 +
    df_features['gender_match'] * 0.1 +
    df_features['birthdate_match'] * 0.17 +
    df_features['height_match_score'] * 0.1 +
    df_features['regimen_match_score'] * 0.03
)

# --- 7. Threshold Matching ---
threshold = 0.75
matched = df_features[df_features['score'] >= threshold]

# --- 8. Output ---
output = pd.DataFrame([
    {
        'new_patient_id': p['pair'][0]['patient_id'],
        'existing_patient_id': p['pair'][1]['patient_id'],
        'first_name_1': p['pair'][0]['first_name'],
        'first_name_2': p['pair'][1]['first_name'],
        'last_name_1': p['pair'][0]['last_name'],
        'last_name_2': p['pair'][1]['last_name'],
        'birthdate_1': p['pair'][0]['birthdate'],
        'birthdate_2': p['pair'][1]['birthdate'],
        'regimen_new_first': p['regimen_new_first'],
        'regimen_new_last': p['regimen_new_last'],
        'regimen_existing_first': p['regimen_existing_first'],
        'regimen_existing_last': p['regimen_existing_last'],
        'height_new_first': p['height_new_first'],
        'height_new_last': p['height_new_last'],
        'height_existing_first': p['height_existing_first'],
        'height_existing_last': p['height_existing_last'],
        'regimen_match_score': p['regimen_match_score'],
        'height_match_score': p['height_match_score'],
        'height_min_diff': p['height_min_diff'],
        'gender_match': p['gender_match'],
        'birthdate_match': p['birthdate_match'],
        'first_name_sim': p['first_name_sim'],
        'last_name_sim': p['last_name_sim'],
        'score': p['score']
    }
    for _, p in matched.iterrows()
])

output.to_csv('../output/matched_patients_bidirectional.csv', index=False)
print(f"✅ Done. {len(output)} matched pairs saved to 'matched_patients_bidirectional.csv'")
