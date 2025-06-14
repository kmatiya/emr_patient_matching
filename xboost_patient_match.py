import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib

# 1. Load data
df = pd.read_csv('../output/matched_patients_verified.csv')

# 2. Preprocessing
df['matched'] = df['matched'].map({'yes': 1, 'no': 0})  # encode target

df['birthdate_1'] = pd.to_datetime(df['birthdate_1'], errors='coerce')
df['birthdate_2'] = pd.to_datetime(df['birthdate_2'], errors='coerce')

# --- Feature engineering ---
# Fuzzy birthdate match (already computed if imported), but ensure consistency
df['birthdate_match'] = (df['birthdate_1'] == df['birthdate_2']).astype(int)

# Fill NaNs for numerical features to prevent model errors
df['height_match_score'] = df['height_match_score'].fillna(0)
df['height_min_diff'] = df['height_min_diff'].fillna(100)  # large default to show mismatch
df['regimen_match_score'] = df['regimen_match_score'].fillna(0)
df['first_name_sim'] = df['first_name_sim'].fillna(0)
df['last_name_sim'] = df['last_name_sim'].fillna(0)

# 3. Features and target
features = [
    'score',
    'first_name_sim',
    'last_name_sim',
    'birthdate_match',
    'height_match_score',
    'height_min_diff',
    'regimen_match_score'
]
X = df[features]
y = df['matched']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 5. Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 6. Predict and evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")

# 7. Save model
joblib.dump(model, 'xgb_patient_match_model.pkl')
print("✅ Model saved as xgb_patient_match_model.pkl")
