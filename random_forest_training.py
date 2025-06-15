import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib

# 1. Load labeled data
df = pd.read_csv('../output/matched_patients_verified.csv')

# 2. Preprocessing
df['matched'] = df['matched'].map({'yes': 1, 'no': 0})

df['birthdate_1'] = pd.to_datetime(df['birthdate_1'], errors='coerce')
df['birthdate_2'] = pd.to_datetime(df['birthdate_2'], errors='coerce')

df['birthdate_match'] = (df['birthdate_1'] == df['birthdate_2']).astype(int)

df['height_match_score'] = df['height_match_score'].fillna(0)
df['height_min_diff'] = df['height_min_diff'].fillna(100)
df['regimen_match_score'] = df['regimen_match_score'].fillna(0)
df['first_name_sim'] = df['first_name_sim'].fillna(0)
df['last_name_sim'] = df['last_name_sim'].fillna(0)

# 3. Features and Target
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

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Evaluate
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")

# 7. Save model
joblib.dump(rf_model, 'rf_patient_match_model.pkl')
print("✅ Model saved as rf_patient_match_model.pkl")
