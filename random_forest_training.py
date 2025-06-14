import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load data
df = pd.read_csv('matched_patients_rapid_fuzz_verified.csv')

# 2. Preprocessing
df['matched'] = df['matched'].map({'yes': 1, 'no': 0})  # encode target

df['birthdate_1'] = pd.to_datetime(df['birthdate_1'], errors='coerce')
df['birthdate_2'] = pd.to_datetime(df['birthdate_2'], errors='coerce')

# Feature engineering: binary exact matches for names and birthdates
df['birthdate_match'] = (df['birthdate_1'] == df['birthdate_2']).astype(int)
df['first_name_match'] = (df['first_name_1'].str.lower() == df['first_name_2'].str.lower()).astype(int)
df['last_name_match'] = (df['last_name_1'].str.lower() == df['last_name_2'].str.lower()).astype(int)

# 3. Features and target
features = ['score', 'birthdate_match', 'first_name_match', 'last_name_match']
X = df[features]
y = df['matched']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train random forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predict and evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Optional: Save model
import joblib
joblib.dump(model, 'rf_patient_match_model.pkl')
print("Model saved as rf_patient_match_model.pkl")
