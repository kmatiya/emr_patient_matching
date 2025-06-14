import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# --- 1. Load Data ---
df = pd.read_csv("matched_patients_rapid_fuzz.csv")

# --- 2. Encode Target Variable ---
df['matched'] = df['matched'].map({'yes': 1, 'no': 0})

# --- 3. Feature Engineering ---
def string_similarity(a, b):
    return 1 if str(a).strip().lower() == str(b).strip().lower() else 0

df['first_name_match'] = df.apply(lambda x: string_similarity(x['first_name_1'], x['first_name_2']), axis=1)
df['last_name_match'] = df.apply(lambda x: string_similarity(x['last_name_1'], x['last_name_2']), axis=1)
df['birthdate_match'] = df.apply(lambda x: string_similarity(x['birthdate_1'], x['birthdate_2']), axis=1)

# --- 4. Feature Selection ---
features = ['score', 'first_name_match', 'last_name_match', 'birthdate_match']
X = df[features]
y = df['matched']

# --- 5. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- 6. Train Logistic Regression Model ---
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)

# --- 7. Predict and Evaluate ---
y_pred = lr.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

# --- 8. Save Model ---
joblib.dump(lr, "lr_patient_match_model.pkl")
print("✅ Logistic Regression model saved as 'lr_patient_match_model.pkl'")
