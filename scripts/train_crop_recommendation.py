# scripts/train_crop_recommendation.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# === Load Data ===
df = pd.read_csv("/data/Crop_recommendation2.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Encode crop labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define multiple models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", C=1.0, probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "LogReg": LogisticRegression(max_iter=500)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.3f}")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print("\nBest Model:", best_model_name)
print(classification_report(y_test, best_model.predict(X_test_scaled), target_names=le.classes_))

# Save artifacts
joblib.dump(best_model, "../ml_artifacts/crop_recommendation_model.pkl")
joblib.dump(le, "../ml_artifacts/crop_label_encoder.pkl")
joblib.dump(scaler, "../ml_artifacts/crop_scaler.pkl")

print("Artifacts saved successfully!")
