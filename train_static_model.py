import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
file_path = "C:/Users/Arunk/Glitchcon/models/malware_dataset.csv"
df = pd.read_csv(file_path, low_memory=False)
print(f"✅ Dataset Loaded Successfully: {df.shape}")

# Drop non-numeric columns
df = df.drop(columns=['sha256', 'appeared'], errors='ignore')  # Remove sha256 and appeared columns

# Split features and labels
X = df.drop(columns=['label'])  # Features
y = df['label']  # Labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Training Set: {X_train.shape}, Testing Set: {X_test.shape}")

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model Training Completed!")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Save trained model
model_path = "C:/Users/Arunk/Glitchcon/models/static_malware_model.pkl"
joblib.dump(model, model_path)
print(f"✅ Model Saved at: {model_path}")
