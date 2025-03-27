import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer

# Load dataset
file_path = "C:/Users/Arunk/Glitchcon/bodmas_malware_category.csv"  # Correct path
df = pd.read_csv(file_path)

# Encode category labels into numbers
df["category"] = df["category"].astype("category").cat.codes

# Convert SHA256 hashes into numerical features
vectorizer = HashingVectorizer(n_features=32, alternate_sign=False)
X = vectorizer.fit_transform(df["sha256"]).toarray()

# Labels
y = df["category"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data
pd.DataFrame(X_train).to_csv("C:/Users/Arunk/Glitchcon/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("C:/Users/Arunk/Glitchcon/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("C:/Users/Arunk/Glitchcon/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("C:/Users/Arunk/Glitchcon/y_test.csv", index=False)

print("✅ Data preprocessing complete! Training data saved.")
