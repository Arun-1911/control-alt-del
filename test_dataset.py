import pandas as pd

# Update the correct dataset filename if needed
dataset_path = "MALWARE ANALYSIS DATASETS_API IMPORT.csv"

try:
    df = pd.read_csv(dataset_path)
    print("✅ Dataset Loaded Successfully!")
    print(df.head())  # Display the first few rows
except FileNotFoundError:
    print(f"❌ ERROR: File not found at {dataset_path}")
except Exception as e:
    print(f"❌ ERROR: {e}")
