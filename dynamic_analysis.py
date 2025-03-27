import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load dataset
file_path = "C:/Users/Arunk/Glitchcon/models/dynamic_data.csv"
df = pd.read_csv(file_path)

# Verify the first few rows to find a valid text column
print(df.head())

# Identify the correct text column (assuming 'behavior_log' contains textual descriptions)
text_column = "behavior_log"  # Change this to the correct column name

# Ensure the column exists
if text_column not in df.columns:
    raise ValueError(f"Column '{text_column}' not found in dataset!")

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust labels if needed

# Preprocess text data
def preprocess_text(text):
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Example usage
sample_text = str(df.iloc[0][text_column])  # Convert to string to avoid errors
inputs = preprocess_text(sample_text)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

print("Predictions:", predictions)
