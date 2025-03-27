import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification

# ✅ Load the dataset
file_path = "Glitchcon/models/dynamic_dataset.csv"  # Update if needed
df = pd.read_csv(file_path)

# ✅ Ensure dataset has the required columns
if "api_calls" not in df.columns or "label" not in df.columns:
    raise ValueError("Dataset must have 'api_calls' and 'label' columns!")

# ✅ Load pre-trained BERT model and tokenizer
model_path = "Glitchcon/models/pretrained_bert_model"  # Update if needed
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# ✅ Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Function to predict malware
def predict_malware(api_sequence):
    inputs = tokenizer(api_sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    return torch.argmax(logits, dim=1).item()

# ✅ Get true labels and predictions
true_labels = df["label"].tolist()
predicted_labels = df["api_calls"].apply(predict_malware).tolist()

# ✅ Compute accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"✅ Model Accuracy on New Data: {accuracy * 100:.2f}%")

# ✅ Save predictions (optional)
df["predicted_label"] = predicted_labels
df.to_csv("Glitchcon/models/dynamic_predictions.csv", index=False)
print("📁 Predictions saved to 'Glitchcon/models/dynamic_predictions.csv'")
