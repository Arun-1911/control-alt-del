import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load dataset
df = pd.read_csv("dynamic_data.csv")

# Display column names to debug
print("Dataset Columns:", df.columns)

# Check for the correct feature column dynamically
possible_feature_columns = ["features", "text", "content", "data"]
feature_column = next((col for col in possible_feature_columns if col in df.columns), None)

if feature_column is None:
    raise ValueError("No valid feature column found in the dataset. Check the column names.")

# Define dataset class
class MalwareDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx][feature_column]  # Use dynamically found column
        label = self.data.iloc[idx]["label"] if "label" in self.data.columns else 0  # Default to 0 if no labels

        encoding = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Split dataset into train and eval
train_size = int(0.8 * len(df))
train_data, eval_data = df[:train_size], df[train_size:]

train_dataset = MalwareDataset(train_data, tokenizer)
eval_dataset = MalwareDataset(eval_data, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Fixed deprecated argument
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset  # Fixes missing eval_dataset error
)

# Start training
trainer.train()
