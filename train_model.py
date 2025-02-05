import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch

# Load dataset
df = pd.read_csv('data/tweet_emotions.csv')

# Preprocess data
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
inputs = tokenizer(df['text'].tolist(), return_tensors='pt', max_length=512, truncation=True, padding='max_length')

# Split data
train_inputs, val_inputs = train_test_split(inputs['input_ids'], test_size=0.1)

# Create dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

train_dataset = TextDataset(train_inputs)
val_dataset = TextDataset(val_inputs)

# Define model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save model
model.save_pretrained('./custom_model')
tokenizer.save_pretrained('./custom_model')
