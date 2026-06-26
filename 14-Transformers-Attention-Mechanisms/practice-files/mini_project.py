from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import evaluate
import torch
import os

# Set environment variable to avoid potential conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Load a smaller dataset for summarization
dataset = load_dataset("knkarthick/samsum")
#print(dataset["train"][0])


# For translation
#dataset = load_dataset("wmt14","en-fr")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Tokenize for summarization
def tokenize_function(examples):
    inputs = ["summarize: "+ doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True,padding="max_length")
    
    # Tokenize the targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=150, truncation=True, padding="max_length")
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

training_args = TrainingArguments(
    output_dir="./results_mini_project",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

trainer.train()

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

sample_text = "Artificial intelligence (AI) has become an integral part of modern society, transforming industries such as healthcare, finance, education, and transportation. In healthcare, AI-powered systems assist doctors in diagnosing diseases more accurately and quickly by analyzing medical images and patient data. In finance, machine learning algorithms help detect fraudulent transactions, assess credit risk, and automate investment strategies. Educational institutions use AI to personalize learning experiences, recommend study materials, and provide instant feedback to students. Self-driving cars and intelligent traffic management systems demonstrate AI's impact on transportation by improving road safety and reducing congestion. Despite these benefits, AI also raises concerns about data privacy, job displacement due to automation, and ethical decision-making. Governments, researchers, and technology companies are working together to develop regulations and responsible AI practices that ensure transparency, fairness, and accountability. As AI continues to evolve, it is expected to create new opportunities for innovation while requiring society to address its challenges carefully."
inputs = tokenizer("summarize: "+ sample_text, return_tensors='pt', max_length=512, truncation=True, padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_length=150,num_beams=4,early_stopping=True)

print("\nGenerated Summary:\n ",tokenizer.decode(outputs[0],skip_special_tokens=True))

# Load metric
# metric = evaluate.load("rouge")
# predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
# references = [tokenizer.decode(r, skip_special_tokens=True) for r in tokenized_datasets["validation"]["summary"]]

# results = metric.compute(predictions=predictions, references=references)
# print(results)