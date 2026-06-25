from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("ag_news")

# Load RoBERTa tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base",num_labels=4)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"],truncation=True,padding="max_length",max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare datasets 
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label","labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Training arguments
training_args = TrainingArguments(
    output_dir="./results2",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500
)

# Trainer
trainer = Trainer(model=model,args=training_args,train_dataset=train_dataset,eval_dataset=test_dataset,tokenizer=tokenizer)

# train model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results: \n",results)



# #GPT3
# from openai import OpenAI
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # Set OpenAI API key
# client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

# try:
#     # Generate text using GPT-3.5 Turbo
#     response = client.chat.completions.create(
#         model = "gpt-3.5-turbo",
#         messages = [
#             {"role":"system","content":"You are a helpful assistant."},
#             {"role":"user","content":"Write a short story about a robot learning to cook."}
#         ],
#         max_tokens=150,
#         temperature=0.7
#     )
    
#     print("Generated Text:\n",response["choices"][0]["message"]["content"].strip())
    
    
# except Exception as e:
#     print(f"An error occured: {e}")