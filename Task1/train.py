
import torch
import accelerate
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments,DataCollatorForTokenClassification, BertTokenizer, BertForTokenClassification
from datasets import load_dataset, Dataset
import evaluate
import seqeval
#tokenization funcs
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    all_labels = examples["tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer,padding=True)
dataset = load_dataset("Gepe55o/mountain-ner-dataset")



for param in model.bert.encoder.layer[:10].parameters():
    param.requires_grad = False

example = dataset["train"][0]
label_list = ["0","1","-100"]
labels = [label_list[i] for i in example[f"tags"]]
seqeval = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names
)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,

)

# Обучение
trainer.train()
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")