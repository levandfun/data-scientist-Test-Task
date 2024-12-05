from sklearn.metrics import precision_score, recall_score, f1_score
from datasets import load_dataset, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import evaluate
import seqeval
import numpy as np
def compute_metrics(classifier,dataset):
    test = dataset
    right=0
    wrong=0
    for i in range(len(test)):

        print(i)
        # parsing through entire dataset in this way will be too long
        if(i>100):
            return right / (right + wrong)
        res=classifier(datasets["test"]["tokens"][i])

        for j in range(len(res)):
            h=0
            for k in range(len(res[j])):

                if (res[j][k]['entity'] == 'LABEL_0' and test[i]["labels"][h] <= 0) or (
                        (res[j][k]['entity'] == 'LABEL_1' or res[j][k]['entity'] == 'LABEL_2') and test[i]["labels"][
                    h] == 1):
                    right += 1
                else:
                    wrong += 1
                h+=1


    return right/(right+wrong)

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


model_path = r"/trained_model"
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

classifier = pipeline("ner", model=model, tokenizer=tokenizer)
datasets=load_dataset("Gepe55o/mountain-ner-dataset")

tokenized_dataset = datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=datasets["train"].column_names
)
precision= compute_metrics(classifier, tokenized_dataset["test"])
print(precision)