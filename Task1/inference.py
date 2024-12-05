from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("Levbut/Ner-mountain")
model = AutoModelForTokenClassification.from_pretrained("Levbut/Ner-mountain")
classifier = pipeline("ner", model=model, tokenizer=tokenizer)
text = "Next on our list is Denali Peak, also known as Mount McKinley, in Alaska."
result = classifier(text)

print(result)