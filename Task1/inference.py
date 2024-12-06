from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("Levbut/Ner-mountain")
model = AutoModelForTokenClassification.from_pretrained("Levbut/Ner-mountain")
classifier = pipeline("ner", model=model, tokenizer=tokenizer)
text = "Next on our list is Denali Peak, also known as Mount McKinley, in Alaska."
result = classifier(text)

print(result)
#output: 
#[{'entity': 'LABEL_0', 'score': 0.9994753, 'index': 1, 'word': 'next', 'start': 0, 'end': 4}, {'entity': 'LABEL_0', 'score': 0.99919146, 'index': 2, 'word': 'on', 'start': 5, 'end': 7}, {'entity': 'LABEL_0', 'score': 0.9972264, 'index': 3, 'word': 'our', 'start': 8, 'end': 11}, {'entity': 'LABEL_0', 'score': 0.99471766, 'index': 4, 'word': 'list', 'start': 12, 'end': 16}, {'entity': 'LABEL_0', 'score': 0.997797, 'index': 5, 'word': 'is', 'start': 17, 'end': 19}, {'entity': 'LABEL_1', 'score': 0.9113941, 'index': 6, 'word': 'den', 'start': 20, 'end': 23}, {'entity': 'LABEL_2', 'score': 0.9320071, 'index': 7, 'word': '##ali', 'start': 23, 'end': 26}, {'entity': 'LABEL_1', 'score': 0.9052254, 'index': 8, 'word': 'peak', 'start': 27, 'end': 31}, {'entity': 'LABEL_2', 'score': 0.8642386, 'index': 9, 'word': ',', 'start': 31, 'end': 32}, {'entity': 'LABEL_0', 'score': 0.99865806, 'index': 10, 'word': 'also', 'start': 33, 'end': 37}, {'entity': 'LABEL_0', 'score': 0.99890435, 'index': 11, 'word': 'known', 'start': 38, 'end': 43}, {'entity': 'LABEL_0', 'score': 0.9978671, 'index': 12, 'word': 'as', 'start': 44, 'end': 46}, {'entity': 'LABEL_1', 'score': 0.6557637, 'index': 13, 'word': 'mount', 'start': 47, 'end': 52}, {'entity': 'LABEL_1', 'score': 0.624543, 'index': 14, 'word': 'mckinley', 'start': 53, 'end': 61}, {'entity': 'LABEL_2', 'score': 0.61280185, 'index': 15, 'word': ',', 'start': 61, 'end': 62}, {'entity': 'LABEL_0', 'score': 0.99973553, 'index': 16, 'word': 'in', 'start': 63, 'end': 65}, {'entity': 'LABEL_0', 'score': 0.9639652, 'index': 17, 'word': 'alaska', 'start': 66, 'end': 72}, {'entity': 'LABEL_0', 'score': 0.97836703, 'index': 18, 'word': '.', 'start': 72, 'end': 73}]
