from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request


class OffensiveLanguageDetector:
    def __init__(self):
        task = "offensive"
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

        self.labels = []
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode("utf-8").split("\n")
            csvreader = csv.reader(html, delimiter="\t")
        self.labels = [row[1] for row in csvreader if len(row) > 1]

        # PT
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # Preprocess text (username and link placeholders)
    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)

    def detect(self, text):
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors="pt")
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        results = []
        for i in range(scores.shape[0]):
            l = self.labels[ranking[i]]
            s = scores[ranking[i]]
            results.append((l, np.round(float(s), 4)))

        return dict(results)


detector = OffensiveLanguageDetector()
results = detector.detect("Shutup")
print(results)
# # Tasks:
# # emoji, emotion, hate, irony, offensive, sentiment
# # stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

# task = "offensive"
# MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

# tokenizer = AutoTokenizer.from_pretrained(MODEL)

# # download label mapping
# labels = []
# mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
# with urllib.request.urlopen(mapping_link) as f:
#     html = f.read().decode("utf-8").split("\n")
#     csvreader = csv.reader(html, delimiter="\t")
# labels = [row[1] for row in csvreader if len(row) > 1]

# # PT
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)

# text = "Good night 😊"
# text = preprocess(text)
# encoded_input = tokenizer(text, return_tensors="pt")
# output = model(**encoded_input)
# scores = output[0][0].detach().numpy()
# scores = softmax(scores)

# # # TF
# # model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
# # model.save_pretrained(MODEL)

# # text = "Good night 😊"
# # encoded_input = tokenizer(text, return_tensors='tf')
# # output = model(encoded_input)
# # scores = output[0][0].numpy()
# # scores = softmax(scores)

# ranking = np.argsort(scores)
# ranking = ranking[::-1]
# for i in range(scores.shape[0]):
#     l = labels[ranking[i]]
#     s = scores[ranking[i]]
#     print(f"{i+1}) {l} {np.round(float(s), 4)}")
