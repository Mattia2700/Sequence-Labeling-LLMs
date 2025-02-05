from datasets import Dataset, DatasetDict, Features, Sequence, Value
import os
from huggingface_hub import login
from dotenv import dotenv_values
import random

def read_conll_file(filepath):
    sentences = []
    tokens = []
    tags = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:  # end of a sentence
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens = []
                    tags = []
            else:
                # assuming token and tag are separated by a tab character
                parts = line.split("\t")
                if len(parts) == 2:
                    token, tag = parts
                    tokens.append(token)
                    tags.append(tag)
                else:
                    # if the line does not split as expected, you can handle it here
                    continue
        # in case file does not end with an empty line:
        if tokens:
            sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences

def dump_conll_file(filepath, sentences, name):
    with open(f"{filepath}/{name}.tsv", "w", encoding="utf-8") as f:
        for sentence in sentences:
            for token, tag in zip(sentence["tokens"], sentence["ner_tags"]):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")

train = os.path.join(os.path.dirname(__file__), "dataset/italian/train.tsv")
data = read_conll_file(train)
print(len(data))

test = os.path.join(os.path.dirname(__file__), "dataset/italian/test.tsv")
data = read_conll_file(test)
print(len(data))

test = Dataset.from_dict({
    "tokens": [item["tokens"] for item in data],
    "ner_tags": [item["ner_tags"] for item in data]
})

random.seed(42)
idxs = list(range(len(test)))
idxs_val = random.sample(idxs, int(len(idxs) * 0.2))
idxs_test = [idx for idx in idxs if idx not in idxs_val]

val = test.select(idxs_val)
test = test.select(idxs_test)

dump_conll_file(os.path.join(os.path.dirname(__file__), "dataset/italian"), val, "val")
dump_conll_file(os.path.join(os.path.dirname(__file__), "dataset/italian"), test, "test_new")