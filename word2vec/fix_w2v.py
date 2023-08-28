import json
import numpy as np
from nltk.corpus import wordnet

def are_antonyms(word1, word2):
    # Get the antonyms of word1
    antonyms = []

    for syn in wordnet.synsets(word1):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())

    return word2 in antonyms

# Load the JSON data
with open('w2v_messy.json', 'r') as f:
    data = json.load(f)

w2v_values = {}

# Iterate over each target in the JSON data
for target, adj in data.items():
    dict = {}
    adjective_names = [v[0] for v in adj]
    values = 1.0 - np.array([v[1] for v in adj]) 
    for (a,v) in zip(adjective_names, values):
        if a in dict.keys():
            continue
        if are_antonyms(target, a):
            print(f"Antonym: {target} {a}")
            continue
        if not a.isascii():
            print(f"Non ascii: {target} {a}")
            continue
        if any([c in a for c in ['/', ':', '=', '(', ')', '[', ']', '{', '}', '<', '>', '@', ';', ',', '.', '?', '!', '#', '$', '%', '^', '&', '*', '_', '~', '`', '"', "'", '\\', '|', '+', '-']]):
            print(f"Invalid character: {target} {a}")
            continue
        dict[a] = v
    w2v_values[target] = dict


# Save the json data
with open('w2v.json', 'w') as f:
    json.dump(w2v_values, f, indent=4)

