import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import defaultdict
import json, os
from tqdm import tqdm

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load concretness ratings
with open('concreteness_ratings.json', 'r') as f:
    concreteness_ratings = json.load(f)

# Load the data from the text file
text = ""
for book in os.listdir("books"):
    with open(os.path.join("books", book), "r") as f:
        text += f.read()

# Tokenize the text
words = word_tokenize(text)
print("Number of words:", len(words))

# Perform POS tagging
tagged_words = pos_tag(words)
print("Number of tagged words:", len(tagged_words))

# A list to hold the noun-adjective pairs
noun_adj_pairs = []

# Initialize an empty dictionary to hold current noun and adjectives describing it
current_noun = defaultdict(list)

# Loop through the tagged words
for i in tqdm(range(len(tagged_words) - 1, -1, -1)):  # we go in reverse order
    word, tag = tagged_words[i]

    # If short
    if len(word) < 3:
        continue

    # Replace periods
    word = word.replace(".", "")

    # Continue if it contains a non-ASCII character
    if any(ord(character) > 127 for character in word):
        continue

    # If it's a noun, save it and its adjectives if any
    if tag in ["NN"]:  # these are tags for singular and plural common/proper nouns
        if word in concreteness_ratings:
            if concreteness_ratings[word]["Conc.M"] < 4:
                noun_adj_pairs.append(current_noun.copy())
                current_noun.clear()
                continue
        else:
            continue
        if word not in current_noun and current_noun:  # if we find a new noun, save the previous one with its adjectives
            noun_adj_pairs.append(current_noun.copy())
            current_noun.clear()
        current_noun[word]
    elif tag in ["JJ", "JJR", "JJS"] and current_noun:  # these are tags for adjectives
        for noun in current_noun:
            current_noun[noun].append(word)  # associate the adjective with the last found noun
            
if current_noun:  # save the last found noun and its adjectives if any
    noun_adj_pairs.append(current_noun)

# Create a list for the noun-adjective pairs
noun_adj_list = []
for pair in noun_adj_pairs:
    for noun, adjs in pair.items():
        noun_adj_list.append({"noun": noun, "adjectives": adjs})

# Write the list to a JSON file
with open('noun_adj_pairs.json', 'w') as f:
    json.dump(noun_adj_list, f, indent=4)