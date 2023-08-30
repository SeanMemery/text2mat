import sys
import os


from flair.data import Sentence
from flair.models import SequenceTagger

import flair
import torch

flair.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load tagger
tagger = SequenceTagger.load("flair/pos-english")

# make example sentence
sentence = Sentence("A red ball")

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER spans
print("The following NER tags are found:")
# iterate over entities and print
for entity in sentence.get_spans("pos"):
    print(entity)


# load tagger
tagger = SequenceTagger.load("flair/ner-english-fast")

# make example sentence
sentence = Sentence("George Washington went to Washington")

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER spans
print("The following NER tags are found:")
# iterate over entities and print
for entity in sentence.get_spans("ner"):
    print(entity)
