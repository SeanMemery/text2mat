import warnings

from flair.models import MultiTagger
from flair.data import Sentence
from flair.models import SequenceTagger


warnings.filterwarnings("ignore")


class FlairChunker():
    def __init__(self):
        self.chunker = SequenceTagger.load('chunk')

    def get_chunk_spans(self, s):
        sentence = Sentence(s)
        self.chunker.predict(sentence)
        spans = sentence.get_spans('np')
        return spans


def process_description(description):
    tags, spans = [], []
    flairchunker = FlairChunker()
    # load tagger for POS and NER
    spans = flairchunker.get_chunk_spans(description)
    tagger = MultiTagger.load(["pos", "ner"])

    for token in description.split():
        sentence = Sentence(token)
        tagger.predict(sentence)

        label_pos = sentence.get_label("pos")
        label_ner = sentence.get_label("ner")

        tags.append([token, label_pos.value, label_ner.value])
    return tags, spans


def get_pos_from_description(pos_labels):
    objs, actions = [], []
    for (token, label_pos, label_ner) in pos_labels:
        if 'NN' in label_pos:
            objs.append(token)
        elif 'VB' in label_pos:
            actions.append(token)
    return objs, actions
