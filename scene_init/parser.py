import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import defaultdict
import json, os
from tqdm import tqdm

class Parser():
    def __init__(self, concreteness_val):

        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        
        # Load concretness ratings
        with open('../data/text/concreteness_ratings.json', 'r') as f:
            self.concreteness_ratings = json.load(f)

        self.concreteness_val = concreteness_val

    def parse_paragraph(self, text):
        words = word_tokenize(text)

        # Perform POS tagging
        tagged_words = pos_tag(words)

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
                if word in self.concreteness_ratings:
                    if self.concreteness_ratings[word]["Conc.M"] < self.concreteness_val:
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

        # Get list of concretness ratings
        concreteness_ratings_list = []
        for pair in noun_adj_list:
            noun = pair["noun"]
            if noun in self.concreteness_ratings:
                concreteness_ratings_list.append(self.concreteness_ratings[noun]["Conc.M"])
            else:
                concreteness_ratings_list.append(0)

        # Sort the noun-adjective pairs by concreteness
        noun_adj_list = [x for _, x in sorted(zip(concreteness_ratings_list, noun_adj_list), key=lambda pair: pair[0])]
        noun_adj_list.reverse()

        # Print the noun-adjective pairs
        output_text = []
        for pair in noun_adj_list:
            text = pair["noun"]
            adjs = pair["adjectives"]
            if len(adjs) > 0:
                for i, adj in enumerate(adjs):
                    if i == 0:
                        text = adj + " " + text
                    else:
                        text = adj + ", " + text
            output_text.append(text)
        return output_text
        #return output_text[:int(len(output_text)/1.5)]
    

if __name__ == "__main__":
    p = Parser()
    text = "\
        The castle, an architect's grand patchwork quilt, seamlessly incorporated a diverse array of materials. \
        Its towering walls were composed of grey-veined marble from distant quarries, solid and imposing, bearing an ancient sense of history. \
        The ornate gate was forged from wrought iron, its intricate designs the handiwork of a master smith, hailing from the northern blacksmithing clans. \
        The turret roofs were lined with copper shingles, weathered to a rich verdigris hue, glinting under the sunshine with an almost mythical allure. \
        Along the castle walls, mosaics of vibrant stained glass windows narrated tales of yore, their colors a dazzling spectacle under the glow of the morning light. \
        Ivory bone statues, artifacts from age-old battles, stood as sentinels, perched upon battlements of the castle that were cloaked in the emerald velvet of moss, hinting at the marriage of man-made grandeur and the gentle touch of nature."
    o = p.parse_paragraph(text)
    print(o)