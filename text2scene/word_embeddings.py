import spacy
import gensim
import json
import os
import sys

sys.path.append('../scene_gen/')

from gensim.models import Word2Vec
from gen_from_text import generate_material_from_text

nlp = spacy.load("en_core_web_lg")

nouns = [
  "brick",
  "aluminum",
  "brass",
  "bronze",
  "metal",
  "chrome",
  "concrete",
  "copper",
  "gold",
  "iron",
  "leather",
  "mahogany",
  "marble",
  "mirror",
  "oak",
  "plaster",
  "plastic",
  "silver",
  "steel",
  "terracotta",
  "sandstone",
  "amethyst",
  "clay",
  "crystal",
  "opal",
]

adjectives = [  
  "dull",
  "shiny",
  "smooth",
  "polished",
  "rough",
  "matte",
  "reflective",
  "worn",
  "cloudy",
  "metallic",
]

def get_neighbours_from_word_w2v(word, number_neighbours=None, model="../../data/bin/word2vec_embeddings-SNAPSHOT.model"):
    """
    Returns the list of (key/neighbour, similarity)
    Model: https://github.com/olivettigroup/materials-word-embeddings
    """
    model = Word2Vec.load(model)
    return model.wv.most_similar(positive=[word], topn=number_neighbours)

def get_neighbours_from_word_fastext(word, number_neighbours, model="../../data/bin/fasttext_embeddings-MINIFIED.model"):
    """
    Returns the list of (key/neighbour, similarity)
    Model: https://github.com/olivettigroup/materials-synthesis-generative-models
    """
    model = gensim.models.keyedvectors.KeyedVectors.load(model)
    model.bucket = 2000000
    return model.wv.most_similar(positive=[word], topn=number_neighbours)

def get_neighours_from_word_custom_model(word, number_neighbours, model):
    """
    Returns the list of (key/neighbour, similarity)
    Model chosen: https://huggingface.co/fse/glove-wiki-gigaword-300 (download through gensim package)
    """
    import gensim.downloader as api

    model = api.load(model)
    return model.wv.most_similar(positive=[word], topn=number_neighbours)

def get_and_filter_neighbour(number_neighbours, model):
    """
    Load the model and get neighbours for all adjectives, then filter neighbours that
    1. similar to the original word with slight variation (e.g. uppercase)
    2. are not adjectives
    """
    filtered_neighbours = {}
    for adjective in adjectives:
        neighbours = []
        if model == "w2v":
            neighbours = get_neighbours_from_word_w2v(adjective, number_neighbours)
        elif model == "fastext":
            neighbours = get_neighbours_from_word_fastext(adjective, number_neighbours)
        else: #custom
            neighbours = get_neighours_from_word_custom_model(adjective, number_neighbours, model)
        list_filtered_neighbours = []
        for (neighbour, score) in neighbours:
            doc = nlp(neighbour)
            pos = [token.pos_ for token in doc if token.pos_ == "ADJ" and token.text.lower() != adjective]
            if pos:
                list_filtered_neighbours.append((neighbour.lower(), score))
        filtered_neighbours[adjective] = list_filtered_neighbours
    return filtered_neighbours

def save_dict_as_json(dict_, name):
    json_string = json.dumps(dict_)
    with open(f"../../data/word_embedding/{name}.json", 'w') as outfile:
        outfile.write(json_string)

def get_and_save_all_word_embeddings(number_neighbours, models):
    """
    Calls 'get_and_filter_neighbour' for each model and saves the results as json at 
        '<root>/lang2mat/data/word_embedding/<model>'
    """
    for model in models:
        dict_word_neighbours = get_and_filter_neighbour(number_neighbours=number_neighbours, model=model)
        save_dict_as_json(dict_word_neighbours, model)

def load_json_file(name):
    with open(f'{name}.json', encoding= 'utf-8') as json_file:
        return json.load(json_file)

def load_word_embeddings(models):
    word_embeddings = {}
    for model in models:
        word_embeddings[model] = load_json_file(f'../../data/word_embedding/{model}')
    return word_embeddings

def produce_materials_name(models):
    """
    Create a dictionary with this structure
        {model: {noun: {adjective: neighbour_adj_and_noun}}} 
    """
    word_embeddings = load_word_embeddings(models)
    dict_name_materials = {}
    for model, v in word_embeddings.items():
        dict_name_materials[model] = {}
        for noun in nouns:
            dict_name_materials[model][noun] = {}
            for adjective, new_adjectives in v.items():
                list_name_materials = []
                for new_adjective in new_adjectives:
                    list_name_materials.append(f"{new_adjective[0]} {noun}")
                dict_name_materials[model][noun][adjective] = list_name_materials
    return dict_name_materials

def get_and_save_all_produced_materials_name(models):
    """
    Call 'produce_materials_name' and save the output as a json file 
        '<root>/lang2mat/data/word_embedding/all_produced_materials_name.json'
    """
    dict_produced_materials_name = produce_materials_name(models)
    save_dict_as_json(dict_produced_materials_name, "all_produced_materials_name")

def count_number_files_in_folder(folder):
    return len(os.listdir(folder))

def create_mdl_files(models, topn, force=False):
    """
    Create mdl files using the following folder structure
        '<root>/lang2mat/data/new_materials/<model>/<noun>/<adjective>/<file>.mdl'
    
    We only require *topn* mdl files but as sometimes we can find some neighbors that are
    not easy to write as files (they have a slash e.g. 'red/brown brass') then we skip them
    until we have *topn* files created. 
    
    The script takes a while to run but it can be stopped and rerun multiple times, the
    only problem that could raise is if you stop a mdl while it is writing it so you have
    a corrupted file that it will still be counted as an existing file. 
    """
    dict_produced_materials_name = load_json_file(f'../../data/word_embedding/all_produced_materials_name')
    import os
    cwd = '../../data/new_materials'
    for model in models:
        os.makedirs(os.path.join(cwd, model), exist_ok=True)
        for noun in nouns:
            os.makedirs(os.path.join(cwd, model, noun), exist_ok=True)
            for adjective in adjectives:
                current_cwd = os.path.join(cwd, model, noun, adjective)
                os.makedirs(current_cwd, exist_ok=True)
                for material_name in dict_produced_materials_name[model][noun][adjective]:
                    current_count = count_number_files_in_folder(current_cwd)
                    if (not os.path.exists(os.path.join(current_cwd, f"{material_name}.mdl")) or force) and current_count < topn:
                        try:
                            print('started...' + material_name)
                            generate_material_from_text(material_name, noun, output_dir=current_cwd)
                            print('finished...' + material_name)
                        except OSError as e:
                            print(f"There might be a problem with the name of the file '{material_name}', i.e. it contains slash or non utf8 characters")
                    else:
                        continue

# k = 2000
models = ["w2v", "fastext", "glove-wiki-gigaword-300"]
# get_and_save_all_produced_materials_name(models)
create_mdl_files(models, 25)
