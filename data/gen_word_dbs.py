from tqdm import tqdm
import matplotlib.colors as colors
import os, torch
import textwrap
from itertools import chain, combinations

from VDB import VDB


clip_version =  "ViT-B/32" #"ViT-L/14"
vdb = VDB(clip_version, reset=True)

### Sentences
with torch.no_grad():
    with open("supervised_nouns.txt", "r") as f:
        nouns = f.read().lower().splitlines()

    with open("supervised_adjectives.txt", "r") as f:
        adjectives = f.read().lower().splitlines()
    
    max_len = 2
    adj_powerset = list(chain.from_iterable(combinations(adjectives, r) for r in range(max_len+1)))
    print(f"Number of sentences: {len(nouns)*len(adj_powerset)}")

    for n in tqdm(nouns):
        for adj in adj_powerset:
            if len(adj) == 0:
                text = n
            else:
                text = ", ".join(adj) + " " + n
            emb = vdb.emb_text(text)
            metadata = {"noun": n, "adjectives": str(adj)}
            if len(adj) == 0:
                vdb.add("sentences_0adj", emb, id=text, metadata=metadata)
            elif len(adj) == 1:
                vdb.add("sentences_1adj", emb, id=text, metadata=metadata)
            elif len(adj) == 2:
                vdb.add("sentences_2adj", emb, id=text, metadata=metadata)

### Cols
with torch.no_grad():
    with open("colors.csv", "r") as f:
        csv = f.read().splitlines()
        for c in tqdm(csv):
            c = c.split(',')
            c_name = c[0].lower().replace("'","")
            c_hex = c[1]
            if len(c_hex) < 1:
                continue
            c_rgb = colors.to_rgb(c_hex)
            embedding = vdb.emb_text(c_name)
            metadatas = {"r": c_rgb[0], "g": c_rgb[1], "b": c_rgb[2]}
            vdb.add("col", embedding, id=c_name, metadata=metadatas)


### Transparency
with torch.no_grad():
    t_words = [
            [
                "Opaque",
                "Solid",
                "Obscure",
                "Non-transparent",
                "Impenetrable",
                "Thick",
                "Dense",
                "metal",
                "stone",
                "rock",
                "wood",
                "marble",
                "polished",
                "rough",
                "plastic",
                "reflective",
            ], # 1.0
            [
                "Opalescent",
                "Milky",
                "Iridescent",
                "Pearlescent",
                "skin",
                "blurry",
                "dark water"
            ], # 0.85
            [
                "Partly opaque",
                "Partly solid",
                "Murky",
                "Tinted",
                "gem",
                "jewel",
            ], # 0.65
            [
                "Ethereal",
                "Misty",
                "Hazy",
                "Cloudy",
                "Foggy",
                "Smoky",  
                "Smokey",
                "water"
            ], # 0.45
            [
                "Semi-transparent",
                "Crystalline",
                "Faintly visible",
                "Barely perceptible",
                "Almost clear",
                "crystal",
                "clear water"
            ], # 0.25
            [
                "Clear",
                "See-through",
                "Translucent",
                "Glass",
                "Crystal-clear",
                "Invisible",
            ], # 0.1
    ]
    for i, t in tqdm(enumerate([1.0, 0.85, 0.75, 0.55, 0.25, 0.1]), total=len(t_words)):
        for n in t_words[i]:
            embedding = vdb.emb_text(n.lower())
            vdb.add("transparent", embedding, id=n, metadata={"t": t})

### IOR
with torch.no_grad():
    ior_words = {
        "air": 1.0,
        "water": 1.33,
        "glass": 1.5,
        "diamond": 2.42,
        "amber": 1.55,
        "amethyst": 1.54,
        "aquamarine": 1.57,
        "beryl": 1.57,
        "calcite": 1.49,
        "citrine": 1.55,
        "corundum": 1.77,
        "emerald": 1.57,
        "fluorite": 1.43,
        "garnet": 1.73,
        "opal": 1.45,
        "peridot": 1.65,
        "quartz": 1.55,
        "ruby": 1.77,
        "sapphire": 1.77,
        "spinel": 1.72,
        "topaz": 1.62,
        "tourmaline": 1.62,
        "turquoise": 1.61,
        "zircon": 1.93,
        "silicon": 3.42,
    }
    for r in tqdm(ior_words.keys()):
        embedding = vdb.emb_text(r.lower())
        vdb.add("ior", embedding, id=r, metadata={"ior": ior_words[r]})
        

### Save Collections
vdb.persist()
