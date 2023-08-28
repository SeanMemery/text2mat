import sys
sys.path.append("../")
from common import MatModel
from tqdm import tqdm
import json
import numpy as np

# 25
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
    "emerald",
    "ruby",
    "sapphire",
    "plaster",
    "sandpaper",
    "carpet",
    "blood",
    "crystal",
    "oak",
    "tarmac",
    "amethyst",
    "coal",
    "paint",
    "stainless steel",
    "pavement",
]
# 10
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

if __name__ == "__main__":

    model_path = "../logs/mlflow/0/60da066d07b64732b2d5af5911608de6/artifacts/model"
    M = MatModel(model_path)
    grey = True

    # Load data 
    with open("./w2v.json", "r") as f:
        data = json.load(f)

    N_neighbours = 20 

    full_json = {}
    for noun in tqdm(nouns):
        n_json = {}
        for adj in adjectives:
            neighbours = list(data[adj].keys())[:N_neighbours]
            prompts = [f"{adj} {noun}"] + [f"{n} {noun}" for n in neighbours]
            imgs = M.get_image_from_text(prompts, greyscale=grey)
            #M.save_imgs(f"./imgs/{adj}_{noun}", imgs, "grey")
            n_json[f"{adj} {noun}"] = {prompts[n+1] : M.vdb.get_lpips(imgs[0], imgs[n+1]) for n in range(N_neighbours) }
        full_json[noun] = n_json 

    with open(f"./lpips.json", "w") as f:
        json.dump(full_json, f, indent=4)



        