import torch, argparse
import re
import os

import mlflow
import clip
import torch.nn as nn
import chromadb

from common_rendering import *
from generator import generate_file, covert_vec
from variables import PROJECT_PATH

k1 = 2
k2 = 2

persist_dir = f"{PROJECT_PATH}/src/ML/data/chroma_db/"
client = chromadb.Client(chromadb.config.Settings(chroma_db_impl="duckdb+parquet",
                                    persist_directory=persist_dir
                                ))
collection_nouns = client.get_collection("noun")
collection_adjs = client.get_collection("adj")
collection_cols = client.get_collection("col")
collection_cols_dict = collection_cols.get()
collection_transparency = client.get_collection("transparent")
collection_ior = client.get_collection("ior")

def mat_vec_to_dict(mat_vec, color, type="OmniPBR"):
    """
    Depending on the *type* of mdl file, we will fill in a dict
    format which will replace some part of the mdl file.
    """
    if type == "OmniPBR":
        return {
            "reflection_roughness_constant": float(mat_vec[1].item()),
            "metallic_constant": float(mat_vec[2].item()),
            "specular_level": float(mat_vec[4].item()),
        }
    elif type == "GlassWithVolume":
        return {
            "reflection_color": [float(v_) for v_ in color[0:3]],
            "ior": float(color[4]),
        }
    elif type == "OmniSurface":
        return {
            "diffuse_reflection_weight": float(mat_vec[0].item()),
            "diffuse_reflection_roughness": float(mat_vec[1].item()),
            "metalness": float(mat_vec[2].item()),
            "specular_reflection_weight": float(mat_vec[3].item()),
            "specular_reflection_roughness": float(mat_vec[4].item()),
            "specular_reflection_anisotropy": float(mat_vec[5].item()),
            "diffuse_reflection_color": [ float(v_) for v_ in color[0:3]],
            "specular_transmission_weight": float(1.0 - color[3]/2),
            "enable_specular_transmission": bool(color[3] < 0.5),
            "specular_reflection_ior": float(color[4]),
        }
    

def mat_vecs_to_dict(names, mat_vecs, vs):
    """
    Call 'mat_vec_to_dict' per each vectir of *names*, *mat_vecs* and *vs*.
    """
    data = {}
    for (name, m, v) in zip(names, mat_vecs, vs):
        data[name] = mat_vec_to_dict(m, v)
    return data

def create_mdl(base_file, name, data, output_dir):
    """
    Read template mdl *base_file*, replace it with the output of the model
    and output it at *output_dir*.
    """
    with open(f'../ML/comparisons/omni_mdl/{base_file}.mdl', 'r') as f:
        text = f.read()

    # Replace the values in the string with those from the dictionary
    for key, value in data.items():
        print(key, value)
        if value == "loc":
            continue
        if isinstance(value, list):
            value = ', '.join(map(str, value))
            value = f"color({value}),"
        elif isinstance(value, float):
            value = f"{value}f,"
        elif isinstance(value, bool):
            value = str(value).lower()
            value = f"{value},"
        text = re.sub(f'{key}:.*', f'{key}: {value}', text)

    # Write the modified string to a new mdl file
    with open(os.path.join(output_dir, f'{name}.mdl'), 'w') as f:
        f.write(text)

def get_v(prompt, clip_model):
    emb = clip_model.encode_text(clip.tokenize(prompt).to("cuda")).detach().cpu().numpy().tolist()

    ### Col
    c1 = 5
    q1 = collection_cols.query(
            query_embeddings=emb,
            n_results=c1
        )
    # softmax weights
    w1 = [ q1["distances"][0][i] for i in range(c1) ]
    w1 = torch.softmax(torch.Tensor(w1), dim=0)
    r = sum([ q1["metadatas"][0][i]["r"]*w1[i] for i in range(c1) ]) 
    g = sum([ q1["metadatas"][0][i]["g"]*w1[i] for i in range(c1) ]) 
    b = sum([ q1["metadatas"][0][i]["b"]*w1[i] for i in range(c1) ]) 

    ### Transparency
    c2 = 3
    q2 = collection_transparency.query(
            query_embeddings=emb,
            n_results=c2
        )
    w2 = [ q2["distances"][0][i] for i in range(c2) ]
    w2 = torch.softmax(torch.Tensor(w2), dim=0)
    t = sum([ q2["metadatas"][0][i]["t"]*w2[i] for i in range(c2) ]) 

    ### Ior
    c3 = 1
    q3 = collection_ior.query(
            query_embeddings=emb,
            n_results=c3
        )
    w3 = [ q3["distances"][0][i] for i in range(c3) ]
    w3 = torch.softmax(torch.Tensor(w3), dim=0)
    ior = sum([ q3["metadatas"][0][i]["ior"]*w3[i] for i in range(c3) ]) 

    return [r, g, b, t, ior]

def pbr_model_from_noun(noun):
    """Pick correct pbr model for *noun*."""
    if noun in ["crystal", "opal", "amethyst"]:
        return "GlassWithVolume"
    return "OmniPBR" 

def generate_material_from_text(prompt, base_file, output_dir):
    """
    Generate mdl material from *prompt* based on *base_file* and 
    output ir at *output_dir*.
    """
    model_path = f"../scene_gen/saved_model/bafd47d4256e42b18c3b4db569ce84bf/artifacts/model"

    print(f"Loading model {model_path}")
    model = mlflow.pytorch.load_model(model_path)
    model.to("cuda")
    model.eval()

    clip_model, _ = clip.load("ViT-B/32", device="cuda")
    clip_model.eval()

    with torch.no_grad():
        text_inputs = "A gray sphere with a surface material of " + prompt
        text_inputs = clip.tokenize([f"{text_inputs}"]).to("cuda")
        text_inputs = clip_model.encode_text(text_inputs).float()

        mat_vec = model.encode(text_inputs).detach().cpu().numpy().squeeze()
        v = get_v(prompt, clip_model)
        print("Material vector: ", mat_vec)
        print("V vector: ", v)
        data = mat_vec_to_dict(mat_vec, v, type=pbr_model_from_noun(base_file))
        create_mdl(base_file, prompt, data, output_dir)
