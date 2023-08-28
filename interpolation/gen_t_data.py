import sys
sys.path.append("../")
from common import MatModel
import torch
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":

    model_path = "../logs/mlflow/0/60da066d07b64732b2d5af5911608de6/artifacts/model"
    M = MatModel(model_path)

    N = 100
    interpolate_N = 10
    for _ in tqdm(range(N)):

        e_T1, t1 = M.vdb.get_supervised_sentence()
        e_T2, t2 = M.vdb.get_supervised_sentence()

        v = [M.interpolate_v(e_T1, e_T2, a/(interpolate_N-1)) for a in range(interpolate_N)]

        e_T = torch.stack([M.interpolate_tensors(e_T1, e_T2, a/(interpolate_N-1)) for a in range(interpolate_N)], dim=0)
        e_Tm = M.get_m(e_T)
        imgs_e_T = M.get_image_from_m(e_Tm, v)

        clip_score1 = M.get_clip_score(e_T1, imgs_e_T)
        clip_score2 = M.get_clip_score(e_T2, imgs_e_T)

        with open("interpolate_t.csv", "a") as f:
            line = f"{t1},{t2},\"{clip_score1}\",\"{clip_score2}\"\n"
            line = line.replace("[","")
            line = line.replace("]","")
            f.write(line)
