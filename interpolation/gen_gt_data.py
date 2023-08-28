import sys
sys.path.append("../")
from common import MatModel
import torch
from PIL import Image
from tqdm import tqdm
import lpips
import cv2

if __name__ == "__main__":

    model_path = "../logs/mlflow/0/60da066d07b64732b2d5af5911608de6/artifacts/model"
    M = MatModel(model_path)

    N = 1
    interpolate_N = 10
    for _ in tqdm(range(N)):

        gt1 = torch.rand(6)
        gt2 = torch.rand(6)
        v1 = torch.rand(5)
        v2 = torch.rand(5)

        v = [M.interpolate_tensors(v1, v2, a/(interpolate_N-1)) for a in range(interpolate_N)]

        gt1_img = M.get_image_from_m(gt1.unsqueeze(0), v1.unsqueeze(0))
        #M.save_imgs(".", gt1_img, "gt1")
        gt1_img = M.vdb.emb_image([Image.fromarray(gt1_img[0])])
        gt2_img = M.get_image_from_m(gt2.unsqueeze(0), v2.unsqueeze(0))
        #M.save_imgs(".", gt2_img, "gt2")
        gt2_img = M.vdb.emb_image([Image.fromarray(gt2_img[0])])

        et1, t1 = M.vdb.annotate_mat(gt1_img)
        et2, t2 = M.vdb.annotate_mat(gt2_img)

        gt = torch.stack([M.interpolate_tensors(gt1, gt2, a/(interpolate_N-1)) for a in range(interpolate_N)], dim=0)
        imgs_gt = M.get_image_from_m(gt, v)
        #M.save_imgs(".", imgs_gt, "gt")

        m1 = M.get_m(et1)
        m2 = M.get_m(et2)

        m = torch.stack([M.interpolate_tensors(m1, m2, a/(interpolate_N-1)) for a in range(interpolate_N)], dim=0)
        imgs_m = M.get_image_from_m(m, v)
        #M.save_imgs(".", imgs_m, "m")

        lpips_1 = []
        lpips_2_A = []
        lpips_2_B = []
        for i in range(interpolate_N):
            lpips_1.append(M.vdb.get_lpips(imgs_gt[i], imgs_m[i]))
            lpips_2_A.append(M.vdb.get_lpips(imgs_gt[0], imgs_m[i]))
            lpips_2_B.append(M.vdb.get_lpips(imgs_gt[-1], imgs_m[i]))

        with open("interpolate_gt.csv", "a") as f:
            line = f"\"{lpips_1}\",\"{lpips_2_A}\",\"{lpips_2_B}\"\n"
            line = line.replace("[","")
            line = line.replace("]","")
            f.write(line)
