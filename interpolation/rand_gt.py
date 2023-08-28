import sys
sys.path.append("../")
from common import MatModel
import torch
from PIL import Image

if __name__ == "__main__":

    model_path = "../logs/mlflow/0/8f3f0a01bbcb43719919b6125fb845b1/artifacts/model"
    M = MatModel(model_path)

    while True:

        input("Press Enter to generate...")
        print("\n-----------------------------------")

        gt1 = torch.rand(6)
        gt2 = torch.rand(6)
        v1 = torch.rand(5)
        v2 = torch.rand(5)

        v = [M.interpolate_tensors(v1, v2, a/10) for a in range(11)]

        gt1_img = M.get_image_from_m(gt1.unsqueeze(0), v1.unsqueeze(0))
        gt1_img = M.vdb.emb_image([Image.fromarray(gt1_img[0])])
        gt2_img = M.get_image_from_m(gt2.unsqueeze(0), v2.unsqueeze(0))
        gt2_img = M.vdb.emb_image([Image.fromarray(gt2_img[0])])

        t1 = M.vdb.get_nearest_neighbors(gt1_img, "sentences", 1)["ids"][0][0]
        t2 = M.vdb.get_nearest_neighbors(gt2_img, "sentences", 1)["ids"][0][0]

        print("t1: ", t1)
        print("t2: ", t2)
        folder_name = f"({t1})_({t2})"

        gt = torch.stack([M.interpolate_tensors(gt1, gt2, a/10) for a in range(11)], dim=0)
        imgs_gt = M.get_image_from_m(gt, v)
        M.save_imgs(f"./{folder_name}/imgs/", imgs_gt, "gt")

        m1 = M.get_m(t1)
        m2 = M.get_m(t2)

        m = torch.stack([M.interpolate_tensors(m1, m2, a/10) for a in range(11)], dim=0)
        imgs_m = M.get_image_from_m(m, v)
        M.save_imgs(f"./{folder_name}/imgs/", imgs_m, "m")

        print("MSE: ", sum(M.get_mse(imgs_m, imgs_gt))/len(imgs_m))
        M.save_split_images(f"./{folder_name}/imgs/", imgs_m, imgs_gt)

        print("-----------------------------------")
