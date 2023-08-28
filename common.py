
import sys, mlflow, re
sys.path.append("../")
from common_rendering import *
import model, cv2
from data.VDB import VDB
from PIL import Image

class MatModel():

    def __init__(self, model_path):
        clip_model =  "ViT-B/32" #"ViT-L/14"
        self.vdb = VDB(clip_model)
        self.app = create_app()
        self.model = mlflow.pytorch.load_model(model_path).cuda().eval()
        self.kwargs = {
            "app":self.app, 
            "training_images":1, 
            "res": 512,
            "save_mdl": False,
        }
    
    def get_clip_score(self, text, img):
        if isinstance(text, str):
            text = self.vdb.emb_text(text)
        img = self.vdb.emb_image([Image.fromarray(im) for im in img])
        return (1.0 - torch.cosine_similarity(text, img, dim=-1)).tolist()

    def get_m(self, text):
        if isinstance(text, str):
            text = self.vdb.emb_text(text)
        if len(text.shape) == 1:
            text = text.unsqueeze(0)
        z = self.model.get_m(text)
        return z.squeeze()
    
    def get_v(self, text):
        if isinstance(text, str):
            text = self.vdb.emb_text(text)
        if len(text.shape) > 1:
            return [self.get_v(t) for t in text]
        return self.vdb.get_v(text)
    
    def get_image_from_m(self, m, v):
        return get_image(m, self.kwargs, v)[0]

    def get_image_from_text(self, text, greyscale=False):
        e_T = self.vdb.emb_text(text)
        m = self.get_m(e_T)
        if greyscale:
            v = [[0.6, 0.6, 0.6, 1, 1]]*m.shape[0]
        else:
            v = self.get_v(e_T)
        return get_image(m, self.kwargs, v)[0]
    
    def interpolate_tensors(self, t1, t2, a):
        return t1 * (1-a) + t2 * a
    
    def interpolate_v(self, t1, t2, a):
        if isinstance(t1, str):
            t1 = self.vdb.emb_text(t1)
        if isinstance(t2, str):
            t2 = self.vdb.emb_text(t2)
        v1 = self.vdb.get_v(t1)
        v2 = self.vdb.get_v(t2)
        v = [v1[i] * (1-a) + v2[i] * a for i in range(len(v1))]
        return v
    
    def save_imgs(self, path, imgs, name):
        for i, img in enumerate(imgs):
            image_array = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
            cv2.imwrite(path + f"{i/10}_{name}.png", image_array)

    def save_split_images(self, path, imgs_m, imgs_gt):
        assert len(imgs_m) == len(imgs_gt)
        for i in range(len(imgs_m)):
            img_m = cv2.cvtColor(imgs_m[i], cv2.COLOR_RGB2BGR) 
            img_gt = cv2.cvtColor(imgs_gt[i], cv2.COLOR_RGB2BGR) 

            image_array = img_m.copy()
            for ii in range(1024):
                for jj in range(ii, 1024):
                    if ii == jj:
                        image_array[ii][jj] = [0, 0, 0]
                    else:
                        image_array[ii][jj] = img_gt[ii][jj]

            cv2.imwrite(path + f"{i/10}_split.png", image_array)

    def get_mse(self, A, B):
        mse = []
        if len(B.shape) < len(A.shape):
            for i in range(len(A)):
                mse.append(np.square(np.subtract(A[i], B)).mean())
        else:
            for i in range(len(A)):
                mse.append(np.square(np.subtract(A[i], B[i])).mean())
        return mse

    def extract(self, names, mat_vecs, vs):

        # Create a dictionary to store the data
        data = {}

        for (name, m, v) in zip(names, mat_vecs, vs):
            # textures = [
            #     "    diffuse_texture",
            #     "    ORM_texture",
            #     "    normalmap_texture"
            # ]
            # noun = name.split(' ')[1]
            # base_mdl = f"omni_mdl/{noun}.mdl"
            # t_dict = []
            # with open(base_mdl, 'r') as f:
            #     text = f.read()
            #     for t in textures:
            #         line = re.search(r'{}:.*'.format(t), text)
            #         if line:
            #             line = line.group().strip()
            #             t_name = line.split(": ")[1].strip()
            #             t_name = t_name.replace("./", "./../omni_mdl/")
            #             t_dict.append(t_name)
            #         else:
            #             t_dict.append("texture_2d(),")

            # Store the data in the dictionary
            data[name] = {
                "diffuse_reflection_weight": float(m[0].item()),
                "diffuse_reflection_roughness": float(m[1].item()),
                "metalness": float(m[2].item()),
                "specular_reflection_weight": float(m[3].item()),
                "specular_reflection_roughness": float(m[4].item()),
                "specular_reflection_anisotropy": float(m[5].item()),
                "diffuse_reflection_color": [ float(v_) for v_ in v[0:3]],
                "specular_transmission_weight": float(1.0 - v[3]/2),
                "enable_specular_transmission": bool(v[3] < 0.5),
                "specular_reflection_ior": float(v[4]),
                # "diffuse_reflection_color_image": t_dict[0],
                # "ORM_texture": t_dict[1],
                # "geometry_normal_image": t_dict[2],
            }

        return data

    def create_mdls(self, data):

        # For each entry in the dictionary
        for name, values in data.items():
            # Open the template text file and read it into a string
            with open('example.mdl', 'r') as f:
                text = f.read()

            # Replace the values in the string with those from the dictionary
            for key, value in values.items():
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

            # Write the modified string to a new text file
            with open(f'gen_mdl/{name}.mdl', 'w') as f:
                f.write(text)


