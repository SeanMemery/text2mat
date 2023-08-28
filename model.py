### Torch
import torch
from torch import nn, Tensor
from torch.nn import functional as F

### Common
from common_rendering import *
import mlflow

### Other
import numpy as np
import random, os, cv2
from PIL import Image
from typing import List, Any, Dict

### ------------------ Models ------------------ ###

class BaseAE(nn.Module):
    def __init__(self) -> None:
        super(BaseAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError
    
    def get_m(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
class AutoEncoder(BaseAE):
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__()
        in_channels = kwargs["channels"][0]
        out_channels = kwargs["channels"][1]
        modules = []
        act = kwargs["act"]
        z_act = kwargs["z_act"]
        final_act = kwargs["final_act"]
        d_rate = kwargs["d_rate"]
        # Build Encoder
        size = kwargs["max_size"] - kwargs["min_size"]
        min_dim = 32
        hidden_dims = [int(in_channels - (i * (in_channels - min_dim) / (size - 1))) for i in range(size)]
        hidden_dims.append(kwargs["latent_dim"])

        ### Ordering of layers
        # 1. Linear
        # 2. BatchNorm
        # 3. Activation
        # 4. Dropout
        ### 

        # Build Encoder
        for _ in range(kwargs["l_mult"]):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[0], hidden_dims[0]),
                    nn.BatchNorm1d(hidden_dims[0]),
                    act(),
                    nn.Dropout(d_rate),
                )
            )
        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    act(),
                    nn.Dropout(d_rate),
                )
            )
        self.encoder = nn.Sequential(*modules)
        self.z_layer = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            z_act(),
        ) 
        
        # Build Decoder
        modules = []
        hidden_dims.reverse()
        hidden_dims[0] = hidden_dims[0] 
        if kwargs["l_mult"]>0:
            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                        nn.BatchNorm1d(hidden_dims[i + 1]),
                        act(),
                        nn.Dropout(d_rate),
                    )
                )
            for _ in range(kwargs["l_mult"]-1):
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                        nn.BatchNorm1d(hidden_dims[-1]),
                        act(),
                        nn.Dropout(d_rate),
                    )
                )
            modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                        nn.BatchNorm1d(hidden_dims[-1]),
                        final_act()
                    )
                )
            self.decoder = nn.Sequential(*modules)
        else:
            for i in range(len(hidden_dims) - 2):
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                        nn.BatchNorm1d(hidden_dims[i + 1]),
                        act(),
                        nn.Dropout(d_rate),
                    )
                )
            modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[-2], hidden_dims[-1]),
                        nn.BatchNorm1d(hidden_dims[-1]),
                        final_act()
                    )
                )
            self.decoder = nn.Sequential(*modules)
        
    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input)

    def decode(self, input: Tensor) -> Tensor:
        d = self.decoder(input)
        return d

    def forward(self, input: Tensor) -> List[Tensor]:
        z = self.encode(input)
        z = self.z_layer(z)
        d = self.decode(z)
        return [d, z]
    
    def get_m(self, x: Tensor) -> Tensor:
        return self.z_layer(self.encoder(x))

    def generate(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]
    
class VariationalAutoEncoder(BaseAE):
    def __init__(self, **kwargs):
        super(VariationalAutoEncoder, self).__init__()
        in_channels = kwargs["channels"][0]
        out_channels = kwargs["channels"][1]
        modules = []
        act = kwargs["act"]
        z_act = kwargs["z_act"]
        final_act = kwargs["final_act"]
        d_rate = kwargs["d_rate"]
        size = kwargs["max_size"] - kwargs["min_size"]
        min_dim = 32
        hidden_dims = [int(in_channels - (i * (in_channels - min_dim) / (size - 1))) for i in range(size)]

        # Build Encoder
        modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[0]+1, hidden_dims[0]),
                    nn.BatchNorm1d(hidden_dims[0]),
                    act(),
                    nn.Dropout(d_rate),
                )
            )
        for _ in range(kwargs["l_mult"]):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[0], hidden_dims[0]),
                    nn.BatchNorm1d(hidden_dims[0]),
                    act(),
                    nn.Dropout(d_rate),
                )
            )
        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    act(),
                    nn.Dropout(d_rate),
                )
            )
        modules.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-2], hidden_dims[-1]),
                nn.BatchNorm1d(hidden_dims[-1]),
                z_act(),
            )
        )
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1], kwargs["latent_dim"])
        self.fc_var = nn.Linear(hidden_dims[-1], kwargs["latent_dim"])
        self.z_act = z_act()

        # Build Decoder
        modules = []
        hidden_dims.append(kwargs["latent_dim"])
        hidden_dims.reverse()
        hidden_dims[0] = hidden_dims[0] + 1
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    act(),
                    nn.Dropout(d_rate),
                )
            )
        for _ in range(kwargs["l_mult"]):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                    nn.BatchNorm1d(hidden_dims[-1]),
                    act(),
                    nn.Dropout(d_rate),
                )
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                            nn.BatchNorm1d(hidden_dims[-1]),
                            final_act()
                            )
        
    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input)

    def decode(self, input: Tensor) -> Tensor:
        d = self.decoder(input)
        d = self.final_layer(d)
        return d

    def forward(self, input: Tensor, w: Tensor, **kwargs) -> List[Tensor]:
        input_w = torch.zeros((input.shape[0], 1 + input.shape[1]), device="cuda")
        input_w[:, :input.shape[1]] = input
        input_w[:, -1] = w.unsqueeze(0)
        result = self.encode(input_w)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        z = self.z_act(z)

        z_w = torch.zeros((z.shape[0], 1 + z.shape[1]), device="cuda")
        z_w[:, :z.shape[1]] = z
        z_w[:, -1] = w.unsqueeze(0)
        
        return  [self.decode(z_w), z, mu, log_var]

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

class MultiLayer(BaseAE):
    def __init__(self, **kwargs):
        super(MultiLayer, self).__init__()
        in_channels = kwargs["channels"][0]
        out_channels = kwargs["latent_dim"]
        act = kwargs["act"]
        z_act = kwargs["z_act"]
        final_act = kwargs["final_act"]
        d_rate = kwargs["d_rate"]
        modules = []

        ### Calc dims 
        dims = [int(in_channels - (i * (in_channels - 8) / (kwargs["size"] - 1))) for i in range(kwargs["size"])]

        ### Encoder Layers
        for i in range(len(dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]),
                    nn.Dropout(d_rate),
                    act()
                )
            )

        ### Final Layer
        modules.append(
            nn.Sequential(
                nn.Linear(dims[-1], out_channels),
                nn.Dropout(d_rate),
                final_act())
        )

        self.encoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input)

    def decode(self, input: Tensor) -> Tensor:
        return input

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(input)
        d = self.decode(z)
        return [d, z]

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]   

models = {
    "AE": AutoEncoder,
    "VAE": VariationalAutoEncoder,
    "bVAE": VariationalAutoEncoder,
    "ML": MultiLayer,
}

### ------------------ General Model ------------------ ###

class ModularModel(BaseAE):
    def __init__(self, **kwargs) -> None:
        super(ModularModel, self).__init__()

        ### Vector Database (set in run.py)
        self.vdb = None

        ### Model Params
        self.model_name = kwargs["model"]
        self.training_images = kwargs["training_images"]
        self.save_imgs = kwargs["exp_params"]["save_imgs"]
        self.res = kwargs["res"]
        self.batch_size = kwargs["data_params"]["train_batch_size"]
        self.latent_dim = kwargs["latent_dim"]
        self.gen_col = kwargs["gen_col"]
        self.rand_supervised = kwargs["exp_params"]["rand_supervised"]

        ### CLIP score normalization
        self.clip_max = 0.5
        self.clip_min = 0

        ### Avg rec and target w
        self.total_rec_w = 0.0
        self.total_target_w = 0.0
        self.total_ratio = 0
        self.steps = 0

        ### Create model
        self.model = models[self.model_name](
            channels=(kwargs["clip_dim"], kwargs["clip_dim"]),
            latent_dim=kwargs["latent_dim"],
            l_mult=kwargs["l_mult"],
            min_size=kwargs["min_size"],
            max_size=kwargs["max_size"],
            z_act=self.get_act_fn(kwargs["latent_act_fn"]),
            final_act=self.get_act_fn(kwargs["final_act_fn"]),
            act=self.get_act_fn(kwargs["act_fn"]),
            d_rate=kwargs["d_rate"],
        )

        ### Image Masking
        radius = int((self.res // 2) * 0.93)
        self.img_mask = np.zeros((self.res, self.res, 4))
        self.img_mask[:, :, 3] = 1
        for i in range(self.res):
            for j in range(self.res):
                if (i-self.res//2)**2 + (j-self.res//2)**2 < radius**2:
                    self.img_mask[i,j] = (1,1,1,1)
        self.img_mask = self.img_mask.astype(np.uint8)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['vdb']
        return state

    def z_layer(self, x):
        return self.model.z_layer(x)

    def get_act_fn(self, act_fn):
        if act_fn=="sigmoid":
            return nn.Sigmoid
        elif act_fn=="tanh":
            return nn.Tanh
        elif act_fn=="relu":
            return nn.ReLU
        elif act_fn=="leaky_relu":
            return nn.LeakyReLU
        elif act_fn=="softplus":
            return nn.Softplus
        elif act_fn=="elu":
            return nn.ELU
        elif act_fn=="selu":
            return nn.SELU
        elif act_fn=="celu":
            return nn.CELU
        elif act_fn=="gelu":
            return nn.GELU
        elif act_fn=="none":
            return nn.Identity
        else:
            print(f"Activation function {act_fn} not implemented")
            raise NotImplementedError
    
    def save_images(self, imgs, w, kwargs: Dict):
        e = kwargs["epoch"]
        save_name = kwargs["save_name"]
        epoch_dir = f"./logs/val_imgs/epoch_{e}/"
        os.makedirs(epoch_dir, exist_ok=True)
        for j in range(0, self.training_images):
            dir = epoch_dir + f"imgs_{j}/"
            os.makedirs(dir, exist_ok=True)  
            for i in range(0, len(imgs[j])):
                image_array = cv2.cvtColor(imgs[j][i] * self.img_mask, cv2.COLOR_RGB2BGR) 
                cv2.imwrite(dir + f"{i:03d}_{save_name}_w({float(w[i]):03f}).png", image_array)
        
    def gen_random_batch(self, app=None, validation=False):
        e_T = []
        T = []

        # If supervised batch
        m, e_I = None, None
        I = []
        w = []
        if self.rand_supervised > 0 and (validation or torch.rand(1) < self.rand_supervised):
            m = torch.rand((self.batch_size, 6), device="cuda")
            col = torch.rand((self.batch_size, 3), device="cpu")
            kwargs = {
                "app": app, 
                "training_images": self.training_images, 
                "res": 224,
                "save_mdl": False,
                "validation": validation,
                "save_imgs": True,
                "save_name": "target",
            }
            e_I, I = self.get_clip_features(m, kwargs, col)
            k = 3
            r_weights = [0.3, 0.55, 0.15]
            for emb in e_I:
                n = torch.rand(1).item()
                if n < r_weights[0]:
                    q = self.vdb.get_nearest_neighbors(emb, "sentences_0adj", k)
                    ind = torch.randint(0, k, (1,)).item()
                    t = q["ids"][0][ind]                
                elif n < r_weights[0] + r_weights[1]:
                    q = self.vdb.get_nearest_neighbors(emb, "sentences_1adj", k)
                    ind = torch.randint(0, k, (1,)).item()
                    t = q["ids"][0][ind]
                else:
                    q = self.vdb.get_nearest_neighbors(emb, "sentences_2adj", k)
                    ind = torch.randint(0, k, (1,)).item()
                    t = q["ids"][0][ind]
                T.append(t)
                ww = (1 - q["distances"][0][ind] - self.clip_min) / (self.clip_max - self.clip_min)
                w.append(ww)
                e_T.append(self.vdb.emb_text(t))
            w = torch.tensor(w, device="cuda")
            e_T = torch.stack(e_T, dim=0)
        else:
            for _ in range(self.batch_size):
                text_emb, text = self.vdb.get_sentence()
                e_T.append(text_emb)
                T.append(text)
            e_T = torch.stack(e_T, dim=0)
            col = [self.vdb.get_col(text_emb) for text_emb in e_T]
        self.steps += self.batch_size

        return e_T, col, T, m, e_I, I, w

    def get_clip_features(self, mat_vecs, kwargs: Dict, v=None):
        imgs = get_image(mat_vecs, kwargs, v)

        image_features = []
        final_imgs = []
        for j in range(0, self.training_images):
            final_imgs.append([Image.fromarray(img * self.img_mask, "RGBA") for img in imgs[j] ])
            image_features.append(self.vdb.emb_image(final_imgs[j]))

        ### Average features across all images
        if self.training_images > 1:
            image_features = torch.stack(image_features, dim=1).mean(dim=1)
        else:
            image_features = image_features[0]

        return image_features, final_imgs

    def encode(self, input: Tensor) -> Tensor:
        return self.model.encode(input) 

    def decode(self, z: Tensor) -> Tensor:
        return self.model.decode(z)
    
    def get_m(self, x: Tensor) -> Tensor:
        return self.model.get_m(x)

    def forward(self, input: Tensor) -> List[Tensor]:
        return self.model.forward(input)
    
    def get_cosine_sim(self, input, target):
        target = target / target.norm(dim=-1, keepdim=True)
        input = input / input.norm(dim=-1, keepdim=True)
        c_sims = torch.zeros(input.shape[0])
        for i in range(target.shape[0]):
            c_sims[i] = input[i].float() @ target[i].float()
        c_sims = (c_sims - self.clip_min) / (self.clip_max - self.clip_min)
        w = 1-c_sims.mean()
        
        ### Calculate average
        self.total_rec_w += c_sims.sum()

        return w

    def get_cosine_sim_norm(self, input, target, w):
        w = w.detach().clone()
        input = input.detach().clone()
        target = target / target.norm(dim=-1, keepdim=True)
        input = input / input.norm(dim=-1, keepdim=True)
        c_sims = torch.zeros(input.shape[0], device="cuda")
        for i in range(target.shape[0]):
            c_sims[i] = input[i].float() @ target[i].float()
        c_sims = (c_sims - self.clip_min) / (self.clip_max - self.clip_min)

        ### Calculate average
        self.total_rec_w += c_sims.sum()
        self.total_ratio += torch.div(c_sims, w).sum()

        return 1-(c_sims - w).mean()   
    
    def loss_function(self, *args, **kwargs) -> dict:
        loss_dict = {}
        loss_config = kwargs["loss_config"]

        clip_kwargs = {
            "app":kwargs["app"], 
            "validation":kwargs["validation"], 
            "training_images":self.training_images, 
            "res": self.res,
            "save_imgs": self.save_imgs,
            "epoch": kwargs["epoch"],
            "save_mdl": False,
            "save_name": "rec",
        }

        ### Get features
        e_T, c, T, m, e_I, I, w = kwargs["batch"]
        if self.model_name == "AE":
            d_e_T, d_m = args
        elif self.model_name == "VAE":
            d_e_T, d_m, mu, log_var = args
            loss_dict[f"KLD"] = loss_config["kld_weight"] * torch.mean(-0.5 * torch.torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        elif self.model_name == "bVAE":
            d_e_T, d_m, mu, log_var = args
            loss_dict[f"KLD"] = loss_config["beta"] * loss_config["kld_weight"] * torch.mean(-0.5 * torch.torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        ### Render step 
        with torch.no_grad():
            d_e_I, d_I = self.get_clip_features(d_m, clip_kwargs, c)

## L1 ###### 
        if loss_config["loss_weights"][0] > 0:
            loss_dict[f"loss_dT_T"] = loss_config["loss_weights"][0] * F.l1_loss(d_e_T, e_T)
        ################################################################################################

## L2 #####  
        if m != None and loss_config["loss_weights"][1] > 0:
            loss_dict[f"loss_dm_m"] = loss_config["loss_weights"][1] * F.l1_loss(d_m, m)
        ################################################################################################

## L3 #####  
        if len(I) > 0 and loss_config["loss_weights"][2] > 0:
            loss_dict[f"loss_I_dI"] = loss_config["loss_weights"][2] * self.vdb.get_lpips(I, d_I)
        ################################################################################################

## L4 ##### 
        if loss_config["loss_weights"][3] > 0:
            if w != None:
                loss_dict["loss_dI_T"] = loss_config["loss_weights"][3] * self.get_cosine_sim_norm(d_e_I, e_T, w)
            else:
                loss_dict["loss_dI_T"] = loss_config["loss_weights"][3] * self.get_cosine_sim(d_e_I, e_T)
        ################################################################################################

        ### Save Imgs
        if kwargs["validation"]: 
            e = kwargs["epoch"]

            if self.save_imgs:
                ### Get weight
                target = d_e_I / d_e_I.norm(dim=-1, keepdim=True)
                input = e_T / e_T.norm(dim=-1, keepdim=True)
                d_w = torch.zeros(input.shape[0], device="cuda")
                for i in range(target.shape[0]):
                    d_w[i] = input[i].float() @ target[i].float()  
                d_w = (d_w - self.clip_min + 0.00001) / (self.clip_max - self.clip_min + 0.00001)

                self.save_images(I, w, {
                    "epoch": kwargs["epoch"],
                    "save_name": "target",
                })
                self.save_images(d_I, d_w, {
                    "epoch": kwargs["epoch"],
                    "save_name": "rec",
                })

                ### Log
                with open(f"./logs/val_imgs/epoch_{e}/losses.txt", "a") as f:
                    f.write(f"Average Rec w: {self.total_rec_w/self.steps:03f} \n")
                    #f.write(f"Average Ratio: {self.total_ratio/self.steps:03f} | Average Rec w: {self.total_rec_w/self.steps:03f} | Average Target w: {self.total_target_w/self.steps:03f} \n")
                    for i in range(len(d_w)):
                        f.write(f"Img: {i:03d} | Rec w: {d_w[i].item():03f} | Target w: {w[i]:03f} | Col: {c[i]} | Text: {T[i]} \n") 
                        f.write(f"Input Mat Vector: {m[i].detach().cpu().numpy()} \n")
                        f.write(f"Predicted Mat Vector: {d_m[i].detach().cpu().numpy()} \n")
                        f.write(f"------------------------------------------------------------------------------ \n")

            mlflow.log_metric("avg_w", self.total_rec_w/self.steps, step=e)

            #self.total_target_w = 0
            self.total_rec_w = 0
            #self.total_ratio = 0
            self.steps = 0

        loss_dict["loss"] = sum(loss_dict.values())
        return loss_dict
