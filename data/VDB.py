import chromadb, clip, torch, json, random, lpips, cv2
import numpy as np
from variables import PROJECT_PATH

class VDB():
    def __init__(self, clip_version, reset=False):
        persist_dir = f"{PROJECT_PATH}/src/scene_gen/data/chroma_db/"
        self.client = chromadb.Client(chromadb.config.Settings(chroma_db_impl="duckdb+parquet",
                                            persist_directory=persist_dir
                                        ))
        if reset:
            self.client.reset()

        self.clip_model, self.clip_preprocess = clip.load(clip_version, device="cuda") 
        self.clip_model.eval()

        self.collection_map = {
            "sentences_0adj": self.client.get_or_create_collection("sentences_0adj", metadata={"hnsw:space": "cosine"}),
            "sentences_1adj": self.client.get_or_create_collection("sentences_1adj", metadata={"hnsw:space": "cosine"}),
            "sentences_2adj": self.client.get_or_create_collection("sentences_2adj", metadata={"hnsw:space": "cosine"}),
            "col": self.client.get_or_create_collection("col", metadata={"hnsw:space": "cosine"}),
            "transparent": self.client.get_or_create_collection("transparent", metadata={"hnsw:space": "cosine"}),
            "ior": self.client.get_or_create_collection("ior", metadata={"hnsw:space": "cosine"})
        }
        self.collection_cols_dict = self.collection_map["col"].get()
        #self.collection_sentence_dict = self.collection_map["sentences"].get()
        self.collection_sentence_dict_0 = self.collection_map["sentences_0adj"].get()
        self.collection_sentence_dict_1 = self.collection_map["sentences_1adj"].get()
        self.collection_sentence_dict_2 = self.collection_map["sentences_2adj"].get()

        self.json_data = json.load(open(f"{PROJECT_PATH}/src/scene_gen/data/text/noun_adj_pairs.json"))
        self.json_len = len(self.json_data)

        self.lpips_loss = lpips.LPIPS(net='alex')

    def emb_text(self, p):
        with torch.no_grad():
            return self.clip_model.encode_text(clip.tokenize(p).to("cuda")).float().squeeze()
    
    def emb_image(self, img):
        with torch.no_grad():
            image_array = [self.clip_preprocess(img).unsqueeze(0) for img in img ]           
            image_array = torch.cat(image_array, dim=0).to("cuda")
            return self.clip_model.encode_image(image_array)
    
    def get_nearest_neighbors(self, emb, collection_name, k):
        return self.collection_map[collection_name].query(
            query_embeddings=emb.detach().cpu().numpy().tolist(),
            n_results=k
        )
    
    def get_lpips(self, im1, im2):
        if isinstance(im1, list):
            assert len(im1) == len(im2), f"im1:{len(im1)} and im2:{len(im2)} must be the same length"
            return sum([self.get_lpips(im1[i], im2[i]) for i in range(len(im1))]) / len(im1)
        else:
            im1 = lpips.im2tensor(cv2.cvtColor(np.array(im1), cv2.COLOR_BGR2RGB)[:,:,::-1])
            im2 = lpips.im2tensor(cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2RGB)[:,:,::-1])
            return float(self.lpips_loss(im1, im2).item())
    
    def persist(self):
        self.client.persist()
    
    def add(self, collection_name, embedding, id, metadata):
        try:
            self.collection_map[collection_name].add(
                embeddings=embedding.detach().cpu().numpy().squeeze().tolist(),
                ids=id,
                metadatas=metadata
            )
        except Exception as e:
            if "already exist" in str(e):
                return
            print(e) 
            print(f"Error adding {id} to {collection_name}")

    def rand_from(self, dict):
        return random.choice(list(dict.keys()))

    def get_supervised_sentence(self):
        ### Random element from self.collection_sentence_dict
        r_weights = [0.3, 0.55, 0.15]
        n = torch.rand(1).item()
        if n < r_weights[0]:
            t = self.collection_sentence_dict_0[self.rand_from(self.collection_sentence_dict_0)]             
        elif n < r_weights[0] + r_weights[1]:
            t = self.collection_sentence_dict_1[self.rand_from(self.collection_sentence_dict_1)]
        else:
            t = self.collection_sentence_dict_2[self.rand_from(self.collection_sentence_dict_2)]
        emb = self.emb_text(t)
        return emb, t
    
    def annotate_mat(self, img_emb):
        if isinstance(img_emb, list):
            return [self.annotate_mat(im) for im in img_emb]
        else:
            q0 = self.get_nearest_neighbors(img_emb, "sentences_0adj", 1)
            q1 = self.get_nearest_neighbors(img_emb, "sentences_1adj", 1)
            q2 = self.get_nearest_neighbors(img_emb, "sentences_2adj", 1)
            q = q0 if q0["distances"][0] < q1["distances"][0] else q1
            q = q if q["distances"][0] < q2["distances"][0] else q2
            n, a =  q["metadatas"][0][0]["noun"], q["metadatas"][0][0]["adjectives"]
            a = a[2:-2].replace("'", "").split(", ")
            text = ", ".join(a) + " " + n
            return self.emb_text(text), text

    def get_sentence(self):
        # Get random element of json
        i = torch.randint(0, self.json_len, (1,)).item()
        json_noun = self.json_data[i]["noun"]
        adj_len = len(self.json_data[i]["adjectives"])
        text = json_noun
        j = torch.randint(0, adj_len + 1, (1,)).item()
        if j > 0:
            indices = set(torch.randint(0, adj_len, (j,)).tolist())
            json_adj = [self.json_data[i]["adjectives"][ind] for ind in indices]
            text = ", ".join(json_adj) + " " + json_noun
        emb = self.emb_text(text)
        return emb, text
    
    def get_col(self, emb):
        q = self.get_nearest_neighbors(emb, "col", 1)
        return [q["metadatas"][0][0]["r"], q["metadatas"][0][0]["g"], q["metadatas"][0][0]["b"]]

    def get_v(self, emb):
        ### Col
        c1 = 5
        q1 = self.get_nearest_neighbors(emb, "col", c1)
        # softmax weights
        w1 = [ q1["distances"][0][i] for i in range(c1) ]
        w1 = torch.softmax(torch.Tensor(w1), dim=0)
        r = sum([ q1["metadatas"][0][i]["r"]*w1[i] for i in range(c1) ]) 
        g = sum([ q1["metadatas"][0][i]["g"]*w1[i] for i in range(c1) ]) 
        b = sum([ q1["metadatas"][0][i]["b"]*w1[i] for i in range(c1) ]) 

        ### Transparency
        c2 = 3
        q2 = self.get_nearest_neighbors(emb, "transparent", c2)
        w2 = [ q2["distances"][0][i] for i in range(c2) ]
        w2 = torch.softmax(torch.Tensor(w2), dim=0)
        t = sum([ q2["metadatas"][0][i]["t"]*w2[i] for i in range(c2) ]) 

        ### Ior
        c3 = 1
        q3 = self.get_nearest_neighbors(emb, "ior", c3)
        w3 = [ q3["distances"][0][i] for i in range(c3) ]
        w3 = torch.softmax(torch.Tensor(w3), dim=0)
        ior = sum([ q3["metadatas"][0][i]["ior"]*w3[i] for i in range(c3) ]) 

        return [r, g, b, t, ior]