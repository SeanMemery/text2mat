import sys
sys.path.append("../")
from common import MatModel
from tqdm import tqdm
import time

if __name__ == "__main__":

    model_path = "../logs/mlflow/0/60da066d07b64732b2d5af5911608de6/artifacts/model"
    M = MatModel(model_path)

    text = "The quick brown fox jumps over the lazy dog"
    text = M.vdb.emb_text(text)

    N = 1000
    avg_time = 0
    for _ in tqdm(range(N)):
        t1 = time.time()
        M.vdb.annotate_mat(text)
        avg_time += time.time()-t1
    avg_time /= N
    print(f"avg_time: {avg_time}")
        