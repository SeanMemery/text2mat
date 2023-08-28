import sys
sys.path.append("../")
from common import MatModel
import parser as pp
import shutil, os

if __name__ == "__main__":

    model_path = "../saved_model/bafd47d4256e42b18c3b4db569ce84bf/artifacts/model"
    M = MatModel(model_path)

    while True:
        p = input("Paragraph: ")

        # Remove old images in ./gen_mdl
        shutil.rmtree("./gen_mdl")
        os.makedirs("./gen_mdl")

        concreteness_val = 3.8
        parser = pp.Parser(concreteness_val)
        names = parser.parse_paragraph(p)
        print("Material Names: ", names)

        mat_vecs = []
        vs = []
        for name in names:
            mat_vecs.append(M.get_m(name).squeeze())
            vs.append(M.get_v(name))

        data = M.extract(names, mat_vecs, vs)
        M.create_mdls(data)