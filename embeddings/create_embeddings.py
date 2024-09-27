import os
import glob
import json
import numpy as np
import natsort
from os.path import exists
from sentence_transformers import SentenceTransformer

def main():

    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

    paths = natsort.natsorted(glob.glob('./outputs/**/*'))
    out = './embeddings'

    data = []
    for file_id, path in enumerate(paths):
        print(path)
        with open(path) as f:
            dirname, filename = path.split('/')[-2:]
            outpath = f'{out}/{dirname}'
            if not exists(outpath):
                os.mkdir(outpath)
            for article_id, line in enumerate(f.readlines()):
                embedding_path = f'{outpath}/{filename}_{article_id}.npy'
                if exists(embedding_path): continue
                item = json.loads(line)
                if item['text'] == '': continue
                embedding = model.encode(item['text'])
                np.save(embedding_path, embedding)

if __name__ == "__main__":
    main()