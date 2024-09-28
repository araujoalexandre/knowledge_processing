import os
import glob
import json
import numpy as np
import natsort
import time
from os.path import exists
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import idr_torch
import torch.distributed as dist
import torch
from tqdm import tqdm

class ArticleDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        dirname, filename = path.split('/')[-2:]
        with open(path) as f:
            articles = f.readlines()
        return dirname, filename, articles

def collate_fn(batch):
    """
    Custom collate function to handle variable-length articles (lines per file).
    """
    batch_output = {"dirname": [], "filename": [], "articles": []}

    for dirname, filename, articles in batch:
        batch_output["dirname"].append(dirname)
        batch_output["filename"].append(filename)
        batch_output["articles"].append(articles)

    return batch_output["dirname"], batch_output["filename"], batch_output["articles"]

def process_batch(batch, model, out, rank):
    dirname, filename, articles = batch
    for dname, fname, article_lines in zip(dirname, filename, articles):
        outpath = f'{out}/{dname}'
        if not exists(outpath):
            os.makedirs(outpath, exist_ok=True)
        
        texts = []
        save_paths = []
        
        for article_id, line in enumerate(article_lines):
            item = json.loads(line)
            if item['text'] == '':
                continue  # Skip empty texts
            embedding_path = f'{outpath}/{fname}_{article_id}.npy'
            if exists(embedding_path):
                continue  # Skip if the embedding already exists
            texts.append(item['text'])
            save_paths.append(embedding_path)
        
        if texts:
            embeddings = model.encode(texts, device=f'cuda:{rank}', batch_size=len(texts))
            for emb, save_path in zip(embeddings, save_paths):
                np.save(save_path, emb)

def main():
    dist.init_process_group(backend='nccl', init_method='env://', world_size=idr_torch.size, rank=idr_torch.rank)
    
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1").cuda(idr_torch.rank)
    
    paths = natsort.natsorted(glob.glob('./outputs/**/*'))
    out = './embeddings'

    dataset = ArticleDataset(paths)
    
    sampler = DistributedSampler(dataset, num_replicas=idr_torch.size, rank=idr_torch.rank, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=8, collate_fn=collate_fn)  # Custom collate_fn
    
    total_batches = len(dataloader)
    
    if idr_torch.rank == 0:
        print(f'Processing {len(paths)} files...')

    # Tracking memory and progress
    if idr_torch.rank == 0:
        progress_bar = tqdm(total=total_batches, desc="Processing")

    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        process_batch(batch, model, out, idr_torch.rank)
        
        # GPU Memory Tracking
        allocated_memory = torch.cuda.memory_allocated(idr_torch.rank) / (1024 ** 2)  # in MB
        reserved_memory = torch.cuda.memory_reserved(idr_torch.rank) / (1024 ** 2)   # in MB
        
        if idr_torch.rank == 0:
            elapsed_time = time.time() - start_time
            batches_left = total_batches - (batch_idx + 1)
            time_per_batch = elapsed_time / (batch_idx + 1)
            estimated_time_left = batches_left * time_per_batch
            progress_bar.update(1)
            progress_bar.set_postfix({
                "Allocated_Mem_MB": f"{allocated_memory:.2f}",
                "Reserved_Mem_MB": f"{reserved_memory:.2f}",
                "Elapsed_Time_Sec": f"{elapsed_time:.2f}",
                "Est_Time_Left_Sec": f"{estimated_time_left:.2f}"
            })

    dist.barrier()  # Ensure all processes finish before exiting

    if idr_torch.rank == 0:
        progress_bar.close()

if __name__ == "__main__":
    main()

