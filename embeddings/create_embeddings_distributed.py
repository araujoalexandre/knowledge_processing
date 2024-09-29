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
import argparse

class ArticleDataset(Dataset):
    """ Dataset class to load articles from files. 
        Each txt file contains multiple articles in JSON format.
    """
    def __init__(self, paths):
        self.paths = paths
        self.number_articles = 0
        self.list_of_indices = []
        for path in paths:
            with open(path) as f:
                lenght = len(f.readlines())
                self.number_articles += lenght
                self.list_of_indices.append(lenght)
        print(f'Number of articles: {self.number_articles}')
        print(f'Number of files: {len(paths)}')
        print(self.list_of_indices)
        self.list_of_indices = np.cumsum(self.list_of_indices)
        print(self.list_of_indices)
    
    def __len__(self):
        return self.number_articles
    
    def __getitem__(self, idx):
        idx_file = np.searchsorted(self.list_of_indices, idx, side='right')
        if idx_file == 0:
            idx_in_file = idx
        else:
            idx_in_file = idx - self.list_of_indices[idx_file - 1]
        with open(self.paths[idx_file]) as f:
            lines = f.readlines()
            line = json.loads(lines[idx_in_file]) 
            return line["id"],  self.paths[idx_file].split('/')[-1], line["text"]

        
def process_batch(batch, model, out, rank):
    articles_id, filename, articles = batch
    embeddings = model.encode(articles, device=f'cuda:{rank}', batch_size=len(articles))
    for article_id, fname, embedding in zip(articles_id, filename, embeddings):
        outpath = f'{out}/{fname}'
        if not exists(outpath):
            os.makedirs(outpath, exist_ok=True)
        embedding_path = f'{outpath}/{fname}_{article_id}.npy'   
        np.save(embedding_path, embedding)

def main(path_to_files='files', out='embeddings', batch_size=32):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=idr_torch.size, rank=idr_torch.rank)
    
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1").cuda(idr_torch.rank)
    
    paths = natsort.natsorted(glob.glob(f'{path_to_files}/*', recursive=True))
    print(paths)
    if not exists(out):
            os.makedirs(out, exist_ok=True)

    dataset = ArticleDataset(paths)
    
    sampler = DistributedSampler(dataset, num_replicas=idr_torch.size, rank=idr_torch.rank, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)  # Custom collate_fn
    
    total_batches = len(dataloader)
    
    if idr_torch.rank == 0:
        print(f'Processing {len(paths)} files...')

    # Tracking memory and progress
    if idr_torch.rank == 0:
        progress_bar = tqdm(total=total_batches, desc="Processing")

    
    for batch_idx, batch in enumerate(dataloader):
        process_batch(batch, model, out, idr_torch.rank)
        
        # GPU Memory Tracking
        allocated_memory = torch.cuda.memory_allocated(idr_torch.rank) / (1024 ** 2)  # in MB
        reserved_memory = torch.cuda.memory_reserved(idr_torch.rank) / (1024 ** 2)   # in MB
        
        if idr_torch.rank == 0:
            progress_bar.update(1)
            progress_bar.set_postfix({
                "Allocated_Mem_MB": f"{allocated_memory:.2f}",
                "Reserved_Mem_MB": f"{reserved_memory:.2f}",
            })

    dist.barrier()  # Ensure all processes finish before exiting

    if idr_torch.rank == 0:
        progress_bar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='create_embeddings_distributed.py',
                    description='Create embeddings for articles in parallel',
                    epilog='Enjoy the program! :)',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path_to_files', type=str, default='files')
    parser.add_argument('--out', type=str, default='embeddings')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    main(path_to_files=args.path_to_files, out=args.out, batch_size=args.batch_size)

