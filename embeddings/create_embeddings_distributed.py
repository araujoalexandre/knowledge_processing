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
                self.number_articles += len(f.readlines())
                self.list_of_indices.append(len(f.readlines()))
        print(f'Number of articles: {self.number_articles}')
        print(f'Number of files: {len(paths)}')
        self.list_of_indices = np.cumsum(self.list_of_indices)      
    
    def __len__(self):
        return len(self.number_articles)
    
    def __getitem__(self, idx):
        idx_file = np.searchsorted(self.list_of_indices, idx, side='right')
        if idx_file == 0:
            idx_in_file = idx
        else:
            idx_in_file = idx - self.list_of_indices[idx_file - 1]
        with open(self.paths[idx_file]) as f:
            lines = f.readlines()
            return self.paths[idx_file], self.paths[idx_file].split('/')[-1], lines[idx_in_file]

        
def process_batch(batch, model, out, rank):
    dirname, filename, articles = batch
    for dname, fname, article_lines in zip(dirname, filename, articles):
        outpath = f'{out}/{fname}'
        if not exists(outpath):
            os.makedirs(outpath, exist_ok=True)
        
        texts = []
        save_paths = []
        
        for article_id, line in enumerate(article_lines):
            item = json.loads(line) 
            embedding_path = f'{outpath}/{fname}_{article_id}.npy'
            texts.append(item['text'])
            save_paths.append(embedding_path)
        
        if texts:
            embeddings = model.encode(texts, device=f'cuda:{rank}', batch_size=len(texts))
            for emb, save_path in zip(embeddings, save_paths):
                np.save(save_path, emb)

def main(path_to_files='files', out='embeddings'):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=idr_torch.size, rank=idr_torch.rank)
    
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1").cuda(idr_torch.rank)
    
    paths = natsort.natsorted(glob.glob(f'{path_to_files}/*.txt', recursive=True))
    print(paths)
    if not exists(out):
            os.makedirs(out, exist_ok=True)

    dataset = ArticleDataset(paths)
    
    sampler = DistributedSampler(dataset, num_replicas=idr_torch.size, rank=idr_torch.rank, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)  # Custom collate_fn
    
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
    parser = argparse.ArgumentParser(
                    prog='create_embeddings_distributed.py',
                    description='Create embeddings for articles in parallel',
                    epilog='Enjoy the program! :)',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path_to_files', type=str, default='files')
    parser.add_argument('--out', type=str, default='embeddings')
    args = parser.parse_args()

    main(path_to_files=args.path_to_files, out=args.out)

