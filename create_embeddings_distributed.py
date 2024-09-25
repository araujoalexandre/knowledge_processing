import os
import glob
import json
import numpy as np
import natsort
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from os.path import exists
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# Dataset class to handle data loading
class TextDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with open(path) as f:
            dirname, filename = path.split('/')[-2:]
            return dirname, filename, f.readlines(), path

# Function to setup the process group for DDP
def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Function to cleanup the process group after training
def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)

    # Load and wrap the SentenceTransformer model for DDP
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1").to(rank)
    model = DDP(model, device_ids=[rank])

    paths = natsort.natsorted(glob.glob('./outputs/**/*'))
    dataset = TextDataset(paths)

    # Create a DistributedSampler to split the data among the processes
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # Create DataLoader with the distributed sampler
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    out = './embeddings'
    
    for batch in dataloader:
        dirname, filename, lines, path = batch
        dirname, filename = dirname[0], filename[0]
        print(f"Rank {rank}: Processing {path[0]}")

        outpath = f'{out}/{dirname}'
        if not exists(outpath):
            os.mkdir(outpath)

        for article_id, line in enumerate(lines[0]):
            embedding_path = f'{outpath}/{filename}_{article_id}.npy'
            if exists(embedding_path): 
                continue

            item = json.loads(line)
            if item['text'] == '':
                continue
            
            # Encode the text using the SentenceTransformer model
            embedding = model.module.encode(item['text'])
            np.save(embedding_path, embedding)

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Use the number of GPUs available
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
