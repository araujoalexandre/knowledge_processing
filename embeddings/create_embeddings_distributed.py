import os
import glob
import json
import numpy as np
import natsort
import time
import webdataset as wds
import idr_torch
import torch.distributed as dist
import torch
import argparse
import tarfile
from os.path import exists
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm.auto import tqdm


def make_dataloader(path, batch_size):

    n_files = len(glob.glob(f"{path}/*tar"))
    start, end = 1, n_files
    files_range = f"{start:06d}..{end:06d}"
    path_tar_files = f"{path}/data-{{{files_range}}}-{end:06d}.tar"

    # the sample file is a dict with keys: ['__key__', '__url__', 'txt', '__local_path__']
    # __key__, __url__, __local_path__ are metadata of the tar reading 
    # the content is in 'txt' in bytes, we parse this content with json.loads
    def make_sample(sample):
        article = json.loads(sample['txt'])
        return sample['__key__'], article['url'], article['text']

    # This is the basic WebDataset definition
    trainset = wds.WebDataset(
        path_tar_files, resampled=False, shardshuffle=False, cache_dir=None, nodesplitter=wds.split_by_node)
    trainset = trainset.map(make_sample).batched(batch_size)
    
    return trainset

def process_batch(batch, model, output_dir, device):
    filenames, urls, articles = batch
    embeddings = model.encode(articles, batch_size=len(articles))
    for filename, url, embedding in zip(filenames, urls, embeddings):
        embedding_path = f'{output_dir}/{filename}.pth'   
        torch.save({
          'url': url,
          'embedding': embedding
        }, embedding_path)

def main(input_dir, output_dir, batch_size, is_distributed=False):

    if not exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if is_distributed:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=idr_torch.size,
            rank=idr_torch.rank
        )
    
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1",
        tokenizer_kwargs={'clean_up_tokenization_spaces': False})
    if torch.cuda.is_available():
        model = model.cuda(idr_torch.rank)
    
    dataset = make_dataloader(input_dir, batch_size)
    
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=idr_torch.size, rank=idr_torch.rank, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=None)

    # count the number of files in total in the tar archives
    # this is a bit slow but might be the only way to get the total number of files
    total_files = 0
    tar_files = glob.glob(f'{input_dir}/*.tar')
    for tar_file in tar_files:
        with tarfile.open(tar_file, 'r') as tar:
            total_files += len(tar.getmembers())

    total_batches = total_files / batch_size

    if idr_torch.rank == 0:
        print(f"total files to process: {len(tar_files)} tar files")
        print(f"total json to process: {total_files} json")
        print(f"total batches to process: {total_batches} json")
        progress_bar = tqdm(total=total_batches, desc="Processing")


    for batch_idx, batch in enumerate(dataloader):
        process_batch(batch, model, output_dir, idr_torch.rank)

        if torch.cuda.is_available(): 
            # GPU Memory Tracking
            allocated_memory = torch.cuda.memory_allocated(idr_torch.rank) / (1024 ** 2)  # in MB
            reserved_memory = torch.cuda.memory_reserved(idr_torch.rank) / (1024 ** 2)   # in MB

        if idr_torch.rank == 0:
            progress_bar.update(1)
            if torch.cuda.is_available():
                progress_bar.set_postfix({
                    "Allocated_Mem_MB": f"{allocated_memory:.2f}",
                    "Reserved_Mem_MB": f"{reserved_memory:.2f}",
                })

    if is_distributed:
        dist.barrier()  # Ensure all processes finish before exiting

    if idr_torch.rank == 0:
        progress_bar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='create_embeddings_distributed.py',
        description='Create embeddings for articles in parallel',
        epilog='Enjoy the program! :)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input_dir', type=str, default='./files')
    parser.add_argument('--output_dir', type=str, default='./embeddings')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.batch_size)


