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
import subprocess
from os.path import exists
from sentence_transformers import SentenceTransformer
# from torch.utils.data import Dataset, DataLoader, DistributedSampler
# from tqdm.auto import tqdm
import threading
from queue import Queue

import warnings
warnings.filterwarnings("ignore")

os.environ['SENTENCE_TRANSFORMERS_HOME'] = "/lustre/fswork/projects/rech/esq/udg63qz/knowladge/knowledge_processing/models/"

# Global queue to hold saving tasks
save_queue = Queue()

def get_env(ngpus):

    def _local_world_rank():
        local_rank = os.environ.get('LOCAL_WORLD_SIZE', False)
        if local_rank: return int(local_rank)
        lws = os.environ["SLURM_STEP_TASKS_PER_NODE"]
        lws = lws.split("(")[0]
        return int(lws)

    get = lambda key: os.environ.get(key, False)

    rank = int(get('RANK') or get('SLURM_PROCID'))
    local_rank = int(get('LOCAL_RANK') or get('SLURM_LOCALID'))
    num_nodes = _local_world_rank() 
    num_tasks = int(get('WORLD_SIZE') or get('SLURM_STEP_NUM_TASKS'))
    is_master = bool(local_rank == 0)
    world_size = num_nodes * ngpus
    assert world_size == num_tasks
    return rank, local_rank, num_nodes, num_tasks, is_master, world_size 

def setup_distributed_training(world_size, rank):
    """ find a common host name on all nodes and setup distributed training """
    # make sure http proxy are unset, in order for the nodes to communicate
    for var in ['http_proxy', 'https_proxy']:
        if var in os.environ:
            del os.environ[var]
        if var.upper() in os.environ:
            del os.environ[var.upper()]
    # get distributed url 
    cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
    stdout = subprocess.check_output(cmd.split())
    host_name = stdout.decode().splitlines()[0]
    dist_url = f'tcp://{host_name}:9000'
    # setup dist.init_process_group
    dist.init_process_group(backend='nccl', init_method=dist_url,
    world_size=world_size, rank=rank)

def save_worker():
    while True:
        task = save_queue.get()
        if task is None:
            break
        path, url, embedding = task
        torch.save({ 'url': url, 'embedding': embedding }, path)
        save_queue.task_done()

def make_dataloader(path, batch_size, rank=0, num_task=1):
    n_files = len(glob.glob(f"{path}/*tar"))
    assert n_files > 0, 'tar files not found'
    start, end = 1, n_files
    files_range = f"{start:06d}..{end:06d}"
    path_tar_files = f"{path}/data-{{{files_range}}}-{n_files:06d}.tar"

    def make_sample(sample):
        # Decode the 'txt' field from bytes to string
        article_json = sample['txt'].decode('utf-8')
        # Parse the JSON string into a Python dictionary
        article = json.loads(article_json)
        return sample['__key__'], article['url'], article['text']

    def shard_filter(src):
        for i, item in enumerate(src):
            if i % num_task == rank:
                yield item

    dataset = wds.DataPipeline(
        wds.SimpleShardList(path_tar_files),
        wds.shuffle(100), # Helps prevent workers from stalling due to sequential reads.
        shard_filter,
        wds.tarfile_to_samples(),
        wds.map(make_sample),
        wds.batched(batch_size)
    )

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=20,
    )

    return dataloader


def main(input_dir, output_dir, batch_size, is_distributed=True):

    # create output dir
    os.makedirs(output_dir, exist_ok=True)

    if is_distributed:
        ngpus = 1 # default is each worker has 1 gpu
        (
            rank, local_rank, num_nodes, 
            num_tasks, is_master, world_size 
        ) = get_env(ngpus)
        setup_distributed_training(world_size, rank)

    # Start the save worker thread
    save_thread = threading.Thread(target=save_worker, daemon=True)
    save_thread.start()

    dataloader = make_dataloader(input_dir, batch_size, local_rank, num_tasks)
    print(f"Dataset loaded on {local_rank}")

    total_files = 6846434

    model = SentenceTransformer(
        "multi-qa-mpnet-base-dot-v1", 
        local_files_only=True,
        tokenizer_kwargs={'clean_up_tokenization_spaces': False},
    )
    model = model.half()

    if torch.cuda.is_available():
        model = model.cuda(local_rank)
    model = model.eval()
    model = torch.compile(model)
    print(f"Model loaded on rank {local_rank}")

    total_batches = total_files // batch_size

    if local_rank == 0:
        print(f"total json to process: {total_files} json")
        print(f"total batches to process: {total_batches} json")

    time_by_batch = []
    chrono = time.time()
    for batch_idx, batch in enumerate(dataloader):

        filenames, urls, articles = batch
        embeddings = model.encode(articles, batch_size=len(articles))

        # send the processed batch to the save thread
        for filename, url, embedding in zip(filenames, urls, embeddings):
            embedding_path = f'{output_dir}/{filename}.pth'   
            save_queue.put((embedding_path, url, embedding))

        seconds_per_batch = time.time() - chrono
        examples_per_second = batch_size / seconds_per_batch
        examples_per_second *= world_size
        chrono = time.time()

        if local_rank == 0 and batch_idx > 2 and batch_idx < 20:
            print(f'files/secs: {examples_per_second}')
            time_by_batch.append(examples_per_second)

        if local_rank == 0 and batch_idx == 20:
            avg_examples_sec = np.mean(time_by_batch) 
            total_seconds = total_files / avg_examples_sec
            n_days = total_seconds // 86400
            n_hours = (total_seconds % 86400) / 3600
            print('Approximated time: {:.0f} days and {:.1f} hours'.format(
                    n_days, n_hours))


    # Wait for all saving tasks to complete
    save_queue.join()

    # Stop the save worker
    save_queue.put(None)
    save_thread.join()

    if is_distributed:
        dist.barrier()

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
    print(args)
    main(args.input_dir, args.output_dir, args.batch_size, True)
