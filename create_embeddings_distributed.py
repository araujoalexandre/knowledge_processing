import os
import glob
import json
import natsort
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample
from torch.utils.data import DataLoader, DistributedSampler
import idr_torch  # Import idr_torch for cluster and SLURM integration

# Function to create dataset with InputExample format
def create_sentence_dataset(paths):
    examples = []
    for path in paths:
        with open(path) as f:
            for article_id, line in enumerate(f.readlines()):
                item = json.loads(line)
                if item['text'] == '':
                    continue
                examples.append(InputExample(texts=[item['text']]))
    return examples

def main(batch_size=64):
    # Initialize the process group using idr_torch and SLURM
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=idr_torch.size,
                            rank=idr_torch.rank)

    # Set the GPU device based on local rank
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")
    # Worker 0 will print batch progress and GPU utilization
    if idr_torch.rank == 0:
        print(f'Loading model...')
    # Load the SentenceTransformer model and wrap it in DDP
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1").to(gpu)
    model = DDP(model, device_ids=[idr_torch.local_rank])
    model.eval()  # Set model to evaluation mode
    

    if idr_torch.rank == 0:
        print(f'Getting Paths')
    # Get paths of files
    paths = natsort.natsorted(glob.glob('./outputs/**/*'))

    if idr_torch.rank == 0:
        print(f'{len(paths)} paths loaded.')
        print('Creating Dataset')
    # Create dataset using SentenceTransformer's InputExample
    dataset = create_sentence_dataset(paths)

    if idr_torch.rank == 0:
        print(f'{len(dataset.data)} sentence found.')
    
    # Use DistributedSampler to split dataset between processes
    sampler = DistributedSampler(dataset, num_replicas=idr_torch.size, rank=idr_torch.rank, shuffle=False)
    
    # Use SentenceTransformer's DataLoader
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=4)

    out = './embeddings'

    # Worker 0 will print batch progress and GPU utilization
    if idr_torch.rank == 0:
        start_time = time.time()
        print(f'Starting Infererence for {len(dataloader)} with batch size {batch_size}')
    
    with torch.no_grad():  # Disable gradients for inference
        for batch_idx, batch in enumerate(dataloader):
            all_texts = [example.texts[0] for example in batch]  # Extract all texts from batch

            # Only worker 0 prints batch number and estimated time
            if idr_torch.rank == 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                avg_time_per_batch = elapsed_time / (batch_idx + 1)
                remaining_batches = len(dataloader) - (batch_idx + 1)
                estimated_remaining_time = remaining_batches * avg_time_per_batch

                print(f"Batch {batch_idx + 1}/{len(dataloader)} - Estimated Time Remaining: {estimated_remaining_time:.2f} seconds")

                # Check GPU utilization
                gpu_memory_usage = cuda.memory_reserved(idr_torch.local_rank) / (1024 ** 3)  # in GB
                gpu_memory_total = cuda.get_device_properties(idr_torch.local_rank).total_memory / (1024 ** 3)  # in GB
                gpu_utilization = f"{gpu_memory_usage:.2f} GB / {gpu_memory_total:.2f} GB"
                print(f"GPU {idr_torch.local_rank} Utilization: {gpu_utilization}")

            # Pass the batch of texts through the model
            embeddings = model.module.encode(all_texts, show_progress_bar=False, batch_size=batch_size)
            
            # Save embeddings
            for text, embedding in zip(all_texts, embeddings):
                text_hash = hash(text)  # Create a unique name for the text (or use filename logic as before)
                embedding_path = f'{out}/{text_hash}.npy'
                np.save(embedding_path, embedding)

    dist.destroy_process_group()  # Cleanup


if __name__ == "__main__":
    batch_size = 64  # Adjust batch size
    main(batch_size=batch_size)

