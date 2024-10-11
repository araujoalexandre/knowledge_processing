import os
import argparse
import torch
import time
import numpy as np
import tarfile
import io
from queue import Queue
from torch.utils.data import DataLoader
from core.models.models import load_model
from core.data.datasets import TarIterableDataset
from core.utils import setup_logging
from core.utils import MessageBuilder

def compute_embeddings(args: argparse.Namespace) -> None:
    """
    Main function to compute embeddings for input data.
    
    Args:
        args: Parsed command-line arguments
    """

    tar_filename = os.path.basename(args.tar_file)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    logger = setup_logging(args)
    message = MessageBuilder()

    # Load and prepare the model
    model = load_model(args.model_name)
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    model.eval()
    if args.float16:
        model = model.half()
    model = torch.compile(model)
    
    # Prepare the dataset and dataloader
    dataset = TarIterableDataset(args.tar_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    total_files = len(dataset)
    total_batches = len(dataloader)
    logger.info(f"Processing {tar_filename} with {total_files} samples in {total_batches} batches")

    results = []

    with torch.no_grad():
        time_by_batch = []
        chrono = time.time()
        
        # Process each batch
        for batch_idx, batch in enumerate(dataloader):
            # Compute embeddings for the batch

            # get the batch data    
            filenames, urls, documents = batch
            embeddings = model.encode(
                documents, batch_size=len(documents),
                show_progress_bar=False
            )

            # Queue the processed batch for saving
            for filename, url, document, embedding in zip(filenames, urls, documents, embeddings):
                embedding_path = f'{args.output_dir}/{filename}.pth'
                results.append((filename, url, document, embedding))

            # Calculate processing speed
            seconds_per_batch = time.time() - chrono
            examples_per_second = args.batch_size / seconds_per_batch
            chrono = time.time()

            # Collect timing data after initial warm-up
            if batch_idx > 2:
                time_by_batch.append(examples_per_second)

            # Update and log progress estimation every 20 batches
            if batch_idx % 20 == 0 and batch_idx > 0:
                avg_examples_sec = np.mean(time_by_batch)
                processed_examples = (batch_idx + 1) * args.batch_size
                remaining_examples = total_files - processed_examples
                remaining_seconds = remaining_examples / avg_examples_sec
                n_days = remaining_seconds / 86400
                n_hours = remaining_seconds / 3600
                n_minutes = remaining_seconds / 60

                # Build and log progress message
                message.add('Filename', tar_filename)
                message.add('Batch', batch_idx+1, width=5, format='.0f')
                message.add('examples/sec', avg_examples_sec, format='.2f')
                if int(n_days) > 0:
                    message.add('Est. remaining time (days)', n_days, format='.1f')
                elif int(n_hours) > 0:
                    message.add('Est. remaining time (hour)', n_hours, format='.1f')
                else:
                    message.add('Est. remaining time (min)', n_minutes, format='.1f')
                message.add('Progress (%)', processed_examples / total_files * 100, format='.2f')
                logger.info(message.get_message())

    # Save all results in a tar file
    output_tar_name = os.path.basename(tar_filename).replace('.tar', '_embeddings.tar')
    output_tar_path = os.path.join(args.output_dir, output_tar_name)
    
    with tarfile.open(output_tar_path, 'w') as tar:
        for filename, url, document, embedding in results:
            # Create a BytesIO object to store the data
            data = io.BytesIO()
            # Save the embedding and URL
            torch.save({'url': url, 'document': document, 'embedding': embedding}, data)
            data.seek(0)
            
            # Create a TarInfo object
            info = tarfile.TarInfo(name=f"{filename}.pth")
            info.size = len(data.getvalue())
            
            # Add the file to the tar archive
            tar.addfile(info, data)
    
    logger.info(f"Saved embeddings to {output_tar_path}")
    logger.info(f"Done. Processed {tar_filename} with {total_files} samples in {total_batches} batches")

