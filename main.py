import argparse
from core.embedding import compute_embeddings
from core.tfidf import compute_tfidf
from core.train_index import train_index
from core.train_model import train_model

def main(args: argparse.Namespace) -> None:
    """
    Main function to execute different modes of operation based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing the mode and other parameters.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    if args.mode == 'embedding':
        compute_embeddings(args)
    elif args.mode == 'tfidf':
        compute_tfidf(args)
    elif args.mode == 'train_index':
        train_index(args)
    elif args.mode == 'train_model':
        train_model(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Embedding generation script")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=['embedding', 'tfidf', 'train_index', 'train_model'],
                        help="Mode of operation: embedding, tfidf, train_index, or train_models")
    parser.add_argument("--ngpus", type=int, default=1, 
                        help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size for processing")
    parser.add_argument("--model_name", type=str, default="multi-qa-mpnet-base-dot-v1", 
                        help="Name of the model to use")
    parser.add_argument("--float16", action="store_true",
                        help="Use half-precision (float16) for model")
    
    # embedding or tfidf mode
    parser.add_argument("--tar_file", type=str, required=True, 
                        help="Path to the input tar file")
    parser.add_argument("--output_dir", type=str, default="./embeddings",
                        help="Directory to save embeddings (for embedding mode)")

    args = parser.parse_args()

    # main function call with the parsed arguments
    main(args)