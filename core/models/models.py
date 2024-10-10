import os
from sentence_transformers import SentenceTransformer

os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Function to load a specified model
def load_model(model_name):
    """
    Load a SentenceTransformer model based on the provided model name.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        Loaded model.

    Raises:
        ValueError: If the specified model is not supported.
    """
    if model_name == "multi-qa-mpnet-base-dot-v1":
        # Load the multi-qa-mpnet-base-dot-v1 model
        model = SentenceTransformer(
            "multi-qa-mpnet-base-dot-v1", 
            local_files_only=True,
            tokenizer_kwargs={'clean_up_tokenization_spaces': False}
        )
    else:
        # Raise an error if the specified model is not supported
        raise ValueError(f"Model {model_name} not found")

    return model

if __name__ == "__main__":

    os.environ['HF_DATASETS_OFFLINE'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'

    model = load_model("multi-qa-mpnet-base-dot-v1")
    print('model loaded and cached')