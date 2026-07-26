import re
import datasets
from typing import Dict, Any

# Constants
DATASET_ID = "google/WaxalNLP"
SAMPLE_RATE = 16000

def clean_text(text: str) -> str:
    """
    Normalizes the transcription text by:
    1. Converting to lowercase.
    2. Removing all punctuation, emojis, and special characters.
    3. Normalizing multiple spaces into a single space.
    
    Note: For a production system, digits should be mapped to their spoken 
    word equivalents in Luganda, Shona, or Lingala. Here we remove non-word 
    characters, which effectively drops standalone weird symbols.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove any characters that are not alphanumeric or spaces
    # \w matches any alphanumeric character and the underscore; \s matches whitespace.
    # We remove underscores as well.
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'_', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_batch(batch: Dict[str, list]) -> Dict[str, list]:
    """
    Applies text normalization to a batch of transcriptions.
    """
    batch["transcription"] = [clean_text(t) for t in batch["transcription"]]
    return batch

def load_and_prepare_dataset(language: str, split: str = "train", streaming: bool = True) -> datasets.IterableDataset:
    """
    Loads the Hugging Face dataset for the specified language, resamples the audio,
    and normalizes the transcriptions.
    
    Args:
        language (str): The language code (e.g., 'lug', 'sna', 'lin').
        split (str): Dataset split ('train', 'validation', 'test').
        streaming (bool): If True, streams the dataset to save disk space.
        
    Returns:
        datasets.IterableDataset: The preprocessed dataset ready for modeling.
    """
    print(f"Loading {split} split for language: {language}...")
    
    # Load dataset
    ds = datasets.load_dataset(
        DATASET_ID,
        name=f"{language}_asr",
        split=split,
        streaming=streaming
    )
    
    # Resample audio on the fly
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLE_RATE))
    
    # Apply text normalization
    # If streaming is True, map applies the function on the fly.
    ds = ds.map(preprocess_batch, batched=True, batch_size=32)
    
    print(f"Dataset for {language} ({split}) loaded and preprocessing pipeline attached.")
    return ds

if __name__ == "__main__":
    # Quick test
    print("Testing data loader...")
    test_ds = load_and_prepare_dataset("lug", split="train", streaming=True)
    sample = next(iter(test_ds))
    print("\n--- Sample Output ---")
    print(f"ID: {sample.get('id', 'N/A')}")
    print(f"Original Audio Array Shape: {sample['audio']['array'].shape}")
    print(f"Sampling Rate: {sample['audio']['sampling_rate']}")
    print(f"Normalized Transcription: {sample['transcription']}")
