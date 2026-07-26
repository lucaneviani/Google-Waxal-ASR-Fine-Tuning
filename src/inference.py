import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from peft import PeftModel
from datasets import load_dataset

# Configuration
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
CHECKPOINT_DIR = "./waxal_qwen_output/checkpoint-500" # Update this if you train for more steps!
DATASET_ID = "google/WaxalNLP"
SAMPLE_RATE = 16000

def generate_transcription(audio_array, processor, model, device):
    """Generates the transcription for a single audio array."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "audio_placeholder"}, 
            {"type": "text", "text": "Transcribe the audio."}
        ]}
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(
        text=text,
        audios=[audio_array],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).to(device)
    
    # Generate text with Anti-Hallucination parameters
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=100,        # Reduced to 100 (enough for ASR, prevents infinite loops)
            repetition_penalty=1.15,   # CRITICAL: Stops the model from repeating "akutte akutte akutte"
            do_sample=False            # Use greedy decoding for precise ASR
        )
    
    # Isolate the newly generated tokens
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response

def main():
    print("🚀 Initializing Inference Pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Base Model and Processor
    print("Loading Base Model and Processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # 2. Merge our fine-tuned LoRA weights!
    print(f"Loading LoRA weights from {CHECKPOINT_DIR}...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    model.eval()
    print("✅ Model ready for inference.")

    # 3. Load the test CSV to know which IDs Zindi wants
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_csv_path = os.path.join(script_dir, "Test.csv")
    
    if not os.path.exists(test_csv_path):
        # Fallback to current working directory (e.g., /kaggle/working)
        test_csv_path = "Test.csv"
        
    print(f"Looking for Test.csv at: {test_csv_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Non riesco a trovare Test.csv. Assicurati che il file si trovi esattamente qui: {test_csv_path}")
        
    test_df = pd.read_csv(test_csv_path)
    submission_dict = {}

    # 4. Run inference on the 3 languages
    languages = ["lug", "sna", "lin"]
    
    for lang in languages:
        print(f"\n🎧 Processing language: {lang}")
        # Load the HF test split for this language
        test_ds = load_dataset(DATASET_ID, name=f"{lang}_asr", split="test", streaming=True)
        test_ds = test_ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLE_RATE))
        
        # We process one by one. In a highly optimized setup we would batch this.
        for sample in tqdm(test_ds):
            audio_id = sample['id']
            # Only predict if Zindi asked for this ID
            if audio_id in test_df['ID'].values:
                audio_array = sample['audio']['array']
                transcription = generate_transcription(audio_array, processor, model, device)
                submission_dict[audio_id] = transcription

    # 5. Format Submission
    print("\n📝 Formatting submission file...")
    submission = pd.DataFrame({
        "ID": test_df["ID"],
        "transcription": test_df["ID"].map(submission_dict)
    })
    
    # Fill any missing transcripts with an empty string just in case
    submission.fillna("", inplace=True)
    
    submission.to_csv("submission.csv", index=False)
    print("✅ Done! 'submission.csv' is ready to be uploaded to Zindi.")

if __name__ == "__main__":
    import datasets # Moved here to avoid import error if run globally
    main()
