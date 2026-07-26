import os
import argparse
import torch
import numpy as np
import jiwer
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    AutoProcessor, 
    Qwen2AudioForConditionalGeneration, 
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from data_loader import load_and_prepare_dataset

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

@dataclass
class DataCollatorQwenAudio:
    """
    Data collator that processes audio arrays and applies Qwen's specific Chat Template.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = []
        texts = []
        
        for feature in features:
            audio_array = feature["audio"]["array"]
            audios.append([audio_array])
            
            transcription = feature["transcription"]
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": "audio_placeholder"}, 
                    {"type": "text", "text": "Transcribe the audio."}
                ]},
                {"role": "assistant", "content": transcription}
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            
        batch = self.processor(
            text=texts,
            audios=audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch

def preprocess_logits_for_metrics(logits, labels):
    """
    EXTREMELY IMPORTANT FOR OOM PREVENTION!
    Extracts the predicted token IDs from the logits (the argmax) and discards the 
    massive probability tensors before they accumulate in GPU memory.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(pred):
    """
    Computes Word Error Rate (WER) and Character Error Rate (CER).
    """
    # predictions are now token IDs because of preprocess_logits_for_metrics
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id to ignore them
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    # Preds might also have -100 if something weird happens
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Calculate metrics, handling empty strings just in case
    wer = jiwer.wer(label_str, pred_str) if label_str else 1.0
    cer = jiwer.cer(label_str, pred_str) if label_str else 1.0
    combined_score = (0.5 * wer) + (0.5 * cer)

    return {"wer": wer, "cer": cer, "combined_score": combined_score}

def main(test_local=False):
    global processor
    print("🚀 Initializing Qwen2-Audio Training Pipeline...")
    
    print("Loading datasets...")
    if test_local:
        train_ds = load_and_prepare_dataset("lug", split="train", streaming=True).take(2)
        val_ds = load_and_prepare_dataset("lug", split="validation", streaming=True).take(2)
    else:
        train_ds = load_and_prepare_dataset("lug", split="train", streaming=True)
        # Limit validation to 100 samples to save time and memory on Kaggle during eval
        val_ds = load_and_prepare_dataset("lug", split="validation", streaming=True).take(100)

    print("✅ Datasets ready.")

    if test_local:
        print("🟢 Local test successful! You can upload this code to Kaggle.")
        return

    print(f"Loading Model {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✅ LoRA applied.")
    
    data_collator = DataCollatorQwenAudio(processor=processor)

    training_args = TrainingArguments(
        output_dir="./waxal_qwen_output",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2, # Keep eval batch size small
        gradient_accumulation_steps=4,
        eval_accumulation_steps=2,    # OFF-LOAD EVAL LOGITS TO CPU TO PREVENT OOM
        learning_rate=1e-4,
        max_steps=500,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        eval_strategy="steps",
        fp16=True,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics, # NEW ANTI-OOM FIX
    )

    print("🔥 Starting Training...")
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_local", action="store_true", help="Run a micro-test to verify syntax locally")
    args = parser.parse_args()
    main(args.test_local)
