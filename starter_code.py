%%capture
!pip install -q \
    transformers>=4.52.0 \
    datasets \
    peft \
    trl \
    jiwer \
    timm \
    librosa \
    soundfile \
    accelerate \
    bitsandbytes \
    huggingface_hub \
    torchao

print("✅ Dependencies installed.")

import dataclasses
from typing import Final

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Choose your model:
#   "google/gemma-3n-E2B-it"  — ~2B effective params (recommended for T4)
#   "google/gemma-4n-E4B-it"  — ~4B effective params (recommended for A100)
MODEL_ID: Final[str] = "google/gemma-3n-E2B-it"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_ID: Final[str] = "google/WaxalNLP"

# Target language — change to any supported Waxal language code.
# Full list: aka, twi, hau, ibo, yor, ewe, lug, swa, lin, amh, tir, ...
LANGUAGE: Final[str] = "sna"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE: Final[str] = (
    "You are an assistant that transcribes speech accurately."
)
USER_MESSAGE: Final[str] = "Please transcribe this audio."

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE: Final[int] = 16_000

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
LORA_R: Final[int] = 8
LORA_ALPHA: Final[int] = 16
LORA_DROPOUT: Final[float] = 0.0
LORA_TARGET_MODULES: Final[list[str]] = ["v_proj", "o_proj"]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
OUTPUT_DIR: Final[str] = f"./gemma3n-asr-{LANGUAGE}"
PER_DEVICE_TRAIN_BATCH_SIZE: Final[int] = 2
GRADIENT_ACCUMULATION_STEPS: Final[int] = 8
LEARNING_RATE: Final[float] = 1e-3
MAX_STEPS: Final[int] = 500       # Increase for better results (e.g. 3000)
MAX_SEQ_LENGTH: Final[int] = 64
EVAL_STEPS: Final[int] = 100
LOGGING_STEPS: Final[int] = 10
SAVE_STEPS: Final[int] = 100
NUM_VALIDATION_EXAMPLES: Final[int] = 200
RANDOM_SEED: Final[int] = 42

print(f"Model : {MODEL_ID}")
print(f"Language: {LANGUAGE}")
print(f"Output  : {OUTPUT_DIR}")

from huggingface_hub import login

# Option A: interactive login (recommended for Colab)
login()

# Option B: paste token directly (not recommended for shared notebooks)
# login(token="hf_...")

from collections.abc import Mapping, Sequence
from typing import Any

import io
import numpy as np
import librosa
import datasets

# ---------------------------------------------------------------------------
# Chat message formatter
# ---------------------------------------------------------------------------

def format_for_chat(
    example: Mapping[str, Any],
    system_message: str = SYSTEM_MESSAGE,
    user_message: str = USER_MESSAGE,
) -> dict[str, Any]:
    """Converts a decoded example into a chat-formatted message list.

    Args:
        example: Dict with 'audio' (decoded) and 'transcription' keys.
        system_message: System prompt text.
        user_message: User prompt text.

    Returns:
        The example with an additional 'messages' key.
    """
    audio = example["audio"]
    # The 'audio' field is already decoded by the datasets library
    audio_array = audio["array"]

    return {
        **example,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": user_message},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": str(example["transcription"])}
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Batch operators (for ds.map)
# ---------------------------------------------------------------------------

def format_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Formats a batch of decoded examples into chat messages."""
    num = len(batch["transcription"])
    examples = [{k: batch[k][i] for k in batch} for i in range(num)]
    formatted = [format_for_chat(ex) for ex in examples]
    return {k: [ex[k] for ex in formatted] for k in formatted[0]}


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

print(f"Loading WaxalNLP — language: {LANGUAGE}, split: train …")
raw_train = datasets.load_dataset(
    DATASET_ID,
    name=f"{LANGUAGE}_asr",
    split="train",
    streaming=True
)

print(f"Loading WaxalNLP — language: {LANGUAGE}, split: validation …")
raw_val = datasets.load_dataset(
    DATASET_ID,
    name=f"{LANGUAGE}_asr",
    split="validation",
    streaming=True
)

# We cast the audio column to ensure it is resampled to our target SAMPLE_RATE
raw_train = raw_train.cast_column("audio", datasets.Audio(sampling_rate=SAMPLE_RATE))
raw_val = raw_val.cast_column("audio", datasets.Audio(sampling_rate=SAMPLE_RATE))

# Apply chat formatting
train_ds = raw_train.map(format_batch, batched=True, batch_size=32)
val_ds = raw_val.map(format_batch, batched=True, batch_size=32)

print("✅ Dataset loaded and formatted.")

# Preview one example
sample = next(iter(train_ds))
print(f"\nTranscription: {sample['transcription']}")
print(f"Audio shape  : {np.array(sample['audio']['array']).shape}")
print(f"Sample rate  : {sample['audio']['sampling_rate']} Hz")

import timm  # Required by Gemma3n; must be imported before transformers loads it
import torch
import transformers


def load_model_and_processor(
    model_id: str,
) -> tuple[
    transformers.Gemma3nForConditionalGeneration,
    transformers.AutoProcessor,
]:
    """Loads a Gemma3n/4n model and processor from the Hub.

    Args:
        model_id: Hugging Face model ID.

    Returns:
        A (model, processor) tuple.
    """
    processor = transformers.AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"

    model = transformers.Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, processor


print(f"Loading {MODEL_ID} …")
model, processor = load_model_and_processor(MODEL_ID)
print(f"✅ Model loaded — device: {model.device}, dtype: {model.dtype}")

import functools


def _mask_labels(
    labels: torch.Tensor,
    proc: transformers.AutoProcessor,
) -> torch.Tensor:
    """Sets padding and special-token positions to -100 (ignored by CE loss).

    Args:
        labels: Label tensor of shape (batch, seq_len).
        proc: The model processor (must have a tokenizer attribute).

    Returns:
        A cloned label tensor with special positions masked to -100.
    """
    labels = labels.clone()
    tokenizer = proc.tokenizer
    special_attrs = [
        "pad_token_id",
        "image_token_id",
        "audio_token_id",
        "boi_token_id",
        "eoi_token_id",
    ]
    mask_ids = [
        getattr(tokenizer, attr)
        for attr in special_attrs
        if getattr(tokenizer, attr, None) is not None
    ]
    if mask_ids:
        labels[
            torch.isin(labels, torch.tensor(mask_ids, device=labels.device))
        ] = -100
    return labels


def collate_fn(
    examples: Sequence[Mapping[str, Any]],
    proc: transformers.AutoProcessor,
) -> dict[str, Any]:
    """Collates a list of chat-formatted examples into a training batch.

    Args:
        examples: List of dicts with 'messages' and 'audio' keys.
        proc: The model processor.

    Returns:
        Dict with 'input_ids', 'attention_mask', and 'labels' tensors.
    """
    texts = [
        proc.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        for ex in examples
    ]
    audios = [np.asarray(ex["audio"]["array"]).flatten() for ex in examples]

    batch = proc(
        text=texts,
        audio=audios,
        return_tensors="pt",
        padding=True,
    )
    batch = {
        k: v.detach().clone() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    batch["labels"] = _mask_labels(batch["input_ids"], proc)
    return batch


collator = functools.partial(collate_fn, proc=processor)
print("✅ Collator ready.")

import peft
import trl


class _SFTTrainer(trl.SFTTrainer):
    """SFTTrainer with a no-op create_model_card to avoid importlib issues."""

    def create_model_card(self, model_name=None, dataset_name=None, tags=None):
        """No-op override; model card creation is skipped."""
        pass


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------
peft_config = peft.LoraConfig(
    task_type="CAUSAL_LM",
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    use_rslora=False,
    use_dora=False,
)

# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------
training_args = trl.SFTConfig(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
    run_name=f"gemma3n-asr-{LANGUAGE}",
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    max_length=MAX_SEQ_LENGTH, # Updated to use max_length
    packing=False,
    dataloader_num_workers=2,
    seed=RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Prepare split datasets
# ---------------------------------------------------------------------------
shuffled_train = (
    train_ds
    .shuffle(buffer_size=1000, seed=RANDOM_SEED)
    .repeat(None)
)

# Take a fixed validation slice for reproducibility
val_ds_fixed = val_ds.take(NUM_VALIDATION_EXAMPLES)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
trainer = _SFTTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=shuffled_train,
    eval_dataset=val_ds_fixed,
    peft_config=peft_config,
)

print("✅ Trainer initialised. Starting fine-tuning …")
trainer.train()
print("✅ Fine-tuning complete.")

import jiwer

def transcribe_batch(
    batch: dict[str, list[Any]],
    model: transformers.Gemma3nForConditionalGeneration,
    proc: transformers.AutoProcessor,
    device: torch.device,
    max_new_tokens: int = 128,
) -> list[str]:
    """Runs inference on a batch and returns decoded transcriptions."""
    messages_list = [msgs[:-1] for msgs in batch["messages"]]
    audio_list = [np.asarray(a["array"]).flatten() for a in batch["audio"]]

    text_prompts = proc.tokenizer.apply_chat_template(
        messages_list, add_generation_prompt=True, tokenize=False
    )
    inputs = proc(
        text=text_prompts,
        audio=audio_list,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=proc.tokenizer.pad_token_id,
        )

    input_len = inputs.input_ids.shape[1]
    decoded = proc.tokenizer.batch_decode(
        outputs[:, input_len:], skip_special_tokens=True
    )
    return [t.strip() for t in decoded]

def run_evaluation(
    eval_dataset: datasets.IterableDataset,
    model: transformers.Gemma3nForConditionalGeneration,
    proc: transformers.AutoProcessor,
    num_samples: int = 200,
    batch_size: int = 4,
) -> dict[str, float]:
    """Evaluates model on eval_dataset and returns WER / CER."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    references: list[str] = []
    predictions: list[str] = []

    ds = eval_dataset.take(num_samples)
    for batch in ds.batch(batch_size=batch_size):
        preds = transcribe_batch(batch, model, proc, device)
        references.extend(batch["transcription"])
        predictions.extend(preds)

    refs_lower = [r.lower() for r in references]
    preds_lower = [p.lower() for p in predictions]

    return {
        "wer": jiwer.wer(refs_lower, preds_lower),
        "cer": jiwer.cer(refs_lower, preds_lower),
    }

# Load test split for final evaluation
test_ds = (
    datasets.load_dataset(
        DATASET_ID,
        name=f"{LANGUAGE}_asr",
        split="test",
        streaming=True
    )
    .cast_column("audio", datasets.Audio(sampling_rate=SAMPLE_RATE))
    .map(format_batch, batched=True, batch_size=32)
)

print("Running evaluation on test split …")
metrics = run_evaluation(test_ds, model, processor)
print(f"\n📊 Results on {LANGUAGE} test set:")
print(f"   WER : {metrics['wer']:.2%}")
print(f"   CER : {metrics['cer']:.2%}")

