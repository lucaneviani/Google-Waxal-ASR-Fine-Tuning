# Google WAXAL ASR Challenge 🎙️

## 📝 Project Overview
This repository contains my solution for the [Zindi Google WAXAL ASR Challenge](https://zindi.africa/competitions/google-waxal-asr-challenge). The primary objective of this competition is to build a robust Automatic Speech Recognition (ASR) system capable of transcribing speech in underrepresented African languages (Luganda, Shona, and Lingala). 

This project demonstrates an end-to-end Machine Learning pipeline, focusing on fine-tuning a Large Audio-Language Model to achieve high accuracy in low-resource language environments.

## 🧠 Methodology & Technical Approach

### 1. Model Selection
Instead of relying on standard text-only LLMs or older ASR architectures, this project leverages **Qwen3-ASR-1.7B**, a state-of-the-art model specifically designed for speech recognition tasks. Its native multimodal architecture makes it exceptionally capable of handling acoustic variations and complex dialects.

### 2. Fine-Tuning Strategy (PEFT/LoRA)
Given the substantial size of the model and the computational constraints, I applied **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA (Low-Rank Adaptation)**. 
- Only the essential projection matrices within the attention mechanism are updated.
- This approach drastically reduces the VRAM required and training time, while preventing catastrophic forgetting of the model's pre-trained multilingual knowledge.

### 3. Data Processing Pipeline
- **Audio Preprocessing:** Audio signals are dynamically resampled and normalized to match the specific input requirements of the Qwen3 audio processor.
- **Text Normalization:** Ground-truth transcripts are rigorously cleaned (removing redundant punctuation and standardizing casing) to ensure the model focuses purely on phonetic-to-text mapping, which stabilizes the loss convergence.

### 4. Evaluation Metrics
The model is evaluated using a combination of:
- **WER (Word Error Rate):** To measure the overall word-level accuracy.
- **CER (Character Error Rate):** Crucial for agglutinative African languages where single character mistakes can change the meaning of a word.

## 📂 Repository Structure
```
.
├── portfolio_showcase/                   # Interactive Web Case Study & Portfolio Showcase
├── eda_outputs/                          # Exploratory Data Analysis visualizations
├── src/                                  # Modular Python scripts for data loading and inference
├── EDA_Report.md                         # Detailed Exploratory Data Analysis report
├── fine_tuning_qwen3.ipynb               # End-to-end Qwen3-ASR fine-tuning and inference pipeline
├── inference_qwen3.ipynb                 # Standalone inference and evaluation notebook
├── Waxal_Challenge_Starter_Code.ipynb    # Baseline competition notebook
└── README.md                             # Project documentation
```

## 🚀 How to Reproduce
1. Clone the repository and ensure you have the required dependencies installed (PyTorch, Transformers, PEFT, Librosa, datasets).
2. Download the competition dataset from Zindi and place it in the `data/` folder.
3. Run the Jupyter Notebook to initiate the data processing and the LoRA fine-tuning loop.

## 🛠️ Tech Stack
- **Python 3**
- **PyTorch**
- **Hugging Face Ecosystem** (`transformers`, `peft`, `datasets`, `evaluate`)
- **Librosa** (Audio Processing)

---
*Created as part of the Zindi Google Waxal ASR Challenge.*
