# Google WAXAL ASR Challenge - ASR Model Fine-Tuning 🎙️

[![🌐 View the project](https://img.shields.io/badge/🌐_View_Live-Interactive_Case_Study-00F0FF?style=for-the-badge)](https://lucaneviani.github.io/Google-Waxal-ASR-Fine-Tuning/)

## 📝 Project Overview
This repository contains my solution for the [Zindi Google WAXAL ASR Challenge](https://zindi.africa/competitions/google-waxal-asr-challenge). The primary objective of this competition is to build a robust Automatic Speech Recognition (ASR) system capable of transcribing speech in underrepresented African languages (Luganda, Shona, and Lingala). 

An end-to-end speech recognition project focused on fine-tuning Qwen3-ASR (1.7B parameters), an ASR model, to improve transcription performance on low-resource African languages, including Lingala, Shona, and Luganda.

Developed as part of the Google Waxal ASR Challenge, the project addresses the challenge of adapting modern speech recognitioning models to languages with limited annotated data. I built a complete training pipeline covering data preprocessing, audio preparation, and model optimization, leveraging Parameter-Efficient Fine-Tuning (PEFT) with LoRA to efficiently adapt the model while reducing computational costs.

The project was developed using cloud-based GPU infrastructure, enabling scalable training, experimentation, and optimization of large-scale speech recognition models.

## 🧠 Methodology & Technical Approach

### 1. Model Selection
Instead of relying on standard text-only LLMs or older ASR architectures, this project leverages **Qwen3-ASR-1.7B**, a state-of-the-art model specifically designed for speech recognition tasks. Its native multimodal architecture makes it capable of handling acoustic variations and complex dialects.

 ### 2. Fine-Tuning Strategy (PEFT & LoRA)
Retraining all 1.7 billion parameters of Qwen3 from scratch would require expensive supercomputer clusters. To make training
efficient and accessible, I used **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA (Low-Rank Adaptation)**.
- **How it works:** Instead of updating the entire network, LoRA **freezes the original model's general knowledge** and
  attaches small, trainable adapter layers. I trained only 1.8% of the model's total weights.
- **Why it matters:** This drastically reduces computational power and GPU memory requirements by over 80%, allowing the entire training and evaluation pipeline to run smoothly in a cloud GPU environment.

### 3. Data Processing Pipeline
- **Audio Standardization:** Audio recordings naturally come in different durations and qualities. The pipeline automatically resamples every recording to a standardized frequency of **16,000 Hz**, which is the required acoustic standard for neural
networks to process human speech.
- **Text Cleaning:** The written text transcripts are cleaned by removing unnecessary punctuation and standardizing letters  to lowercase. This ensures the model focuses purely on connecting spoken sounds to written words without getting confused by grammar formatting.

### 4. Evaluation Metrics
The model is evaluated using a combination of:
- **Word Error Rate (WER):** Measures the percentage of complete words that were transcribed incorrectly. A lower WER means the model captures spoken sentences with higher precision.
- **Character Error Rate (CER):** Measures spelling accuracy letter-by-letter. This is especially crucial for African languages, where words often have complex prefixes and suffixes, and even a single incorrect letter can change the entire meaning of a word.

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
- **Cloud Computing Environment** 


