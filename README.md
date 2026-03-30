# NVIDIA Nemotron Model Reasoning Challenge (Score: 0.64)

This repository contains the end-to-end pipeline for the **NVIDIA Nemotron Model Reasoning Challenge**. The project focuses on enhancing the logical consistency and mathematical accuracy of the **Llama-3.1-Nemotron-70B-Instruct** model through advanced reasoning labels and optimized fine-tuning.

## Performance Summary
* **Final Public Score**: 0.64
* **Base Model**: Llama-3.1-Nemotron-70B-Instruct
* **Methodology**: Supervised Fine-Tuning (SFT) with Chain-of-Thought (CoT) Distillation.

## Core Features & Uniqueness
What sets this implementation apart is the focus on **Reasoning-Aware Alignment**:

* **CoT Label Generation**: Utilized a specialized "Reasoning-Extractor" to generate high-quality Chain-of-Thought steps for the training set, forcing the model to "think" before providing a final answer.
* **Nemotron-Specific Prompting**: Implemented the `Helpful assistant` template optimized for Nemotron's reward model training, ensuring maximum alignment with NVIDIA’s instruction-following standards.
* **Quantized Low-Rank Adaptation (QLoRA)**: Used 4-bit NormalFloat (NF4) quantization to fit the massive 70B parameter model into consumer-grade/enterprise GPU memory without losing reasoning depth.
* **Dynamic Sequence Length**: Configured a 4096-token context window to accommodate long, multi-step mathematical proofs and logical puzzles.

## Hardware & Infrastructure
This pipeline was built to utilize NVIDIA’s most efficient inference and training hardware:

* **Accelerator**: Optimized for **NVIDIA L40S** and **H100** GPUs.
* **Memory Management**: Leveraged `Unsloth` and `Xformers` for 2x faster training and 60% less memory usage.
* **Software Stack**: Python 3.12, CUDA 12.4, and PyTorch 2.5.

## Repository Structure
* `nvidia-nemotron-training-cot-labels.ipynb`: The main training engine, handling data ingestion and SFT.
* `src/reasoning_parser.py`: Logic to extract and format CoT steps from training data.
* `configs/nemotron_70b_sft.yaml`: Hyperparameters for the 0.64 score run.
* `requirements.txt`: Environment dependencies for reproducible results.

## Datasets Used
* **Primary Dataset**: NVIDIA Nemotron Challenge - Model Reasoning Data.
* **Synthetic Labels**: Custom-generated reasoning traces designed to improve logic in complex prompt scenarios.

## Installation
1.  **Clone the Repo**: `git clone https://github.com/InterstellarMC/Nemotron-Reasoning.git`
2.  **Install Dependencies**: `pip install -r requirements.txt`
3.  **Launch Training**: Run the `.ipynb` notebook or execute `python train.py --config configs/nemotron_70b_sft.yaml`

---
*Developed for the NVIDIA Nemotron Model Reasoning Challenge.*
