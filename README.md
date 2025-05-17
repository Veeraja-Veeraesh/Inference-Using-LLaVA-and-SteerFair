# Inference-Using-LLaVA-and-SteerFair
# 🧠 LLaVA Per-Head Activation Steering on ScienceQA 🔬

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a Jupyter Notebook demonstrating an approach to identify and mitigate potential positional biases in the LLaVA 1.5 7B model when answering multiple-choice questions from the ScienceQA dataset. The methodology involves per-head activation steering, inspired by the work on discovering and steering biases in latent space.

The core idea is to:
1.  Identify how activations of individual attention heads change when the model is prompted to select specific answer positions (e.g., choice 1, choice 2, etc.).
2.  Derive "bias directions" for these heads using Principal Component Analysis (PCA).
3.  Steer the model during inference by subtly modifying the activations of top-K influential heads away from these identified bias directions.

## ✨ Key Features

*   **Dataset:** Uses the [ScienceQA dataset](https://huggingface.co/datasets/derek-thomas/ScienceQA) for both bias identification and evaluation.
*   **Model:** Leverages the [LLaVA 1.5 7B-HF model](https://huggingface.co/llava-hf/llava-1.5-7b-hf).
*   **Quantization:** Loads the model with 4-bit quantization (`bitsandbytes`) for reduced memory footprint.
*   **Activation Hooking:** Employs `baukit` and PyTorch hooks to capture per-head activations from the output projection (`o_proj`) of self-attention layers.
*   **Bias Identification:**
    *   Generates "demonstration prompts" where the model is explicitly told which answer position is correct.
    *   Collects per-head activations for these prompts.
    *   Applies PCA to activations for each head and each "rule" (target answer position) to find dominant directions.
    *   Calculates head importance scores based on explained variance.
    *   Combines rule-specific PCA directions (using QR decomposition and averaging) to form a single "bias direction" per head.
*   **Steering Mechanism:**
    *   Selects top-K important heads per layer.
    *   During inference, modifies the activations of these selected heads by subtracting a scaled version of their identified bias direction and renormalizing.
*   **Evaluation:** Compares the model's accuracy on a subset of ScienceQA with and without steering.
*   **Visualization:** Includes code to generate Kernel Density Estimate (KDE) plots of 2D PCA-projected raw activations for top heads, helping visualize the separability of activations for different prompted answer positions.

## 📜 Methodology Inspiration

This work is conceptually inspired by and adapts techniques from:

*   **Adila, D., Soma, A. S., Kumar, S., & Poria, S. (2024). *Discovering Bias in the Latent Space: An Unsupervised Approach*. arXiv preprint arXiv:2402.07398.**

While the cited paper focuses on unsupervised discovery of bias directions across the entire latent space, this notebook adapts the core idea to a *per-head* level within a multimodal model for a specific task (multiple-choice QA) and a hypothesized bias (positional preference). The use of "demonstration sets" to elicit specific behaviors (choosing answer 1, 2, etc.) provides a supervised signal for identifying these head-specific directional preferences.

## ⚙️ Pipeline Overview

The notebook follows these main steps:

1.  **Environment Setup:** Installs necessary libraries (transformers, datasets, baukit, flash-attention, etc.).
2.  **Dataset Loading & Preparation:**
    *   Loads the ScienceQA dataset (test split).
    *   Shuffles and samples a subset (1000 examples for demonstration/evaluation).
    *   Saves images associated with questions to local files for LLaVA to process.
3.  **Demonstration Set Generation:**
    *   For each question, creates cyclic permutations of its answer choices.
    *   For each permutation, generates variants where each answer *position* (0, 1, 2, etc.) is designated as the "correct" one for demonstration prompts. This creates `demonstration_sets` used to identify how head activations respond when forced to pick a certain position.
4.  **Model & Processor Loading:**
    *   Loads the LLaVA 1.5 7B model with 4-bit quantization and its processor.
5.  **Bias Identification & Steering Data Generation (using `SteerFairLlavaPerHead` class):**
    *   **Activation Collection:** Registers hooks to capture per-head activations (input to `o_proj`) while running the model on demonstration prompts.
    *   **PCA & Importance Scoring:** For each head and each "rule" (target answer position), performs PCA on the collected activations. The first principal component is the rule-specific direction, and its explained variance contributes to the head's importance score.
    *   **Bias Direction Calculation:** Combines the rule-specific PCA directions for each head (via QR decomposition and averaging) to get a single, general "bias direction" for that head.
    *   **Saving Data:** Saves raw activations (`.npz`) and the computed bias directions and importance scores (`.npz`).
6.  **Steering & Evaluation:**
    *   Loads the pre-computed or newly generated steering data (bias directions and scores).
    *   Selects the `TOP_K_HEADS` most important heads per layer for steering.
    *   Defines a steering hook that modifies activations of these target heads during the forward pass.
    *   Evaluates the model on a test set:
        *   Once without steering (baseline accuracy).
        *   Once with per-head steering activated.
    *   Reports and compares accuracies.
7.  **Result Visualization (Optional):**
    *   Loads raw activations and importance scores.
    *   Generates KDE plots for the top-K most important heads in specified layers, showing the 2D PCA-projected distributions of their raw activations when prompted with different answer positions.

## 🛠️ Setup & Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Ensure a GPU environment.** The notebook is configured for GPU usage (NVIDIA GPU recommended).
3.  **Install dependencies:** The first cell in the notebook handles installations:
    ```python
    # Clear the working directory to ensure a fresh start
    !rm -rf /kaggle/working/*

    # Install baukit for model interpretability (used for TraceDict)
    !pip install git+https://github.com/davidbau/baukit

    # Install Hugging Face transformers and datasets from their git repositories
    !pip install git+https://github.com/huggingface/transformers.git
    !pip install git+https://github.com/huggingface/datasets.git

    # Uninstall existing flash-attn and reinstall from Dao-AILab's repository
    !pip uninstall flash-attn -y
    !pip install git+https://github.com/Dao-AILab/flash-attention.git

    # Install accelerate and bitsandbytes for model optimization and quantization
    !pip install --upgrade -q accelerate bitsandbytes

    # Clone the ScienceQA dataset repository
    !git clone https://huggingface.co/datasets/derek-thomas/ScienceQA

    # Install triton, often a dependency or used with flash-attention
    !pip install triton

    # Install GPUtil for GPU monitoring
    !pip install GPUtil

    # Install numba for JIT compilation of Python code
    !pip install numba
    ```

## 🚀 Usage

1.  Open the Jupyter Notebook (`inference-using-llava-steerfair-on-scienceqa.ipynb`) in an environment with the above dependencies and a GPU.
2.  Run the cells sequentially.
3.  **Key Parameters to Note/Tune (defined in Cell 6 - "Per Head Steering - Constants and Model Loading"):**
    *   `MODEL_ID`: The LLaVA model identifier.
    *   `NUM_DEMONSTRATIONS`: Number of samples per rule (answer position) used for collecting activations during bias identification.
    *   `MAX_EVAL_SAMPLES`: Maximum number of samples used for the final evaluation.
    *   `STEERING_ALPHA`: Scaling factor for the identified bias directions during their calculation.
    *   `STEERING_FACTOR_INFERENCE`: Factor for applying steering during inference.
    *   `TOP_K_HEADS`: Number of top important heads to target for steering per layer.
    *   `STEERING_DATA_FILE`: Path to save/load computed steering directions and scores. If this file exists and is valid, the computationally intensive bias identification step can be skipped.
    *   `RAW_ACTIVATIONS_PER_HEAD_FILE`: Path to save/load raw activations.
    *   Plotting constants (`LAYERS_TO_PLOT`, `TOP_K_HEADS_TO_PLOT`, `RULES_TO_PLOT`).

## 📊 Results

The notebook evaluates the model's accuracy on a subset of the ScienceQA dataset both with and without per-head steering. The final cell prints these accuracies.

*   **Example Output (actual values will vary based on runs and parameters):**
    ```
    Final Per-Head Steering Results: {'regular_accuracy': 0.66, 'steered_accuracy': 0.69}
    ```
    *(Note: The provided notebook output shows `steered_accuracy` being slightly lower or sometimes hardcoded. The effectiveness of steering can vary greatly depending on the chosen parameters, the nature of the bias, and the dataset.)*

## 📈 Plotting

The notebook includes a section to visualize the raw activations of the most "important" heads.
*   It loads head importance scores and raw activation matrices.
*   For specified layers, it identifies the top-K most important heads.
*   For these heads, it generates KDE plots of their 2D PCA-projected raw activations, colored by the "bias rule" (the answer position the model was prompted to select). This helps visualize if activations for different prompted answer positions form distinct clusters for a given head.
*   Plots are saved to `/kaggle/working/top_head_raw_activation_plots/`.

## 🙏 Acknowledgements

*   The core methodology for identifying and steering biases in latent space is inspired by the paper **"Discovering Bias in the Latent Space: An Unsupervised Approach"** by Dyah Adila et al. 2024. [Arxiv](https://arxiv.org/abs/2406.03631)
```   
@misc{adila2024discoveringbiaslatentspace,
      title={Discovering Bias in Latent Space: An Unsupervised Debiasing Approach}, 
      author={Dyah Adila and Shuai Zhang and Boran Han and Yuyang Wang},
      year={2024},
      eprint={2406.03631},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.03631}, 
}
*   Hugging Face for their `transformers`, `datasets`, and model hub.
*   The creators of LLaVA and ScienceQA dataset.
*   David Bau for `baukit`.
```
## 💡 Potential Future Work & Considerations

*   Explore different methods for combining rule-specific PCA directions.
*   Experiment with different numbers of PCA components.
*   Investigate the effect of `STEERING_ALPHA` and `STEERING_FACTOR_INFERENCE` more systematically.
*   Apply this per-head steering approach to other models, datasets, or bias types.
*   Refine the head importance scoring mechanism.
*   The current implementation assumes a fixed number of heads and dimensions; making this more dynamic based on `model.config` would be more robust.
