# Deep Sequence Learning and Transformer Architectures

This repository contains the complete implementation of deep sequence models and customized Transformer blocks from scratch for Problem 1 and Problem 2.

## Repository Directory Structure
README.md
requirements.txt
results.csv
report.pdf
report.docx

problem1/
    problem1.ipynb
    figures/
        class_balance.png
        har_best_confusion_matrix.png
        adding_learning_curves_T200.png
        adding_mse_by_T.png
        gradient_norms.png
        hidden_size_sweep.png
        clipping_ablation.png

problem2/
    problem2.ipynb
    figures/
        positional_encoding.png
    generated_samples.txt


## Notes

- `problem1/problem1.ipynb` is the standalone Problem 1 notebook and has been run top to bottom. It contains custom-cell checks, 3-seed HAR comparison, the adding-task sequence-length sweep, gradient analysis, a hidden-size sweep, and clipping ablation. It downloads UCI HAR if needed and saves figures under `problem1/figures/`.
- `problem2/problem2.ipynb` contains Problem 2. It downloads UCI HAR and Tiny Shakespeare into its own runtime if needed, so it can run separately from Problem 1. Each separate Colab runtime writes its own results.csv; combine those files if you need one joint results table. Context scaling records sampled host RSS growth (and CUDA allocation when available).
- The recorded Problem 1 run used CPU, h = 64, HAR seeds 1/2/3 (5 epochs each), and seed 42 for adding-task/ablation runs (2 epochs). Mean HAR accuracy was 0.6212 (RNN), 0.7350 (LSTM), and 0.8345 (GRU). Detailed metrics and per-epoch timings are in `results.csv`.
- Adding-task test MSE at T = 50/100/200/400 was RNN 0.1617/0.1837/0.1517/0.1714, LSTM 0.1975/0.2286/0.2129/0.2392, and GRU 0.1610/0.1843/0.1531/0.1722. None met the 0.05 solve threshold in this two-epoch run.
- The total wall-clock duration and T4 runtime were not recorded. Run both notebooks in Jupyter or separate Google Colab runtimes with the versions in `requirements.txt`; each runtime keeps its own files and results.csv.
- `report.docx` includes the completed Problem 1 results. `report.pdf` has not been regenerated from this updated Word report.
- The recurrent notebook implements custom recurrent cells from their equations and validates them against PyTorch.
- The transformer notebook builds causal self-attention and positional encodings by hand, without `nn.MultiheadAttention` or `nn.Transformer`.
- Chosen variants (selected for this work, not from the student ID): Problem 1 uses hidden size h = 64 for RNN, LSTM, and GRU; Problem 2 uses width d = 128 with 4 heads; the three-seed runs use seeds 1, 2, and 3.

## Tasks Implemented
**Problem 1**: Custom RNN/LSTM/GRU cells verified against PyTorch, 3-seed Human Activity Recognition classification, Adding Problem runs at T = 50/100/200/400, gradient flow, hidden-size sweep, and clipping ablation.
**Problem 2**: Scaled Dot Product Attention, Multihead Attention, Causal Transformer Character-level LM, and HAR Transformer Classifier.

