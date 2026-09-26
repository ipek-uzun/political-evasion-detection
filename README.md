# SemEval-2026 CLARITY Challenge: Task 1

This repository contains our team's submission for **Task 1 of the SemEval-2026 CLARITY Challenge**, focusing on political interview response clarity classification. 

Given a politician's response to a journalist's question, our system predicts one of three labels: Clear Reply, Ambivalent, or Clear Non-Reply.


## Methodology 

1. **Preprocessing:** Minimal cleaning on QEvasion Parquet datasets (Unicode normalization, whitespace cleanup) loaded directly from Hugging Face.
2. **Classical Baselines:** TF-IDF unigram features paired with linear models and XGBoost ensembles.
3. **Transformer Fine-Tuning:** Fine-tuning DeBERTa-v3 variants as sentence-pair classifiers (`[CLS] Question [SEP] Answer [SEP]`) using gradient accumulation and mixed-precision training.

4. ## Results

- **Best Model:** Fine-tuned `DeBERTa-v3-base` achieving a **0.70 Macro F1-score**.
- **Baselines:** Tested classical feature-based models including TF-IDF with Logistic Regression, Linear SVM, Decision Trees, and XGBoost (reaching up to 0.59–0.60 Macro F1).
- **Handling Imbalance:** Applied stratified splitting and class-weighted cross-entropy loss to improve minority-class detection, notably boosting performance on the challenging *Clear Non-Reply* class.

