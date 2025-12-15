# Radiomics Top-M Gated Pipeline

This repository trains a radiomics + gating + logistic regression pipeline. The gating network produces per-feature logits that are filtered through a 3-stage curriculum to enforce **exactly M radiomics features** per sample at inference.

## Top-M gating schedule
- **Stage 1 (warm-up):** soft sigmoid weights with L1 sparsity.
- **Stage 2 (exploration):** Gumbel-TopM + straight-through estimator (STE), with temperature and noise annealing.
- **Stage 3 (hard):** deterministic TopM with STE gradients; no Gumbel noise.

The classifier remains a linear logistic regression head applied on masked radiomics features. During evaluation/inference the mask is always a deterministic Top-M selector (optionally scaled by the sigmoid outputs when `use_continuous_weight_on_selected` is enabled).

## How to run
1. Configure `gating_config` in `classifier_training_acl.py` to set:
   - `enabled`: turn the Top-M curriculum on/off (keeps backward compatibility when `False`).
   - `top_m`: number of radiomics features to keep per sample.
   - `use_continuous_weight_on_selected`: multiply the hard mask by sigmoid logits for smoother weighting.
   - `alpha_l1` / `sparsity_enabled`: sparsity regularization strength and toggle.
   - `stage_ratios`, `tau_schedule`, `lam_schedule`: stage durations plus temperature/noise annealing.
2. Instantiate the model with `CNNWithGlobalMasking(..., gating_config=gating_config)`.
3. Train with `train_cnn_with_masking(..., gating_config=gating_config)`. The loop prints the current stage, temperature, noise level, and average selected features each epoch.
4. Evaluate with `evaluate_model(..., gating_config=gating_config)`, which enforces deterministic Top-M masking.

## Radiomics normalization
Radiomics features are z-scored with mean/std computed on the training fold before splitting into train/validation subsets. Reuse the saved stats for validation/test to keep gating decisions scale-invariant.
