# White-Box Data Reconstruction in Trained ReLU Networks

This repository contains the experimental code for my MSc dissertation project on **provable privacy attacks in trained neural networks**. The project studies whether the parameters of a trained neural network can leak information about the training data.

The setting is deliberately controlled: binary classification with 1-D synthetic data and shallow ReLU networks. In this setting, trained ReLU networks are piecewise-linear, so their breakpoints and margin geometry can be inspected directly. The core experiment trains 2-layer ReLU classifiers, extracts candidate reconstruction points from the trained model, and measures whether those candidates converge to the original training support points.

![Training dynamics](assets/training_dynamics.gif)

## Key idea

After a ReLU network reaches 100% training accuracy, continued training increases the margin and changes the model geometry. The attack uses white-box access to the trained network, scans the piecewise-linear regions, and analytically solves for points where the model output hits the margin. These margin-hit candidates are then compared against the original support points.

## What this repo demonstrates

* A PyTorch pipeline for training 1-D two-layer ReLU classifiers.
* A margin-crossing candidate extractor based on the network’s piecewise-linear structure.
* Experiments comparing feature-learning and lazy/NTK-like regimes.
* Post-training analysis of reconstruction precision, distance-to-support, duplicate candidates, and endpoint robustness.
* Supporting mathematical notes exploring no-false-positive guarantees under No-Flat and path-norm/TV-minimizer assumptions.

## Main takeaway

In feature-learning regimes, the extracted candidates concentrate close to the true support points, suggesting stronger empirical reconstruction than conservative theoretical lower bounds predict. In contrast, lazy/NTK-like training provides a useful negative control, where the same reconstruction behaviour is much weaker.

## Repository structure

```text
src/          Core training, extraction, aggregation, plotting, and statistics scripts
slurm/        HPC job scripts used for running multi-seed experiments
assets/       Selected README figures and animations
docs/         Project summary and supporting theory note
results_sample/ Optional small output sample for reproducibility checks
```

## Reproducibility note

The full dissertation experiments were run on HPC using Slurm arrays. This repository focuses on making the core X.3 two-layer reconstruction experiment understandable and reproducible. Large raw result folders are not included; selected figures and sample outputs are provided instead.
